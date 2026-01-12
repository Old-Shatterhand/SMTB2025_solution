import copy
import time
import pickle
import argparse
from pathlib import Path
from typing import Literal

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM_MODELS = {
    "esm_t6": "facebook/esm2_t6_8M_UR50D",
    "esm_t12": "facebook/esm2_t12_35M_UR50D",
    "esm_t30": "facebook/esm2_t30_150M_UR50D",
    "esm_t33": "facebook/esm2_t33_650M_UR50D",
    "esm_t36": "facebook/esm2_t36_3B_UR50D",
}
EARLY_STOPPING_PATIENCE = 10

class HalfFrozenESM(torch.nn.Module):
    def __init__(self, esm_name, unfreeze: slice, output_logits: bool = False):
        super().__init__()
        self.esm = AutoModel.from_pretrained(esm_name)
        # if unfreeze.start < 0 or unfreeze.start >= self.esm.config.num_hidden_layers or unfreeze.stop < 0 or unfreeze.stop > self.esm.config.num_hidden_layers:
        #     raise ValueError(f"unFreeze slice is out of bounds for {esm_name}. The limits are [0, {self.esm.config.num_hidden_layers}]")
        self.head = torch.nn.Linear(self.esm.config.hidden_size, 1)
        self.output_logits = output_logits
        for i in range(0, self.esm.config.num_hidden_layers):
            for param in self.esm.encoder.layer[i].parameters():
                param.requires_grad = i not in range(unfreeze.start, unfreeze.stop)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)  # CLS token
        out = self.head(pooled_output)
        return out if not self.output_logits else torch.sigmoid(out)


class ProteinDataset(Dataset):
    """A dummy dataset of protein sequences and target regression values."""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_fn(tokenizer, batch):
    """Custom collate function for tokenization and batching."""
    sequences, targets = zip(*batch)
    # Tokenize the batch, ensuring padding and truncation
    tokenized = tokenizer(sequences,
                          return_tensors="pt",
                          padding=True,
                          truncation=True,
                          max_length=128)
    # The targets are converted to a float tensor
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) # [B, 1]
    return tokenized.to(DEVICE), targets.to(DEVICE)


def validation_loop(model, dataloader, loss_fn, device):
    """Runs the validation process and calculates average loss."""
    model.eval()
    val_loss, samples = 0, 0
    
    with torch.no_grad(): # Disable gradient calculations during validation
        for batch, targets in dataloader:
            if hasattr(batch, "input_ids"):
                predictions = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            else:
                predictions = model(batch)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item() * targets.size(0)
            samples += targets.size(0)

    avg_val_loss = val_loss / samples
    return avg_val_loss


def train_loop(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs, device):
    """Runs the selective fine-tuning process with validation."""
    print("\nStarting Training...")
    model.train()
    best_val_loss = float('inf')
    best_model_state = None
    losses = {"train": [], "val": []}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, samples = 0, 0
        start_time = time.time()

        for batch, targets in train_dataloader:
            optimizer.zero_grad()

            if hasattr(batch, "input_ids"):
                predictions = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            else:
                predictions = model(batch)
            loss = loss_fn(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            samples += targets.size(0)

        avg_train_loss = train_loss / samples
        losses["train"].append(avg_train_loss)
        
        # Run Validation after each epoch
        avg_val_loss = validation_loop(model, val_dataloader, loss_fn, device)
        losses["val"].append(avg_val_loss)

        end_time = time.time()
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Time: {end_time - start_time:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    
    return best_model_state, losses


def predict(model, dataloader):
    """Generates predictions for the given dataloader."""
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for batch, targets in dataloader:
            if hasattr(batch, "input_ids"):
                pred = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            else:
                pred = model(batch)
            predictions.extend(pred.squeeze().cpu().numpy())
            labels.extend(targets.squeeze().cpu().numpy())

    return predictions, labels


def routine(data_path: Path, out_folder: Path, model_name: str, unfreeze: tuple, task: Literal["regression", "binary", "classification"], force: bool = False, lr=1e-4):
    out_folder.mkdir(parents=True, exist_ok=True)
    loss_file = out_folder / f"loss_unfrozen_{model_name}_{unfreeze[0]}_{unfreeze[1]}_{lr}.csv"
    result_file = out_folder / f"predictions_unfrozen_{model_name}_{unfreeze[0]}_{unfreeze[1]}_{lr}.pkl"
    if not force and result_file.exists() and loss_file.exists():
        print("Results already exist.")
        return
    
    data = pd.read_csv(data_path)
    if task == "regression":
        labels = data[data["split"] == "valid"]["label"].values
        pred = np.full_like(labels, np.mean(data[data["split"] == "train"]["label"].values))
        print(f"Baseline MSE (mean predictor) on validation set: {np.sqrt(np.mean((labels - pred)**2)):.4f}")

    train_dataset = ProteinDataset(data[data["split"] == "train"]["sequence"].to_list(), data[data["split"] == "train"]["label"].to_list())
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(tokenizer, batch))

    valid_dataset = ProteinDataset(data[data["split"] == "valid"]["sequence"].to_list(), data[data["split"] == "valid"]["label"].to_list())
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: collate_fn(tokenizer, batch))

    test_dataset = ProteinDataset(data[data["split"] == "test"]["sequence"].to_list(), data[data["split"] == "test"]["label"].to_list())
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: collate_fn(tokenizer, batch))

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODELS[model_name])
    model = HalfFrozenESM(ESM_MODELS[model_name], unfreeze=slice(*unfreeze)).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {'params': list(filter(lambda p: p.requires_grad, model.parameters())), 'lr': lr}
    ])
    loss_fn = (lambda preds, targets: torch.sqrt(torch.nn.MSELoss()(preds, targets))) if task == "regression" else torch.nn.BCEWithLogitsLoss() if task == "binary" else torch.nn.CrossEntropyLoss()
    torch.manual_seed(42)

    best_model_state, losses = train_loop(model, train_dataloader, valid_dataloader, loss_fn, optimizer, epochs=100, device=DEVICE)
    model.load_state_dict(best_model_state)
    train_predictions, train_labels = predict(model, train_dataloader)
    valid_predictions, valid_labels = predict(model, valid_dataloader)
    test_predictions, test_labels = predict(model, test_dataloader)

    with open(result_file, "wb") as f:
        pickle.dump(((train_predictions, train_labels), (valid_predictions, valid_labels), (test_predictions, test_labels)), f)
    pd.DataFrame(losses).to_csv(loss_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--out-folder", type=Path, required=True, help="Output folder to save results.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the ESM model to use.")
    parser.add_argument("--unfreeze", type=int, nargs=2, required=True, help="Slice indices to unfreeze layers, e.g., --unfreeze 0 6 to unfreeze first 6 layers.")
    parser.add_argument("--task", type=str, choices=["regression", "binary", "classification"], required=True, help="Type of prediction task.")
    parser.add_argument("--force", action="store_true", help="Force re-computation even if results exist.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
    args = parser.parse_args()

    routine(
        data_path=args.data_path,
        out_folder=args.out_folder,
        model_name=args.model_name,
        unfreeze=(args.unfreeze[0], args.unfreeze[1]),
        task=args.task,
        force=args.force,
        lr=args.lr
    )
