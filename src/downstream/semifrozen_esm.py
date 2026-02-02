import copy
import time
import pickle
import argparse
from pathlib import Path
from typing import Any, Literal

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

class SemiFrozenESM(torch.nn.Module):
    def __init__(self, esm_name: str, unfreeze: slice, n_outputs: int = 1, output_logits: bool = False):
        """
        Initializes the SemiFrozenESM model.

        Args:
            esm_name: Name of the pre-trained ESM model.
            unfreeze: Slice object indicating which layers to unfreeze.
            n_outputs: Number of output neurons for the regression/classification head.
            output_logits: If True, applies sigmoid activation to the output.
        """
        super().__init__()
        self.esm = AutoModel.from_pretrained(esm_name)
        n_layers = self.esm.config.num_hidden_layers
        start = 0 if unfreeze.start is None else unfreeze.start
        stop = n_layers if unfreeze.stop is None else unfreeze.stop
        # if unfreeze.start < 0 or unfreeze.start >= self.esm.config.num_hidden_layers or unfreeze.stop < 0 or unfreeze.stop > self.esm.config.num_hidden_layers:
        #     raise ValueError(f"unFreeze slice is out of bounds for {esm_name}. The limits are [0, {self.esm.config.num_hidden_layers}]")
        self.head = torch.nn.Linear(self.esm.config.hidden_size, n_outputs)
        self.output_logits = output_logits

        # Unfreeze layers according to the unfreeze slice by setting requires_grad
        for param in self.esm.parameters(): # first freeze all
            param.requires_grad = False
        for i in range(0, self.esm.config.num_hidden_layers): # now unfreeze selected
            for param in self.esm.encoder.layer[i].parameters():
                param.requires_grad = i in range(start, stop)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SemiFrozenESM model.
        
        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens should be attended to.
        
        Returns:
            Output tensor after passing through the model and head.
        """
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)  # mean pooling
        out = self.head(pooled_output)
        return out if not self.output_logits else torch.sigmoid(out)


class ProteinDataset(Dataset):
    """A dummy dataset of protein sequences and target regression values."""
    def __init__(self, sequences: list[str], targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


def collate_fn(tokenizer, batch, task: str):
    """Custom collate function for tokenization and batching."""
    sequences, targets = zip(*batch)
    # Tokenize the batch, ensuring padding and truncation
    tokenized = tokenizer(sequences,
                          return_tensors="pt",
                          padding=True,
                          truncation=True,
                          max_length=128)
    if task == "class":
        targets = torch.tensor(np.stack(targets), dtype=torch.float32) # [B, C]
    else: # regression or binary
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # [B,1]
    return tokenized.to(DEVICE), targets.to(DEVICE)


def validation_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module) -> float:
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


def train_loop(
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epochs: int
    ) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Runs the selective fine-tuning process with validation."""
    print("\nStarting Training...")
    model.train()
    best_val_loss = float('inf')
    best_model_state = None
    losses = {"train": [], "val": []}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
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
        avg_val_loss = validation_loop(model, val_dataloader, loss_fn)
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


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[list, list]:
    """
    Generates predictions for the given dataloader.
    
    Args:
        model: The trained model.
        dataloader: DataLoader for the dataset to predict on.
    
    Returns:
        A tuple containing lists of predictions and true labels.
    """
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for batch, targets in dataloader:
            if hasattr(batch, "input_ids"):
                pred = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            else:
                pred = model(batch)
            predictions.extend(pred.detach().cpu().tolist())
            labels.extend(targets.detach().cpu().tolist())

    return predictions, labels


def routine(
        data_path: Path, 
        out_folder: Path, 
        model_name: str, 
        unfreeze: tuple[int, int | None],
        task: Literal["regression", "binary", "class"], 
        force: bool = False, 
        lr: float = 1e-4
    ):
    out_folder.mkdir(parents=True, exist_ok=True)
    unf_tag = f"{unfreeze[0]}_{unfreeze[1] if unfreeze[1] is not None else 'end'}"
    loss_file = out_folder / f"loss_unfrozen_{model_name}_{unf_tag}_{lr}.csv"
    result_file = out_folder / f"predictions_unfrozen_{model_name}_{unf_tag}_{lr}.pkl"
    if not force and result_file.exists() and loss_file.exists():
        print("Results already exist.")
        return
    
    data = pd.read_csv(data_path)
    if task == "class":
        non_label_cols = {"ID", "id", "sequence", "split"}
        label_cols = [c for c in data.columns if c not in non_label_cols]
    if task == "regression":
        labels = data[data["split"] == "valid"]["label"].values
        pred = np.full_like(labels, np.mean(data[data["split"] == "train"]["label"].values))
        print(f"Baseline MSE (mean predictor) on validation set: {np.sqrt(np.mean((labels - pred)**2)):.4f}")

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODELS[model_name])

    if task == "class":
        # targets are vectors 
        y_train = data[data["split"] == "train"][label_cols].values.astype(np.float32)
        y_valid = data[data["split"] == "valid"][label_cols].values.astype(np.float32)
        y_test  = data[data["split"] == "test"][label_cols].values.astype(np.float32)

        train_dataset = ProteinDataset(
            data[data["split"] == "train"]["sequence"].to_list(),
            list(y_train)
        )
        valid_dataset = ProteinDataset(
            data[data["split"] == "valid"]["sequence"].to_list(),
            list(y_valid)
        )
        test_dataset = ProteinDataset(
            data[data["split"] == "test"]["sequence"].to_list(),
            list(y_test)
        )
    else:
        train_dataset = ProteinDataset(
            data[data["split"] == "train"]["sequence"].to_list(),
            data[data["split"] == "train"]["label"].to_list()
        )
        valid_dataset = ProteinDataset(
            data[data["split"] == "valid"]["sequence"].to_list(),
            data[data["split"] == "valid"]["label"].to_list()
        )
        test_dataset = ProteinDataset(
            data[data["split"] == "test"]["sequence"].to_list(),
            data[data["split"] == "test"]["label"].to_list()
        )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(tokenizer, batch, task))
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: collate_fn(tokenizer, batch, task))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: collate_fn(tokenizer, batch, task))

    n_outputs = len(label_cols) if task == "class" else 1
    model = SemiFrozenESM(ESM_MODELS[model_name], unfreeze=slice(unfreeze[0], unfreeze[1]), n_outputs=n_outputs).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {'params': list(filter(lambda p: p.requires_grad, model.parameters())), 'lr': lr}
    ])
    loss_fn = (lambda preds, targets: torch.sqrt(torch.nn.MSELoss()(preds, targets))) if task == "regression" else torch.nn.BCEWithLogitsLoss()
    torch.manual_seed(42)

    best_model_state, losses = train_loop(model, train_dataloader, valid_dataloader, loss_fn, optimizer, epochs=100)
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
    parser.add_argument("--unfreeze", type=int, nargs="+", required=True, help="Unfreeze layers [start, stop). If stop is omitted, unfreezes from start to the last layer. ""Example: --unfreeze 30 33 or --unfreeze 30")
    parser.add_argument("--task", type=str, choices=["regression", "binary", "class"], required=True, help="Type of prediction task.")
    parser.add_argument("--force", action="store_true", help="Force re-computation even if results exist.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
    args = parser.parse_args()
    if len(args.unfreeze) == 1:
        unfreeze = (args.unfreeze[0], None)
    elif len(args.unfreeze) == 2:
        unfreeze = (args.unfreeze[0], args.unfreeze[1])
    else:
        raise ValueError("--unfreeze must have 1 or 2 integers")

    routine(
        data_path=args.data_path,
        out_folder=args.out_folder,
        model_name=args.model_name,
        unfreeze=unfreeze,
        task=args.task,
        force=args.force,
        lr=args.lr
    )
