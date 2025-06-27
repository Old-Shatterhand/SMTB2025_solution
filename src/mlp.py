import pickle
from pathlib import Path

from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MetricCollection, Accuracy, AUROC, \
    MatthewsCorrCoef as MCC, MeanAbsoluteError as MAE, MeanSquaredError as MSE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        self.load_state_dict(torch.load(path))


def build_dataloader(df, embed_path, **dataloader_kwargs):
    embed_path = Path(embed_path)
    embeddings = []
    for idx in df["ID"].values:
        with open(embed_path / f"{idx}.pkl", "rb") as f:
            embeddings.append(pickle.load(f))
    inputs = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    targets = torch.tensor(df['label'].values, dtype=torch.float).to(DEVICE)
    return DataLoader(TensorDataset(inputs, targets), **dataloader_kwargs)


def get_loss(mode):
    if mode == "regression":
        return torch.nn.MSELoss()
    elif mode == "classification":
        return torch.nn.CrossEntropyLoss()
    elif mode == "binary":
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_metrics(mode, num_classes=None):
    if mode == "regression":
        return MetricCollection([MSE(), MAE()])
    elif mode == "classification":
        return MetricCollection([
            Accuracy(task="multiclass", num_classes=num_classes),
            AUROC(task="multiclass", num_classes=num_classes),
            MCC(task="multiclass", num_classes=num_classes),
        ])
    elif mode == "binary":
        return MetricCollection([Accuracy(task="binary"), AUROC(task="binary"), MCC(task="binary")])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def reshape_predictions(predictions, targets, mode):
    if mode == "regression":
        return predictions.reshape(targets.shape)
    elif mode == "classification":  # ??
        return predictions.argmax(dim=1) if len(predictions.shape) > 1 else predictions
    elif mode == "binary":  # ??
        return torch.sigmoid(predictions).round()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def train(input_dim, hidden_dim, output_dim, mode, data_path, embeds_path, log_folder, epochs):
    model = MLP(input_dim, hidden_dim, output_dim).to(DEVICE)
    loss = get_loss(mode).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    df = pd.read_csv(data_path)
    train_dataloader = build_dataloader(
        df[df["split"] == "train"], embed_path=embeds_path, batch_size=32, shuffle=True
    )
    valid_dataloader = build_dataloader(
        df[df["split"] == "valid"], embed_path=embeds_path, batch_size=32, shuffle=True
    )

    metrics = {
        "train": get_metrics(mode, num_classes=output_dim if mode == "classification" else None),
        "valid": get_metrics(mode, num_classes=output_dim if mode == "classification" else None),
    }
    
    metrics_log = {"train": [], "valid": []}
    loss_log = {"train": [], "valid": []}
    for epoch in range(epochs):
        for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (train):"):
            outputs = model(inputs)
            outputs = reshape_predictions(outputs, targets, mode)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["train"].update(outputs.cpu(), targets.detach().cpu())
            loss_log["train"].append(loss.detach().cpu())
        metrics_log["train"].append(metrics["train"].compute())
        metrics["train"].reset()
        
        with torch.no_grad():
            for inputs, targets in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{epochs} (valid):"):
                outputs = model(inputs)
                outputs = reshape_predictions(outputs, targets, mode)
                loss = criterion(outputs, targets)

                metrics["valid"].update(outputs.cpu(), targets.cpu())
                loss_log["valid"].append(loss.detach().cpu())
        metrics_log["valid"].append(metrics["valid"].compute())
        metrics["valid"].reset()
    
    log = Path(log_folder)
    log.mkdir(parents=True, exist_ok=True)
    model.save(log / "model.pth")
    pd.DataFrame(metrics_log["train"]).astype(float).to_csv(log / "train_metrics.csv", index=False)
    pd.DataFrame(metrics_log["valid"]).astype(float).to_csv(log / "valid_metrics.csv", index=False)
    pd.DataFrame(loss_log["train"], columns=["loss"]).astype(float).to_csv(log / "train_loss.csv", index=False)
    pd.DataFrame(loss_log["valid"], columns=["loss"]).astype(float).to_csv(log / "valid_loss.csv", index=False)
    print("Training complete. Model and metrics saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a MLP model.")
    parser.add_argument("--input-dim", type=int, required=True, help="Input dimension of the model.")
    parser.add_argument("--hidden-dim", type=int, required=True, help="Hidden dimension of the model.")
    parser.add_argument("--output-dim", type=int, required=True, help="Output dimension of the model.")
    parser.add_argument("--mode", type=str, choices=["regression", "classification", "binary"], required=True,
                        help="Mode of the model.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data CSV file.")
    parser.add_argument("--embeds-path", type=str, required=True, help="Path to the embeddings directory.")
    parser.add_argument("--log-folder", type=str, required=True, help="Folder to save logs and model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")

    args = parser.parse_args()
    train(args.input_dim, args.hidden_dim, args.output_dim, args.mode, args.data_path, args.embeds_path,
          args.log_folder, args.epochs)