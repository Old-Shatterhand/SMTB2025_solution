import copy
import pickle
import time
import argparse
from pathlib import Path
from typing import Any, Literal

import scipy
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.downstream.analyze import build_wp_dataloader, prepare_dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 10

class GlobalProbe(nn.Module):
    def __init__(self, input_dim: int, num_outputs: int, num_layers: int, task: Literal["regression", "binary", "multi-class", "multi-label"]):
        """
        A global probe that learns to weight the contributions of different layers' representations.
        
        Args:
            input_dim (int): The dimensionality of the input representations from each layer.
            num_outputs (int): The number of output classes for classification tasks or 1 for regression
            num_layers (int): The number of layers to consider for probing.
            task (Literal["regression", "binary", "multi-class", "multi-label"]): The type of prediction task being performed, which determines the loss function used during training.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_layers + 1), requires_grad=True)
        self.weight_norm = nn.Softmax(dim=0)
        self.linear = nn.Linear(input_dim, num_outputs)
        self.head = None
        if task == "multi-class":
            self.head = nn.Softmax(dim=1)
        elif task == "multi-label":
            self.head = nn.Sigmoid()

    def forward(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Combines the representations from different layers using learned weights and applies a linear layer for prediction.
        
        Args:
            batch (list[torch.Tensor]): A list of tensors, where each tensor corresponds to the representations from a specific layer 
            for the entire batch. The list should have a length equal to the number of layers being probed (num_layers + 1).
        
        Returns:
            torch.Tensor: The output predictions after combining the layer representations and applying the linear layer.
        """
        pd_weights = self.weight_norm(self.weights)
        weighted_sum = sum(w * layer for w, layer in zip(pd_weights, batch))
        x = self.linear(weighted_sum)
        if self.head is not None:
            x = self.head(x)
        return x


class GlobalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, embed_dir: Path, max_layers: int, label: str, device: torch.device):
        """
        Initializes the dataset by loading the representations for the specified split and preparing the labels.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the dataset information, including splits and labels.
            split (str): The split to load (e.g., "train", "val", "test").
            embed_dir (Path): The directory where the layer representations are stored.
            max_layers (int): The maximum number of layers to consider for probing.
            label (str): The name of the column in the DataFrame that contains the labels.
            device (torch.device): The device to load the tensors onto (e.g., CPU or GPU).
        """
        self.device = device
        self._layer_limit = max_layers + 1
        self._current_limit = self._layer_limit  # The limit up to which layers are currently being used (can be adjusted during training to avoid multiple loads of the data)
        self.data, self.labels = self._load(df[df["split"] == split], embed_dir, label)
    
    def _load(self, df: pd.DataFrame, embed_dir: Path, label: str) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Loads the representations for the specified split and prepares the labels.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the dataset information for the specified split.
            embed_dir (Path): The directory where the layer representations are stored.
            label (str): The name of the column in the DataFrame that contains the labels.
        """
        train_X, labels = build_wp_dataloader(df, embed_dir / "layer_0", label)
        data = [torch.tensor(train_X).to(self.device)] + [
            torch.tensor(build_wp_dataloader(df, embed_dir / f"layer_{i}", label)[0]).to(self.device) for i in range(1, self._layer_limit)
        ]
        labels = torch.tensor(labels).to(self.device)
        return data, labels
    
    def set_layer_limit(self, limit: int) -> None:
        """
        Sets the current layer limit for the dataset, which determines how many layers' representations are used during training.
        
        Args:
            limit (int): The new layer limit to set. Must be between 1 and the maximum layer limit defined during initialization.
        """
        if limit < 1 or limit > self._layer_limit:
            raise ValueError(f"Layer limit must be between 1 and {self._layer_limit}")
        self._current_limit = limit

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.data[0].shape[0]

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        """Returns the representations and label for a given index."""
        return tuple([d[idx] for d in self.data[:self._current_limit]]), self.labels[idx]

    def embed_dim(self) -> int:
        """Returns the dimensionality of the embeddings."""
        return self.data[0].shape[1]


def validation_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module) -> float:
    """Runs the validation process and calculates average loss.
    
    Args:
        model (torch.nn.Module): The model being evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        loss_fn (torch.nn.Module): The loss function to calculate the validation loss.

    Returns:
        float: The average validation loss across all batches.
    """
    model.eval()
    val_loss, samples = 0, 0
    
    with torch.no_grad(): # Disable gradient calculations during validation
        for batch, targets in dataloader:
            predictions = model(batch)
            loss = loss_fn(predictions, targets.reshape(predictions.shape))
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
    ) -> tuple[np.ndarray | None, dict[str, list[float]]]:
    """Runs the selective fine-tuning process with validation.
    
    Args:
        model (torch.nn.Module): The model being trained.
        train_dataloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        loss_fn (torch.nn.Module): The loss function to calculate the training and validation loss.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epochs (int): The number of training epochs.
        
    Returns:
        tuple[np.ndarray | None, dict[str, list[float]]]: The best model weights and the training/validation losses.
    """
    print("\nStarting Training...")
    torch.manual_seed(42)
    model.train()
    best_val_loss = float('inf')
    best_weights = None
    losses = {"train": [], "val": []}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, samples = 0, 0
        start_time = time.time()

        for batch, targets in train_dataloader:
            optimizer.zero_grad()

            predictions = model(batch)
            loss = loss_fn(predictions, targets.reshape(predictions.shape))

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
            best_weights = copy.deepcopy(model.weights.detach().cpu().numpy())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    
    return best_weights, losses


def routine(
        data_path: Path,
        embed_dir: Path,
        num_layers: int,
        num_outputs: int,
        task: Literal["regression", "binary", "class"], 
    ) -> None:
    """Main routine to run the global probing process.
    
    Args:
        data_path (Path): The path to the CSV file containing the dataset.
        embed_dir (Path): The directory where the layer representations are stored and where results will be saved.
        num_layers (int): The number of layers to consider for probing.
        num_outputs (int): The number of output classes for classification tasks or 1 for regression.
        task (Literal["regression", "binary", "class"]): The type of prediction task being performed, which determines the loss function used during training.
    """
    dataset_name = data_path.stem
    df, label, val_name, _, _ = prepare_dataset(dataset_name, data_path, max_rows=100)
    train_dataset = GlobalDataset(df, "train", embed_dir, num_layers, label, device=DEVICE)
    valid_dataset = GlobalDataset(df, val_name, embed_dir, num_layers, label, device=DEVICE)
    weights, losses = [], []
    loss_fn = lambda preds, targets: torch.sqrt(torch.nn.MSELoss()(preds, targets))
    if task == "binary":
        loss = torch.nn.BCEWithLogitsLoss()
    elif task == "multi-class" or task == "multi-label":
        loss = torch.nn.CrossEntropyLoss()
    
    for curr_max_layer in range(1, num_layers + 1):
        print(f"\n=== Training with up to layer {curr_max_layer} ===")
        train_dataset.set_layer_limit(curr_max_layer)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        valid_dataset.set_layer_limit(curr_max_layer)
        valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        model = GlobalProbe(train_dataset.embed_dim(), num_outputs, curr_max_layer, task).to(DEVICE)
        optimizer = torch.optim.Adam([{'params': list(filter(lambda p: p.requires_grad, model.parameters())), 'lr': 0.0001}])

        epoch_weights, epoch_losses = train_loop(model, train_dataloader, valid_dataloader, loss_fn, optimizer, epochs=100)
        weights.append(epoch_weights)
        losses.append(epoch_losses)
    
    print("\n".join([str(list(scipy.special.softmax(x))) for x in weights]))
    with open(embed_dir / f"probe_weights_{dataset_name}_{num_outputs}.pkl", "wb") as f:
        pickle.dump((weights, losses), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--embed-path", type=Path, required=True, help="Output folder to save results.")
    parser.add_argument("--num-layers", type=int, required=True, help="Number of layers in the model to consider for probing.")
    parser.add_argument("--num-outputs", type=int, required=True, help="Number of output classes for classification tasks or 1 for regression.")
    parser.add_argument("--task", type=str, choices=["regression", "binary", "multi-class", "multi-label"], required=True, help="Type of prediction task.")
    args = parser.parse_args()

    routine(
        data_path=args.data_path,
        embed_dir=args.embed_path,
        num_layers=args.num_layers,
        num_outputs=args.num_outputs,
        task=args.task
    )
