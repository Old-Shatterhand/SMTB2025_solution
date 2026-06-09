import pickle
import argparse
import time

import torch
import pandas as pd
import torch.nn as nn
import lightning as L
from transformers import AutoTokenizer
from pathlib import Path

from src.viz.constants import LAYERS, PLM_MODELS


def collate(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    stacked_x = torch.cat(xs, dim=0)
    stacked_y = torch.cat(ys, dim=0)
    return stacked_x, stacked_y


class ESMDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, df: pd.DataFrame, model: str):
        """
        Args:
            path: Path to the directory containing ESM embeddings.
            df: DataFrame containing the dataset with columns "ID" and "labels".
            model: Name of the ESM model to use for tokenization.
        """
        self.data = df
        self.embedding_path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(PLM_MODELS[model])

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the ESM embedding and the corresponding labels for a given index."""
        row = self.data.iloc[idx]
        y = self.tokenizer(row["labels"]).input_ids[1:-1]
        with open(Path(self.embedding_path) / f"{row['ID']}.pkl", "rb") as f:
            embedding = pickle.load(f)
        return torch.tensor(embedding), torch.tensor(y)


class ESMDataModule(L.LightningDataModule):
    def __init__(
        self,
        embed_path: Path,
        df_path: Path,
        model: str,
        layer: int,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.embed_path = embed_path
        self.df = pd.read_csv(df_path)
        self.model = model
        self.layer = layer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Sets up the datasets for training, validation, and testing."""
        self.train_ds = ESMDataset(self.embed_path, self.df[self.df["split"] == "train"], self.model)
        self.val_ds = ESMDataset(self.embed_path, self.df[self.df["split"] == "val"], self.model)
        self.test_ds = ESMDataset(self.embed_path, self.df[self.df["split"] == "test"], self.model)

    def _get_dataloader(self, dataset: ESMDataset, shuffle: bool = True):
        """
        Returns a DataLoader for the given dataset.
        
        Args:
            dataset: The dataset for which to create the DataLoader.
            shuffle: Whether to shuffle the data (default: True).
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        """Returns the DataLoader for the training dataset."""
        return self._get_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        """Returns the DataLoader for the validation dataset."""
        return self._get_dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        """Returns the DataLoader for the test dataset."""
        return self._get_dataloader(self.test_ds, shuffle=False)


class ESMModel(L.LightningModule):
    def __init__(self):
        """A simple feedforward neural network for masked language modeling."""
        super().__init__()
        self.layer1 = nn.LazyLinear(256)
        self.layer2 = nn.Linear(256, 33)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, embedding_dim).
        
        Returns: 
            Output tensor of shape (batch_size, seq_length, vocab_size).
        """
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

    def shared_step(self, batch, step: str):
        """
        Shared step for training, validation, and testing.
        
        Args:
            batch: A batch of data containing input tensors and labels.
            step: A string indicating the current step ("train", "val", or "test").
        
        Returns:
            The computed loss for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log(f"{step}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Defines the training step."""
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Defines the validation step."""
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Defines the test step."""
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for training."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}


def run_single_layer(args: argparse.Namespace, layer: int):
    """
    Runs the training process for a single layer of ESM embeddings.
    
    Args:
        args: Command-line arguments containing paths and training parameters.
        layer: The layer of ESM embeddings to train on.
    """
    logger = L.pytorch.loggers.CSVLogger("logs", version=str(int(time.time() * 1000)))
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=logger,
        gradient_clip_val=1.0,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(monitor="val/loss", mode="min"),
            L.pytorch.callbacks.EarlyStopping(monitor="val/loss", mode="min", patience=30),
        ],
    )
    data_module = ESMDataModule(
        df_path=args.df_path,
        embed_path=args.embed_path / f"layer_{layer}",
        model=args.model,
        layer=layer,
        batch_size=args.batch_size,
    )
    model = ESMModel()
    trainer.fit(model, data_module)
    pd.read_csv(Path(logger.log_dir) / "metrics.csv").to_csv(args.embed_path / f"layer_{layer}" / "mlm.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for masked language modeling on ESM embeddings.")
    parser.add_argument(
        "--df-path",
        type=Path,
        required=True,
        help="Path to the CSV file containing the dataset.",
    )
    parser.add_argument(
        "--embed-path",
        type=Path,
        required=True,
        help="Path to the directory containing ESM embeddings.",
    )
    parser.add_argument("--model", type=str, required=True, help="ESM model name.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs to train.")
    args = parser.parse_args()

    for layer in range(LAYERS[args.model] + 1):
        run_single_layer(args, layer)
