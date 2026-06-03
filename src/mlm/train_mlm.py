import torch.nn as nn
import pickle
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import torch

import lightning as L


def collate(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    stacked_x = torch.cat(xs, dim=0)
    stacked_y = torch.cat(ys, dim=0)
    return stacked_x, stacked_y


class ESMDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, df: pd.DataFrame, model: str, layer: int):
        self.data = df
        self.embedding_path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
        self.train_ds = ESMDataset(self.embed_path, self.df[self.df["split"] == "train"], self.model, self.layer)
        self.val_ds = ESMDataset(self.embed_path, self.df[self.df["split"] == "val"], self.model, self.layer)
        self.test_ds = ESMDataset(self.embed_path, self.df[self.df["split"] == "test"], self.model, self.layer)

    def _get_dataloader(self, dataset, shuffle: bool = True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_ds, shuffle=False)


class ESMModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.LazyLinear(256)
        self.layer2 = nn.Linear(256, 33)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

    def shared_step(self, batch, step: str):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log(f"{step}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model for masked language modeling on ESM embeddings.")
    parser.add_argument(
        "--df-path",
        type=Path,
        default="data/mlm/data.csv",
        help="Path to the CSV file containing the dataset.",
    )
    parser.add_argument(
        "--embed-path",
        type=Path,
        default="data/esm_embeddings",
        help="Path to the directory containing ESM embeddings.",
    )
    parser.add_argument("--model", type=str, default="facebook/esm2_t6_8M_UR50D", help="ESM model name.")
    parser.add_argument("--layer", type=int, default=0, help="Layer of ESM embeddings to use.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum number of epochs to train.")
    args = parser.parse_args()

    # trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="auto", devices="auto")
    # trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="auto", devices="auto")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=L.pytorch.loggers.CSVLogger("logs"),
        gradient_clip_val=1.0,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(monitor="val/loss", mode="min"),
            L.pytorch.callbacks.EarlyStopping(monitor="val/loss", mode="min", patience=50),
        ],
    )
    data_module = ESMDataModule(
        df_path=args.df_path,
        embed_path=args.embed_path / f"layer_{args.layer}",
        model=args.model,
        layer=args.layer,
        batch_size=args.batch_size,
    )
    model = ESMModel()
    trainer.fit(model, data_module)
