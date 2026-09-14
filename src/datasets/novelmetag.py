from pathlib import Path

import numpy as np
import pandas as pd

import argparse
from src.datasets.sample_mlm import mask_sequence


def sample_mlm(
        fasta_path: str, 
        save_path: Path,
        mask_token: str = "<mask>",
        mask_prob: float = 0.15,
        train_size: float = 0.8,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> None:
    """
    Process the NovelMetaG dataset from a FASTA file and save it to the specified path.
    
    Args:
        fasta_path (str): Path to the input FASTA file.
        save_path (Path): Directory to save the processed dataset.
    """
    np.random.seed(random_state)
    sequences = []
    labels = []
    
    with open(fasta_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                label = line.strip()[1:]  # Assuming label is the second word in the header
                labels.append(label)
            else:
                sequence = line.strip()
                sequences.append(sequence)

    data = []
    for seq, meta_label in zip(sequences, labels):
        masked_seq, old_labels, positions = mask_sequence(seq)
        if len(old_labels) > 0:
            data.append(
                {
                    "sequence": masked_seq,
                    "labels": old_labels,
                    "positions": positions,
                    "split": np.random.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1]),
                    "metag_label": meta_label
                }
            )
    
    save_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    df[["ID", "sequence", "labels", "positions", "split", "metag_label"]].to_csv(save_path / "novelmetag.csv", index=False)
    print(f"NovelMetaG dataset saved to {save_path}/novelmetag.csv")


def sample_ntp(
        fasta_path: Path,
        save_path: Path,
        train_size: float = 0.8,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> pd.DataFrame:
    """
    Sample a dataset for NTP (Next Token Prediction).

    Args:
        sequences (pd.Series): A pandas Series containing the sequences to sample from.
        n_samples (int): The number of samples to generate.
        train_size (float): The proportion of the dataset to include in the training split.
        val_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): The random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled sequences, labels, and split information.
    """
    np.random.seed(random_state)
    sequences = []
    labels = []
    
    with open(fasta_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                label = line.strip()[1:]  # Assuming label is the second word in the header
                labels.append(label)
            else:
                sequence = line.strip()
                sequences.append(sequence)

    data = []
    for seq, meta_label in zip(sequences, labels):
        pos = np.random.randint(1, min(len(seq) - 1, 1022))
        data.append({
            "sequence": seq[:pos],
            "labels": seq[pos],
            "metag_label": meta_label
        })

    df = pd.DataFrame(data)
    df["split"] = np.random.choice(["train", "val", "test"], size=len(df), p=[train_size, val_size, 1 - train_size - val_size])
    df["ID"] = [f"P{i:05d}" for i in range(len(df))]
    df[["ID", "sequence", "labels", "split", "metag_label"]].to_csv(save_path / "novelmetag_ntp.csv", index=False)
    print(f"NovelMetaG dataset saved to {save_path}/novelmetag_ntp.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the NovelMetaG dataset from a FASTA file.")
    parser.add_argument("--fasta-path", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--save-path", type=str, help="Directory to save the processed dataset.")
    parser.add_argument("--mask-token", type=str, default="<mask>", help="Token to use for masking.")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Probability of masking a token.")
    parser.add_argument("--train-size", type=float, default=0.8, help="Proportion of training data.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Proportion of validation data.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--ntp", action="store_true", help="Sample for Next Token Prediction instead of MLM.")

    args = parser.parse_args()

    if args.ntp:
        sample_ntp(
            fasta_path=Path(args.fasta_path),
            save_path=Path(args.save_path),
            train_size=args.train_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
    else:
        sample_mlm(
            fasta_path=args.fasta_path,
            save_path=Path(args.save_path),
            mask_token=args.mask_token,
            mask_prob=args.mask_prob,
            train_size=args.train_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
