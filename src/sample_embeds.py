import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def sample_dataset(data_path: Path, embed_path: Path, save_path: Path, num_samples: int):
    """
    Sample a dataset from the given data path and embed it using the specified embedding path.
    
    :data_path: path to the dataset
    :embed_path: path to the embedding model
    :num_samples: number of samples to take from the dataset
    """
    df = pd.read_csv(data_path)
    ratio = num_samples / len(df)
    sampled_df = []

    print(f"Sampling {num_samples} entries from {data_path}...")
    for split in df["split"].unique():
        split_df = df[df["split"] == split]
        sampled_split_df = split_df.sample(frac=ratio, random_state=42)
        sampled_df.append(sampled_split_df)
    sampled_df = pd.concat(sampled_df, ignore_index=True)
    sampled_ids = sampled_df["ID"].tolist()

    layers = list(filter(lambda x: x.startswith("layer_"), os.listdir(embed_path)))
    for layer in layers:
        print(f"Sampling embeddings for layer {layer}...")
        (save_path / layer).mkdir(parents=True, exist_ok=True)
        for idx in sampled_ids:
            shutil.copy(embed_path / layer / f"{idx}.pkl", save_path / layer / f"{idx}.pkl")
    
    sampled_df.to_csv(save_path / f"sampled_{data_path.stem}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a dataset and its embeddings.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--embed-path", type=Path, required=True, help="Path to the embedding directory.")
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the sampled data and embeddings.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to take from the dataset.")

    args = parser.parse_args()

    sample_dataset(args.data_path, args.embed_path, args.save_path, args.num_samples)
    print(f"Sampled {args.num_samples} entries from {args.data_path} and saved to {args.save_path}.")
