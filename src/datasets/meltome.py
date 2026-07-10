import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import load_dataset


def process_meltome_atlas(save_path: Path) -> None:
    """
    Process the Meltome Atlas dataset and save it as a CSV file.
    
    Args:
        save_path (Path): Directory to save the processed dataset.
    """
    # splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'valid': 'data/valid-00000-of-00001.parquet'}
    # train = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["train"])
    # train["split"] = "train"
    # test = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["test"])
    # test["split"] = "test"
    # valid = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["valid"])
    # valid["split"] = "valid"
    # df = pd.concat([train, test, valid])

    species = {
        "Thermus_thermophilus": 0,
        "Picrophilus_torridus": 1,
        "Geobacillus_stearothermophilus": 2,
        "Mus_musculus": 3,
        "Escherichia_coli": 4,
        "Bacillus_subtilis": 5,
        "Saccharomyces_cerevisiae": 6,
        "Drosophila_melanogaster": 7,
        "Danio_rerio": 8,
        "Arabidopsis_thaliana": 9,
        "Caenorhabditis_elegans": 10,
        "Oleispira_antarctica": 11,
        "Homo_sapiens": 12,
    }

    ds = load_dataset("cradle-bio/meltome_cluster_split")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas(), ds["valid"].to_pandas()])
    df["species"] = df["seq_id"].apply(lambda x: "_".join(x.split("_")[1:3]))
    df.rename(columns={"target": "label"}, inplace=True)
    
    df = df[["sequence", "label", "species"]]
    df = df.groupby("sequence").agg({"label": "mean", "species": lambda x: x.values[0]}).reset_index()
    df["ID"] = [f"P{i:05d}" for i in range(len(df))]
    
    df["split"] = df["species"].apply(lambda x: {"Thermus_thermophilus": "valid", "Geobacillus_stearothermophilus": "valid", "Danio_rerio": "test"}.get(x, "train"))
    df[["ID", "sequence", "label", "split"]].to_csv("meltome_atlas_spec_split.csv", index=False)
    print(f"Processed Meltome Atlas w/ cold-species split dataset saved to {save_path / 'meltome_atlas_spec_split.csv'}")
    
    df["split"] = np.random.choice(["train", "test", "valid"], size=len(df), p=[0.7, 0.2, 0.1])
    df[["ID", "sequence", "label", "split"]].to_csv(save_path / "meltome_atlas.csv", index=False)
    print(f"Processed Meltome Atlas dataset saved to {save_path / 'meltome_atlas.csv'}")
    
    df.rename(columns={"label": "Tm"}, inplace=True)
    df["label"] = df["species"].map(species)
    df[["ID", "sequence", "label", "split", "species", "Tm"]].to_csv(save_path / "meltome_atlas_species.csv", index=False)
    print(f"Processed Meltome Atlas species dataset saved to {save_path / 'meltome_atlas_species.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    process_meltome_atlas(args.save_path)
