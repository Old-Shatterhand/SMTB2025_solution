from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def process_meltome_atlas(save_path: Path) -> None:
    """
    Process the Meltome Atlas dataset and save it as a CSV file.
    
    Args:
        save_path (Path): Directory to save the processed dataset.
    """
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'valid': 'data/valid-00000-of-00001.parquet'}
    train = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["train"])
    train["split"] = "train"
    test = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["test"])
    test["split"] = "test"
    valid = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["valid"])
    valid["split"] = "valid"

    df = pd.concat([train, test, valid])
    df["ID"] = [f"P{i:05d}" for i in range(len(df))]
    df.rename(columns={"target": "label"}, inplace=True)
    df[["ID", "sequence", "label", "split"]].to_csv(save_path / "meltome_atlas.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    process_meltome_atlas(args.save_path)
