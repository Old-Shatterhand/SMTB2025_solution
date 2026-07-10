import argparse
from pathlib import Path

import pandas as pd


def process_gb1(save_path: Path) -> None:
    """
    Process the GB1 dataset and save it to the specified path.
    Args:
        save_path (Path): The path to save the processed dataset.
    """
    df = pd.read_csv("hf://datasets/SaProtHub/Dataset-GB1-fitness/dataset.csv")
    df.rename(columns={"protein": "sequence", "stage": "split"}, inplace=True)
    df["ID"] = [f"P{i:06d}" for i in range(len(df))]
    df[["ID", "sequence", "label", "split"]].to_csv(save_path / "gb1.csv", index=False)
    print(f"Processed GB1 dataset saved to {save_path / 'gb1.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    process_gb1(args.save_path)
