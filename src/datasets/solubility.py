import argparse
from pathlib import Path

import pandas as pd


def process_solubility(save_path: Path) -> None:
    """
    Process the solubility dataset and save it as a single CSV file.

    Args:
        save_path (Path): The directory where the processed dataset will be saved.
    """
    df_train = pd.read_csv("hf://datasets/proteinea/solubility/solubility_training.csv")
    df_train["split"] = "train"
    df_valid = pd.read_csv("hf://datasets/proteinea/solubility/solubility_validation.csv")
    df_valid["split"] = "valid"
    df_test = pd.read_csv("hf://datasets/proteinea/solubility/solubility_testing.csv")
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test])
    df.rename(columns={"sequences": "sequence", "labels": "label"}, inplace=True)
    df = df[df["sequence"].map(lambda x: len(x) <= 1022)].reset_index(drop=True)
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    df["sequence"] = df["sequence"].map(lambda x: x if x.startswith("M") else "M" + x)

    df[["ID", "sequence", "label", "split"]].to_csv(save_path / "solubility.csv", index=False)
    print("Solubility dataset saved to", save_path / "solubility.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True)
    args = parser.parse_args()

    process_solubility(args.save_path)
