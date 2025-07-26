from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def process_stability(save_path: Path):
    """Process the fluorescence dataset and save it to the specified path."""
    df_train = pd.read_parquet("hf://datasets/proteinglm/stability_prediction/data/train-00000-of-00001.parquet")
    df_train["split"] = "train"
    df_valid = pd.read_parquet("hf://datasets/proteinglm/stability_prediction/data/valid-00000-of-00001.parquet")
    df_valid["split"] = "valid"
    df_test = pd.read_parquet("hf://datasets/proteinglm/stability_prediction/data/test-00000-of-00001.parquet")
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test])
    df.rename(columns={"seq": "sequence"}, inplace=True)
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    (p := Path(save_path)).mkdir(parents=True, exist_ok=True)
    df[["ID", "sequence", "label", "split"]].to_csv(p / "stability.csv", index=False)
    print(f"Stability dataset saved to {save_path}/stability.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    process_stability(args.save_path)
