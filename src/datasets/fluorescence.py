from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def process_fluorescence(save_path: str):
    df_train = pd.read_csv("hf://datasets/proteinea/fluorescence/fluorescence_train.csv")
    df_train["split"] = "train"
    df_valid = pd.read_csv("hf://datasets/proteinea/fluorescence/fluorescence_valid.csv")
    df_valid["split"] = "valid"
    df_test = pd.read_csv("hf://datasets/proteinea/fluorescence/fluorescence_test.csv")
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test])[["primary", "log_fluorescence", "split"]]
    df.rename(columns={"primary": "sequence", "log_fluorescence": "label"}, inplace=True)
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    (p := Path(save_path)).mkdir(parents=True, exist_ok=True)
    df[["ID", "sequence", "label", "split"]].to_csv(p / "fluorescence.csv", index=False)
    print(f"Fluorescence dataset saved to {save_path}/fluorescence.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    process_fluorescence(args.save_path)
