from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def process_deeploc_data(save_path: Path):
    df_train = pd.read_parquet("hf://datasets/bloyal/deeploc/deeploc-train.parquet")
    df_train["split"] = "train"
    df_valid = pd.read_parquet("hf://datasets/bloyal/deeploc/deploc-val.parquet")
    df_valid["split"] = "valid"
    df_test = pd.read_parquet("hf://datasets/bloyal/deeploc/deeploc-test.parquet")
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    df["ID"] = [f"P{i:05d}" for i in range(len(df))]
    df.rename(columns={"Sequence": "sequence"}, inplace=True)

    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / 'deeploc2.csv', index=False)
    print("DeepLoc2 10-class dataset saved to", save_path / 'deeploc2.csv')

    df = df[["sequence", "Membrane", "split", "ID"]]
    df.rename(columns={"Membrane": "label"}, inplace=True)
    df.to_csv(save_path / 'deeploc2_bin.csv', index=False)
    print("DeepLoc2 binary dataset saved to", save_path / 'deeploc2_bin.csv')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True)
    args = parser.parse_args()

    process_deeploc_data(args.save_path)
