from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def process_esol_data(save_path: Path) -> None:
    df_train = pd.read_csv("hf://datasets/AI4Protein/eSOL/train.csv")
    df_train["split"] = "train"
    df_valid = pd.read_csv("hf://datasets/AI4Protein/eSOL/valid.csv")
    df_valid["split"] = "valid"
    df_test = pd.read_csv("hf://datasets/AI4Protein/eSOL/test.csv")
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test], ignore_index=True)[["aa_seq", "label", "split"]]
    df["ID"] = [f"P{i:05d}" for i in range(len(df))]
    df.rename(columns={"aa_seq": "sequence"}, inplace=True)
    save_path.mkdir(parents=True, exist_ok=True)
    df[["ID", "sequence", "label", "split"]].to_csv(save_path / "esol.csv", index=False)
    print("eSOL dataset processed and saved.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    process_esol_data(Path(args.save_path))
