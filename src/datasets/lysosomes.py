import argparse
from pathlib import Path

import pandas as pd
import numpy as np

def process_lysosomes(save_path: Path, natural: bool = False) -> None:
    """
    Process the lysosomes dataset and save it to the specified path.
    
    Args:
        save_path (Path): Directory to save the processed dataset.
        natural (bool): Whether to process the natural dataset.
    """
    suffix = "_natural" if natural else ""
    df = pd.read_excel("/home/rjo21/Downloads/41587_2022_1618_MOESM3_ESM.xlsx", sheet_name="progen" if not natural else "natural")

    df.rename(columns={"normalized_relative_activity": "label"}, inplace=True)
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    df["split"] = np.random.choice(["train", "val", "test"], size=len(df), p=[0.7, 0.2, 0.1])

    df = df[df["label"] != "-"]
    df[["ID", "sequence", "label", "split", "name"]].to_csv(save_path / f"lysosomes{suffix}.csv", index=False)
    print(f"Lysosomes dataset saved to {save_path}/lysosomes{suffix}.csv")

    df.drop(columns=["label"], inplace=True)
    df.rename(columns={"functional?": "label"}, inplace=True)
    df["label"] = df["label"].astype(int)
    df = df[df["label"] != "-"]
    df[["ID", "sequence", "label", "split", "name"]].to_csv(save_path / f"lysosomes_bin{suffix}.csv", index=False)
    print(f"Lysosomes classification dataset saved to {save_path}/lysosomes_bin{suffix}.csv")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process Lysosomes dataset")
    argparser.add_argument("--save-path", type=Path, required=True, help="Path to the output directory where the processed CSV will be saved")
    argparser.add_argument("--natural", action="store_true", help="Whether to process the natural dataset")
    args = argparser.parse_args()
    process_lysosomes(args.save_path, natural=args.natural)
