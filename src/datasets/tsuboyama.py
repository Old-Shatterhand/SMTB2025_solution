import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def process_tsuboyama(protein_gym_dir: Path, save_path: Path):
    dfs = []
    for file in protein_gym_dir.glob("*Tsuboyama*.csv"):
        df = pd.read_csv(protein_gym_dir / file)
        df["source"] = file.stem
        dfs.append(df[["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin", "source"]])
    df = pd.concat(dfs, axis=0, ignore_index=True).reset_index(drop=False)

    df["split"] = np.random.choice(["train", "val", "test"], size=len(df), p=[0.7, 0.2, 0.1])
    df["ID"] = [f"P{i:06d}" for i in range(len(df))]

    df.rename(columns={"DMS_score": "label", "mutated_sequence": "sequence"}, inplace=True)
    df[["ID", "mutant", "sequence", "split", "label", "source"]].to_csv(save_path / "tsuboyama.csv", index=False)
    print(f"Processed Tsuboyama dataset saved to {save_path / 'tsuboyama.csv'}")

    df.rename(columns={"DMS_score_bin": "label"}, inplace=True)
    df[["ID", "mutant", "sequence", "split", "label", "source"]].to_csv(save_path / "tsuboyama_bin.csv", index=False)
    print(f"Processed Tsuboyama dataset saved to {save_path / 'tsuboyama_bin.csv'}")
    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process Tsuboyama dataset")
    argparser.add_argument("--protein-gym-dir", type=Path, required=True, help="Path to the directory containing ProteinGym CSV files")
    argparser.add_argument("--save-path", type=Path, required=True, help="Path to the output directory where the processed CSV will be saved")
    args = argparser.parse_args()

    process_tsuboyama(args.protein_gym_dir, args.save_path)
