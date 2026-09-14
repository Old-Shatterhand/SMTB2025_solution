import argparse
from pathlib import Path

import pandas as pd
import copy


GB1_WT = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

def generate_mutants(wt, mutants):
    mut_seq = copy.copy(wt)
    for mutant in mutants:
        orig, pos, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
        if mut_seq[pos] != orig:
            raise ValueError(
                f"Original amino acid {orig} does not match the amino acid at position {pos+1} in the wild-type sequence."
            )
        mut_seq = mut_seq[:pos] + mut + mut_seq[pos + 1 :]
    return mut_seq


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


def process_gb1_significant(save_path: Path) -> None:
    """
    Process the GB1 dataset to include only significant mutations and save it to the specified path.
    Args:
        save_path (Path): The path to save the processed dataset.
    """
    mutations = [
        ["M1M"], ["V54A"], ["A24E"], ["G9A"], ["T11A"], ["G41L"],  ["V54G"], ["G9A", "T11A"], ["G41L", "V54G"], ["V39W", "G41F"], # ["V7W", "Y33S"],
        ["D40G", "G41E"], ["L5T", "A26G"], ["Y3H", "L5T"], ["L5T", "I6G"], ["L5T", "D22I"], ["L7P", "T51D"], ["Y3A"], ["Y3C"], ["L5N"], ["L5S"], ["F30N"]
    ]
    seqs = []
    for mutation in mutations:
        seqs.append(generate_mutants(GB1_WT, mutation))
    df = pd.DataFrame({"sequence": seqs, "mutation": ["_".join(m) for m in mutations]})
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    df[["ID", "sequence", "mutation"]].to_csv(save_path / "gb1_mut.csv", index=False)
    print(f"Processed GB1 dataset saved to {save_path / 'gb1_mut.csv'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the processed dataset")
    parser.add_argument("--significant", action="store_true", help="Process only significant mutations")
    args = parser.parse_args()

    if args.significant:
        process_gb1_significant(args.save_path)
    else:
        process_gb1(args.save_path)
