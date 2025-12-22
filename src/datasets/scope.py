from argparse import ArgumentParser
from pathlib import Path

from Bio import SeqIO
import numpy as np
import pandas as pd


def seq2rec(seq) -> dict:
    """
    Convert a Bio.SeqRecord to a dictionary with sequence and SCOPe classification levels.

    Args:
        seq (SeqRecord): A Bio.SeqRecord object representing a protein sequence.
    Returns:
        dict: A dictionary containing the sequence and its SCOPe classification levels.
    """
    classes = seq.description.split(" ")[1].split(".")
    return {
        "sequence": str(seq.seq).upper(), 
        "class_": classes[0], 
        "fold": ".".join(classes[:2]), 
        "superfamily": ".".join(classes[:3]), 
        "family": ".".join(classes[:4]), 
        "scope": ".".join(classes), 
        "scope_id": seq.id
    }


def process_scope_40(save_path: Path, secstr_path: Path) -> None:
    """
    Process SCOPe 40 dataset and save to specified path.

    Args:
        save_path (Path): Directory to save the processed dataset.
    """
    # TODO: Download SCOPe_40 from https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa

    with open(save_path / "scope_40_208.fasta", "r") as f:
        scope = list(SeqIO.parse(f, "fasta"))

    data = [seq2rec(seq) for seq in scope]
    df = pd.DataFrame(data)# sorted(dict(df[args.level].value_counts()).items(), key=lambda x: x[1], reverse=True)
    class_map = {c: i for i, (c, _) in enumerate(sorted(dict(df["class_"].value_counts()).items(), key=lambda x: x[1], reverse=True))}
    fold_map = {f: i for i, (f, _) in enumerate(sorted(dict(df["fold"].value_counts()).items(), key=lambda x: x[1], reverse=True))}
    superfamily_map = {sf: i for i, (sf, _) in enumerate(sorted(dict(df["superfamily"].value_counts()).items(), key=lambda x: x[1], reverse=True))}
    family_map = {fam: i for i, (fam, _) in enumerate(sorted(dict(df["family"].value_counts()).items(), key=lambda x: x[1], reverse=True))}
    df["class_"] = df["class_"].apply(lambda c: class_map[c])
    df["fold"] = df["fold"].apply(lambda f: fold_map[f])
    df["superfamily"] = df["superfamily"].apply(lambda sf: superfamily_map[sf])
    df["family"] = df["family"].apply(lambda fam: family_map[fam])
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    df = df.astype({"class_": "int32", "fold": "int32", "superfamily": "int32", "family": "int32"})
    df["split"] = np.random.choice(["train", "val", "test"], size=len(df), p=[0.8, 0.1, 0.1])
    # df = pd.read_csv(save_path / "scope_40_208.csv")
    secstr_df = pd.read_csv(secstr_path)
    df = df.merge(secstr_df[["scope_id", "secstr_8c", "secstr_3c"]], on="scope_id", how="left")

    df[["ID", "sequence", "scope", "class_", "fold", "superfamily", "family", "scope_id", "secstr_8c", "secstr_3c", "split"]].to_csv(save_path / "scope_40_208_test.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the processed dataset")
    parser.add_argument("--secstr-path", type=Path, required=False, help="Path to DSSP secondary structure file")
    args = parser.parse_args()

    process_scope_40(args.save_path, args.secstr_path)