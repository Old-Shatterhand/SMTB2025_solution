import argparse
import requests
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def fetch_uniprot(uniprot_id: str) -> str:
    """
    Fetches the FASTA sequence for a given UniProt ID.
    
    Args:
        uniprot_id (str): The UniProt ID for which to fetch the sequence.
        
    Returns:
        str: The amino acid sequence corresponding to the UniProt ID.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    response.raise_for_status()
    fasta_data = response.text
    sequence = "".join(fasta_data.split("\n")[1:])
    return sequence


def process_binding_data(data_path: Path, save_path: Path) -> None:
    """
    Processes the binding data and saves it to a CSV file.
    
    Args:
        data_path (Path): The path to the input data directory containing the binding residues file.
        save_path (Path): The path to the output directory where the processed CSV will be saved.
    """
    n_data = {"uniprot_id": [], "binding_residues": []}
    with open(data_path / "binding_residues_2.5_small.txt") as f:
        for line in f.readlines():
            idx, aas = line.strip().split("\t")
            n_data["uniprot_id"].append(idx)
            n_data["binding_residues"].append([int(x) for x in aas.split(",")])

    data = {"UniProt_ID": [], "sequence": [], "label": []}
    for uid, value in tqdm(zip(n_data["uniprot_id"], n_data["binding_residues"])):
        try:
            sequence = fetch_uniprot(uid)
            label = ["X"] * len(sequence)
            for pos in value:
                label[pos - 1] = "M"
            label = "".join(label)
            data["label"].append(label)
            data["UniProt_ID"].append(uid)
            data["sequence"].append(sequence)
        except:
            pass
    pd.DataFrame(data).to_csv(save_path / "bind_ligands.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process binding data.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the input data directory.")
    parser.add_argument("--save-path", type=Path, required=True, help="Path to the output directory where the processed CSV will be saved.")
    args = parser.parse_args()

    process_binding_data(args.data_path, args.save_path)
