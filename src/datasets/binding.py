import requests
from pathlib import Path

from tqdm import tqdm
import pandas as pd

PATH = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB" / "datasets" / "ionbind"

def fetch_uniprot(uniprot_id: str) -> str:
    """Fetches the FASTA sequence for a given UniProt ID."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    response.raise_for_status()
    fasta_data = response.text
    sequence = "".join(fasta_data.split("\n")[1:])
    return sequence


n_data = {"uniprot_id": [], "binding_residues": []}
with open(PATH / "development_set" / "binding_residues_2.5_small.txt") as f:
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
pd.DataFrame(data).to_csv(PATH.parent / "bind_ligands.csv", index=False)
