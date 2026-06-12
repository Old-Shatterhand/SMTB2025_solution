import shutil
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP


def run_dssp(scope_path: Path, scope_pdb_path: Path) -> None:
    """
    Run DSSP on the SCOPe dataset to extract secondary structure information. 
    The function copies ENT files from the SCOPe dataset to PDB files, runs DSSP on each PDB file, and saves the results in a CSV file.

    Args:
        scope_path (Path): Path to the SCOPe dataset containing ENT files.
        scope_pdb_path (Path): Path to store the PDB files for DSSP processing
    """
    scope_pdb_path.mkdir(parents=True, exist_ok=True)

    for file in tqdm(scope_path.glob("*.ent")):
        shutil.copy(file, scope_pdb_path / (file.stem + ".pdb"))

    data = {"scope_id": [], "secstr_8c": []}
    for pdb_file in tqdm(scope_pdb_path.glob("*.pdb")):
        try:
            dssp = DSSP(PDBParser().get_structure(pdb_file.stem, pdb_file)[0], pdb_file)
        except Exception as e:
            print(f"Failed to process {pdb_file.stem}: {e}")
            data["scope_id"].append(pdb_file.stem)
            data["secstr"].append(None)
            continue
        data["scope_id"].append(pdb_file.stem)
        data["secstr_8c"].append("".join([dssp[a][2] for a in list(dssp.keys())]))
    df = pd.DataFrame(data)
    df["secstr_3c"] = df["secstr_8c"].apply(lambda x: None if x != x else x.replace("G", "H").replace("I", "H").replace("B", "E").replace("S", "C").replace("T", "C").replace("-", "C"))
    df.to_csv(scope_path / "scope_40_dssp_secstr.csv", index=False)
    print(f"DSSP results saved to {scope_path / 'scope_40_dssp_secstr.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSSP on SCOPe dataset")
    parser.add_argument("--scope-path", type=Path, required=True, help="Path to the SCOPe dataset")
    parser.add_argument("--scope-pdb-path", type=Path, required=True, help="Path to store the PDB files for DSSP")
    args = parser.parse_args()

    run_dssp(args.scope_path, args.scope_pdb_path)
