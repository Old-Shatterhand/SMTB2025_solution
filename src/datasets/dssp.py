import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

SCOPe = (Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SCOPe_40")
SCOPe_pdb = (Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SCOPe_40_pdb")
SCOPe_pdb.mkdir(parents=True, exist_ok=True)

for file in tqdm(SCOPe.glob("*.ent")):
    shutil.copy(file, SCOPe_pdb / (file.stem + ".pdb"))

data = {"scope_id": [], "secstr_8c": []}
for pdb_file in tqdm(SCOPe_pdb.glob("*.pdb")):
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
df.to_csv(SCOPe / "scope_40_dssp_secstr.csv", index=False)
