from argparse import ArgumentParser
import re
from pathlib import Path

import pandas as pd


ROOT = Path('/') / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB" / "datasets"
PROTEINNET = Path('/') / "scratch" / "SCRATCH_SAS" / "roman" / "casp7"

class Separator(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5
            self.fall = True
            return True
        else:
            return False


def letter_to_num(string, dict_):
    """ Convert string of letters to list of ints """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num


def read_record(file_, num_evo_entries):
    """ Read a Mathematica protein record from file and convert into dict. """
    
    dict_ = {}

    while True:
        next_line = file_.readline()
        for case in Separator(next_line):
            if case('[ID]' + '\n'):
                id_ = file_.readline()[:-1]
                dict_.update({'id': id_})
            elif case('[PRIMARY]' + '\n'):
                dict_.update({'primary': file_.readline()[:-1]})
            elif case('\n'):
                return dict_
            elif case(''):
                return None


def read_casp_file(filepath):
    data = {}
    with open(filepath, "r") as file_:
        while True:
            d = read_record(file_, 0)
            if d is None:
                break
            data[d['id']] = d["primary"]
    return data


def process_proteinnet(save_path: Path):
    """
    Process the ProteinNet CASP7 dataset and save it to a CSV file.
    
    Args:
        save_path (Path): Directory to save the processed dataset.
    """
    casp_train = read_casp_file(PROTEINNET / "training_100")
    casp_valid = read_casp_file(PROTEINNET / "testing")
    casp_test = read_casp_file(PROTEINNET / "validation")

    dfs = []
    for split, split_dict in [("train", casp_train), ("valid", casp_valid), ("test", casp_test)]:
        df = pd.DataFrame.from_dict(split_dict, orient="index", columns=["sequence"])
        df["split"] = split
        dfs.append(df)
    proteinnet_df = pd.concat(dfs)
    proteinnet_df.reset_index(inplace=True)
    proteinnet_df.rename(columns={"index": "ID"}, inplace=True)
    proteinnet_df.to_csv(save_path / "casp7.csv", index=False)
    print(f"ProteinNet CASP7 dataset saved to {save_path / 'casp7.csv'}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    process_proteinnet(args.save_path)
