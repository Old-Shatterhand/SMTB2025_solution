from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def process_fluorescence(save_path: Path) -> None:
    """
    Process the fluorescence dataset and save it to the specified path.
    
    Args:
        save_path (Path): Directory to save the processed dataset.
    """
    df_train = pd.read_csv("hf://datasets/proteinea/fluorescence/fluorescence_train.csv")
    df_train["split"] = "train"
    df_valid = pd.read_csv("hf://datasets/proteinea/fluorescence/fluorescence_valid.csv")
    df_valid["split"] = "valid"
    df_test = pd.read_csv("hf://datasets/proteinea/fluorescence/fluorescence_test.csv")
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test])[["primary", "log_fluorescence", "split"]]
    df.rename(columns={"primary": "sequence", "log_fluorescence": "label"}, inplace=True)
    df["ID"] = [f"P{idx:05d}" for idx in range(len(df))]
    df["sequence"] = df["sequence"].apply(lambda x: "M" + x)
    
    save_path.mkdir(parents=True, exist_ok=True)
    df[["ID", "sequence", "label", "split"]].to_csv(save_path / "fluorescence.csv", index=False)
    print(f"Fluorescence dataset saved to {save_path}/fluorescence.csv")


def get_std_boundaries(gmm, n_std: int = 2):
    """Get the n-std boundaries for each component"""
    boundaries = []
    
    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]  # Extract scalar mean
        variance = gmm.covariances_[i, 0, 0]  # Extract scalar variance
        std = np.sqrt(variance)
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        boundaries.append((lower_bound, upper_bound))
    
    return boundaries


def to_classification(save_path: Path) -> None:
    """
    Convert the fluorescence regression dataset to a binary classification dataset
    based on Gaussian Mixture Model (GMM) thresholds and save it.
    
    Args:
        save_path (Path): Directory to save the classification dataset.
    """
    file_path = Path(save_path) / "fluorescence.csv"
    if not file_path.exists():
        process_fluorescence(save_path)
    
    df = pd.read_csv(file_path)
    low_max, high_min = 1.8683750258864091, 3.1064110593867644
    
    def label_to_class(label: float) -> int:
        if label <= low_max:
            return 0
        elif label >= high_min:
            return 1
        else:
            return -1  # Exclude intermediate values

    df["label"] = df["label"].apply(label_to_class)
    df_classification = df[df["label"] != -1][["ID", "sequence", "label", "split"]]
    
    df_classification.to_csv(save_path / "fluorescence_classification.csv", index=False)
    print(f"Fluorescence classification dataset saved to {save_path}/fluorescence_classification.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the processed dataset")
    parser.add_argument("--class", action="store_true", dest="class_", help="Convert to classification dataset")
    args = parser.parse_args()

    if args.class_:
        to_classification(args.save_path)
    else:
        process_fluorescence(args.save_path)