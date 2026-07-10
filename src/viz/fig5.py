import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from matplotlib import gridspec, pyplot as plt
import pandas as pd

from src.viz.plot_utils import set_subplot_label
from src.viz.utils import compute_metric
from src.viz.constants import MODEL_COLORS, MODEL_MARKERS, DATASET_NAMES, DATASET2TASK

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

def plot_fig5():
    df = pd.read_csv(BASE / "datasets" / "fluorescence.csv")
    low_max, high_min = 1.8683750258864091, 3.1064110593867644

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.hist(df["label"], bins=1001)
    ax.axvline(low_max, color="darkred", linestyle="--")
    ax.axvline(high_min, color="darkred", linestyle="--")
    ax.set_xlabel("log(Fluorescence)")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 300)
    ax.grid()

    plt.tight_layout()
    plt.savefig("paper_figures/5_fluorescence.pdf", dpi=300, bbox_inches="tight")
