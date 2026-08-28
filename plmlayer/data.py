"""Dataset loading, task inference and the subsampling that makes this cheap.

Column conventions follow the research code
(``src/downstream/probe_layer.py::prepare_dataset``) so existing dataset CSVs in
``src/datasets/`` load unchanged:

* ``ID``, ``sequence`` are required.
* the label column is ``label``, or ``labels`` if ``label`` is absent.
* the split column is ``split``, whose validation level is spelled ``valid`` by
  the HuggingFace-derived datasets and ``val`` by the locally split ones.
* ``positions`` (residue-level only) is a list of 0-based residue indices.

The subsampling is the whole reason this is a tool rather than a cluster job:
the paper's Figure 3E-H shows 15-20% of the training data identifies a layer
reaching >=95% of the best achievable performance, with the same layer picked
across all three seeds.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from plmlayer.types import Task

__all__ = ["Dataset", "load_dataset", "infer_task", "DEEPLOC_LABELS"]

# DeepLoc2.0 stores its 10 classes as separate binary columns.
DEEPLOC_LABELS = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
    "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus",
    "Peroxisome",
]

_VAL_NAMES = ("valid", "val")


@dataclass(slots=True)
class Split:
    ids: list[str]
    sequences: list[str]
    y: np.ndarray
    positions: list[list[int]] | None = None

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(slots=True)
class Dataset:
    name: str
    task: Task
    train: Split
    val: Split
    n_classes: int | None = None
    residue_level: bool = False

    def describe(self) -> str:
        lvl = "residue-level" if self.residue_level else "protein-level"
        cls = f", {self.n_classes} classes" if self.n_classes else ""
        return (
            f"{self.name}: {self.task} ({lvl}{cls}), "
            f"{len(self.train)} train / {len(self.val)} val"
        )


def infer_task(y: np.ndarray, label_cols: list[str]) -> Task:
    """Guess the task from the label column(s). An explicit ``task=`` always wins.

    The tricky case is high-cardinality classification: SCOPe ``fold`` has 1257
    classes, which any "few unique values" rule calls regression. What separates
    it from a real regression target is that ``src/datasets/scope.py`` encodes
    classes as *dense integer codes* -- whole numbers covering nearly every value
    in their range -- which continuous measurements essentially never do.
    """
    if len(label_cols) > 1:
        return "multi-label"

    values = pd.Series(y).dropna().to_numpy()
    if values.dtype == object:
        return "multi-class" if len(np.unique(values)) > 2 else "binary"

    values = values.astype(float)
    uniq = np.unique(values)
    if len(uniq) <= 1:
        return "regression"
    if len(uniq) == 2:
        return "binary"

    whole = np.allclose(uniq, np.round(uniq))
    if whole and uniq.min() >= 0:
        distinctness = len(uniq) / len(values)  # ~1.0 means an ID, not a class
        if distinctness < 0.5 and len(uniq) <= 2000:
            if len(uniq) > 50:
                warnings.warn(
                    f"treating {label_cols[0]!r} as multi-class with {len(uniq)} "
                    "classes because its values are non-negative whole numbers. "
                    "Integer labels are genuinely ambiguous -- pass task="
                    "'regression' explicitly if that is what this column is.",
                    stacklevel=2,
                )
            return "multi-class"
    return "regression"


def _pick_label_columns(df: pd.DataFrame, label_col: str | list[str] | None) -> list[str]:
    if isinstance(label_col, list):
        return label_col
    if label_col:
        return [label_col]
    if all(c in df.columns for c in DEEPLOC_LABELS):
        return list(DEEPLOC_LABELS)
    if "label" in df.columns:
        return ["label"]
    if "labels" in df.columns:
        return ["labels"]
    raise ValueError(
        "no label column found: expected 'label', 'labels', the DeepLoc2.0 "
        "class columns, or an explicit --label-col"
    )


def _subsample(
    df: pd.DataFrame, n: int | None, task: Task, label_cols: list[str], seed: int
) -> pd.DataFrame:
    """Cap a split at ``n`` rows, preserving class balance where meaningful."""
    if n is None or len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    if task in ("binary", "multi-class") and len(label_cols) == 1:
        # Stratify so rare classes survive the cut; without this a 238-fold
        # SCOPe task loses whole classes and the probe silently degrades.
        groups = df.groupby(label_cols[0], observed=True)
        share = n / len(df)
        picks = []
        for _, grp in groups:
            take = max(1, int(round(len(grp) * share)))
            picks.append(grp.sample(n=min(take, len(grp)), random_state=int(rng.integers(1 << 31))))
        out = pd.concat(picks)
        if len(out) > n:
            out = out.sample(n=n, random_state=seed)
        return out
    return df.sample(n=n, random_state=seed)


def load_dataset(
    path: str | Path,
    *,
    task: Task | None = None,
    label_col: str | list[str] | None = None,
    max_train: int | None = 4000,
    max_val: int | None = 1000,
    seed: int = 42,
    residue_level: bool = False,
    val_fraction: float = 0.2,
) -> Dataset:
    """Load a CSV, infer the task, split it, and subsample to the budget."""
    path = Path(path)
    df = pd.read_csv(path)

    for required in ("ID", "sequence"):
        if required not in df.columns:
            raise ValueError(f"{path}: missing required column {required!r}")

    label_cols = _pick_label_columns(df, label_col)
    df = df.dropna(subset=["sequence", *label_cols])

    if "sampled" in df.columns:  # honoured by the research datasets
        df = df[df["sampled"].astype(bool)]

    raw_y = df[label_cols[0]].to_numpy() if len(label_cols) == 1 else df[label_cols].to_numpy()
    resolved_task: Task = task or infer_task(raw_y, label_cols)

    if residue_level and "positions" not in df.columns:
        raise ValueError(
            f"{path}: residue-level extraction needs a 'positions' column of "
            "0-based residue indices"
        )

    # --- splits --------------------------------------------------------
    if "split" in df.columns:
        present = set(df["split"].astype(str).unique())
        val_name = next((v for v in _VAL_NAMES if v in present), None)
        if val_name is None:
            raise ValueError(f"{path}: 'split' column has no {_VAL_NAMES} level")
        train_df = df[df["split"].astype(str) == "train"]
        val_df = df[df["split"].astype(str) == val_name]
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"{path}: train or {val_name} split is empty")
    else:
        shuffled = df.sample(frac=1.0, random_state=seed)
        cut = int(len(shuffled) * (1 - val_fraction))
        train_df, val_df = shuffled.iloc[:cut], shuffled.iloc[cut:]

    train_df = _subsample(train_df, max_train, resolved_task, label_cols, seed)
    val_df = _subsample(val_df, max_val, resolved_task, label_cols, seed + 1)

    n_classes = None
    if resolved_task == "multi-label":
        n_classes = len(label_cols)
    elif resolved_task in ("binary", "multi-class"):
        n_classes = int(pd.concat([train_df, val_df])[label_cols[0]].nunique())

    def build(frame: pd.DataFrame) -> Split:
        y = (
            frame[label_cols].to_numpy(dtype=float)
            if len(label_cols) > 1
            else frame[label_cols[0]].to_numpy()
        )
        if resolved_task in ("binary", "multi-class"):
            y = y.astype(int)
        elif resolved_task == "regression":
            y = y.astype(float)
        positions = None
        if residue_level:
            # literal_eval, never eval: the research code runs eval() on CSV
            # contents, which is arbitrary code execution from a data file.
            positions = [
                list(ast.literal_eval(p)) if isinstance(p, str) else list(p)
                for p in frame["positions"]
            ]
        return Split(
            ids=frame["ID"].astype(str).tolist(),
            sequences=frame["sequence"].astype(str).tolist(),
            y=y,
            positions=positions,
        )

    return Dataset(
        name=path.stem,
        task=resolved_task,
        train=build(train_df),
        val=build(val_df),
        n_classes=n_classes,
        residue_level=residue_level,
    )
