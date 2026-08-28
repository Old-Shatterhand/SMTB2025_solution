"""Top-level API: dataset + checkpoint in, chosen layer and truncated model out."""

from __future__ import annotations

from pathlib import Path

import torch

from plmlayer.adapters import HFAdapter
from plmlayer.data import load_dataset
from plmlayer.extract import extract
from plmlayer.registry import resolve_spec
from plmlayer.select import SelectionResult, select_layer
from plmlayer.truncate import save_truncated
from plmlayer.types import Task

__all__ = ["suggest_layer", "load_adapter"]


def load_adapter(model: str, *, device: str | None = None, cache_dir: str | None = None):
    """Load the right adapter for ``model`` (preset key, HF id, or local path)."""
    spec = resolve_spec(model)
    if spec.adapter == "esmc":
        raise NotImplementedError(
            "ESM-C uses the separate `esm` SDK and is not wired up yet; "
            "use an ESM-2 checkpoint or another transformers model."
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return HFAdapter(spec, device=device, cache_dir=cache_dir).load()


def suggest_layer(
    data: str | Path,
    model: str,
    *,
    task: Task | None = None,
    label_col: str | list[str] | None = None,
    probe: str = "knn",
    k: int = 10,
    max_train: int = 4000,
    max_val: int = 1000,
    schedule: str = "full",
    residue_level: bool = False,
    n_seeds: int = 3,
    seed: int = 42,
    device: str | None = None,
    cache_dir: str | None = None,
    out: str | Path | None = None,
    progress: bool = True,
) -> SelectionResult:
    """Find the best layer of ``model`` for ``data``; optionally save it truncated."""
    adapter = load_adapter(model, device=device, cache_dir=cache_dir)
    ds = load_dataset(
        data,
        task=task,
        label_col=label_col,
        max_train=max_train,
        max_val=max_val,
        seed=seed,
        residue_level=residue_level,
    )

    train = extract(
        adapter, ds.train.sequences, ds.train.ids,
        positions=ds.train.positions, progress=progress,
    )
    val = extract(
        adapter, ds.val.sequences, ds.val.ids,
        positions=ds.val.positions, progress=progress,
    )

    result = select_layer(
        ds, train, val,
        model_name=adapter.spec.hf_id,
        probe=probe, k=k, schedule=schedule, n_seeds=n_seeds, seed=seed,
    )
    if out is not None:
        save_truncated(adapter, result, out)
    return result
