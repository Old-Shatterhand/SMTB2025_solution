"""The layer search.

Grounded in Figure 3E-H of the paper: 15-20% of the training data is enough to
identify a layer reaching >=95% of the best achievable performance, and the same
layer was picked across all three seeds. That is what makes an exhaustive sweep
over layers affordable -- the cost is in the data, not the layers, and a single
forward pass yields every layer at once.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from plmlayer.data import Dataset
from plmlayer.extract import EmbeddingSet
from plmlayer.metrics import HIGHER_IS_BETTER, compute_metric, default_metric
from plmlayer.probe import fit_predict
from plmlayer.types import Task

__all__ = ["SelectionResult", "select_layer", "layer_schedule"]


def layer_schedule(n_layers: int, mode: str = "full", stride: int | None = None) -> list[int]:
    """Which layers to probe. ``full`` sweeps all of them (the default)."""
    if mode == "full":
        return list(range(n_layers + 1))
    if mode == "coarse":
        step = stride or max(1, round(n_layers / 8))
        coarse = list(range(0, n_layers + 1, step))
        if coarse[-1] != n_layers:
            coarse.append(n_layers)
        return coarse
    raise ValueError(f"unknown schedule {mode!r}")


@dataclass(slots=True)
class SelectionResult:
    """What the tool concluded, and how much to trust it."""

    model: str
    dataset: str
    task: Task
    metric: str
    probe: str

    best_layer: int
    best_score: float
    last_layer: int
    last_layer_score: float

    curve: dict[int, float]
    n_train: int
    n_val: int

    #: raw argmax of the curve, before the shallowest-within-tolerance rule
    peak_layer: int = -1
    peak_score: float = float("nan")
    #: every layer statistically indistinguishable from the peak
    plateau: list[int] = field(default_factory=list)
    tolerance: float = 0.02

    seed_layers: list[int] = field(default_factory=list)
    seed_agreement: float = float("nan")
    residue_level: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def gain_over_last(self) -> float:
        """Relative improvement of the chosen layer over the conventional last one.

        This is the paper's Figure 1A statistic, and the tool's headline number.
        """
        last = self.last_layer_score
        if not math.isfinite(last) or abs(last) < 1e-12:
            return float("nan")
        if HIGHER_IS_BETTER.get(self.metric, True):
            return self.best_score / last - 1.0
        return last / self.best_score - 1.0

    @property
    def depth_fraction(self) -> float:
        return self.best_layer / self.last_layer if self.last_layer else 0.0

    def summary(self) -> str:
        gain = self.gain_over_last
        gain_txt = "n/a" if math.isnan(gain) else f"{gain:+.1%}"
        lines = [
            f"model    {self.model}",
            f"dataset  {self.dataset}  ({self.task}, {self.metric}, {self.probe} probe)",
            f"data     {self.n_train} train / {self.n_val} val",
            "",
            f"best layer      {self.best_layer} of {self.last_layer} "
            f"({self.depth_fraction:.0%} depth)   {self.metric} = {self.best_score:.4f}",
            f"last layer      {self.last_layer}"
            f"{' ' * 15}{self.metric} = {self.last_layer_score:.4f}",
            f"gain over last  {gain_txt}",
        ]
        if self.peak_layer != self.best_layer:
            lines.append(
                f"peak layer      {self.peak_layer}"
                f"{' ' * 15}{self.metric} = {self.peak_score:.4f}  "
                f"(within {self.tolerance:.0%}; took the shallowest of {self.plateau})"
            )
        if self.seed_layers:
            lines.append(
                f"seed agreement  {self.seed_agreement:.0%} "
                f"(layers chosen across seeds: {sorted(set(self.seed_layers))})"
            )
        lines.extend(f"note     {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["curve"] = {str(k): v for k, v in self.curve.items()}
        d["gain_over_last"] = self.gain_over_last
        d["depth_fraction"] = self.depth_fraction
        return d


def _score_layer(
    train: EmbeddingSet, val: EmbeddingSet, dataset: Dataset,
    layer: int, metric: str, probe: str, k: int,
    train_rows: np.ndarray | None = None,
) -> float:
    tx, ty = train.layer(layer), dataset.train.y
    if train_rows is not None:
        tx, ty = tx[train_rows], ty[train_rows]
    pred = fit_predict(
        tx, ty, val.layer(layer), dataset.val.y,
        task=dataset.task, probe=probe, k=k,
    )
    return compute_metric(pred, dataset.val.y, metric, dataset.task)


def _plateau(curve: dict[int, float], metric: str, peak: int, tolerance: float) -> list[int]:
    """Layers whose score is within ``tolerance`` (relative) of the peak."""
    higher = HIGHER_IS_BETTER.get(metric, True)
    best = curve[peak]
    if not math.isfinite(best):
        return [peak]
    band = tolerance * abs(best)
    keep = []
    for layer, score in curve.items():
        if not math.isfinite(score):
            continue
        if (score >= best - band) if higher else (score <= best + band):
            keep.append(layer)
    return sorted(keep) or [peak]


def _choose(curve: dict[int, float], metric: str, tolerance: float) -> tuple[int, int, list[int]]:
    """Return ``(chosen, peak, plateau)``.

    Among layers indistinguishable from the peak, take the **shallowest**: it
    scores the same but yields a smaller, faster truncated model, and it is a
    markedly more stable choice than a raw argmax over a flat region.
    """
    peak = _argbest(curve, metric)
    plateau = _plateau(curve, metric, peak, tolerance)
    return min(plateau), peak, plateau


def _argbest(curve: dict[int, float], metric: str) -> int:
    higher = HIGHER_IS_BETTER.get(metric, True)
    finite = {k: v for k, v in curve.items() if math.isfinite(v)}
    if not finite:
        raise RuntimeError("every layer scored NaN; the probe or labels are degenerate")
    return (max if higher else min)(finite, key=finite.get)


def select_layer(
    dataset: Dataset,
    train: EmbeddingSet,
    val: EmbeddingSet,
    *,
    model_name: str = "",
    metric: str | None = None,
    probe: str = "knn",
    k: int = 10,
    schedule: str = "full",
    layers: Sequence[int] | None = None,
    n_seeds: int = 3,
    seed_fraction: float = 0.8,
    seed: int = 42,
    tolerance: float = 0.02,
) -> SelectionResult:
    """Probe each candidate layer and pick the best."""
    metric = metric or default_metric(dataset.task)
    n_layers = train.n_states - 1
    candidates = list(layers) if layers is not None else layer_schedule(n_layers, schedule)

    curve = {
        layer: _score_layer(train, val, dataset, layer, metric, probe, k)
        for layer in candidates
    }
    best, peak, plateau = _choose(curve, metric, tolerance)

    # Stability check: does the choice survive re-drawing the training subsample?
    # The paper found the same layer picked across all three of its seeds.
    seed_layers: list[int] = []
    n_train = len(dataset.train)
    if n_seeds > 0 and n_train > 20:
        rng = np.random.default_rng(seed)
        take = max(10, int(n_train * seed_fraction))
        for _ in range(n_seeds):
            rows = rng.choice(n_train, size=take, replace=False)
            sub = {
                layer: _score_layer(train, val, dataset, layer, metric, probe, k, rows)
                for layer in candidates
            }
            try:
                seed_layers.append(_choose(sub, metric, tolerance)[0])
            except RuntimeError:
                continue

    agreement = (
        sum(layer == best for layer in seed_layers) / len(seed_layers)
        if seed_layers
        else float("nan")
    )

    notes: list[str] = []
    if len(plateau) > 1:
        notes.append(
            f"layers {plateau} are within {tolerance:.0%} of the peak; "
            f"took layer {best} for the smallest model at equal performance"
        )
    if best == n_layers:
        notes.append(
            "the last layer won: this dataset is one of the ~18% where the "
            "conventional choice is already best; truncation buys nothing here"
        )
    if seed_layers and agreement < 0.5:
        notes.append(
            "layer choice is unstable across resampling -- raise max_train, or "
            "treat the whole peak region as equally good"
        )
    if schedule == "coarse":
        notes.append("coarse schedule: only a subset of layers was probed")

    return SelectionResult(
        model=model_name or train.meta.get("hf_id", "?"),
        dataset=dataset.name,
        task=dataset.task,
        metric=metric,
        probe=probe,
        best_layer=best,
        best_score=curve[best],
        peak_layer=peak,
        peak_score=curve[peak],
        plateau=plateau,
        tolerance=tolerance,
        last_layer=n_layers,
        last_layer_score=curve.get(n_layers, float("nan")),
        curve=curve,
        n_train=n_train,
        n_val=len(dataset.val),
        seed_layers=seed_layers,
        seed_agreement=agreement,
        residue_level=dataset.residue_level,
        notes=notes,
    )
