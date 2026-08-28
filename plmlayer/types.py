"""Core data types for plmlayer.

Deliberately free of torch/transformers imports so that specs, layouts and the
calibration contract can be imported (and unit-tested) without model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

__all__ = [
    "Task",
    "Pooling",
    "TokenLayout",
    "ModelSpec",
    "ResidueLevelUnsupported",
    "CalibrationError",
]

# The four task types used throughout the study. Kept identical to
# src/viz/constants.py::DATASET2TASK so results stay comparable.
Task = Literal["regression", "binary", "multi-class", "multi-label"]

# "mean_residue" pools over residues only (requires a 1:1 residue alignment);
# "mean" pools over every non-special token and is a *different quantity* for
# subword models -- the two must never be silently compared.
Pooling = Literal["mean_residue", "mean", "cls", "last", "max", "none"]


class CalibrationError(RuntimeError):
    """Raised when a tokenizer's residue alignment cannot be established."""


class ResidueLevelUnsupported(RuntimeError):
    """Raised when residue-level output is requested from a subword model."""


@dataclass(frozen=True, slots=True)
class TokenLayout:
    """Where the real residues sit inside a tokenized sequence.

    Established empirically by :mod:`plmlayer.calibrate`, never guessed. For a
    sequence of ``n`` residues the token ids are laid out as::

        [ prefix x n_prefix ][ residue_0 ... residue_{n-1} ][ suffix x n_suffix ]

    ``residue_level`` is False when the tokenizer merges residues into subword
    units (ProtGPT2), which makes any per-residue indexing meaningless.
    """

    n_prefix: int
    n_suffix: int
    residue_level: bool
    unk_residues: frozenset[str] = frozenset()
    source: Literal["calibrated", "declared"] = "calibrated"

    def residue_slice(self, n_tokens: int) -> slice:
        """Slice selecting the residue tokens out of a length-``n_tokens`` row."""
        stop = n_tokens - self.n_suffix if self.n_suffix else n_tokens
        return slice(self.n_prefix, stop)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything model-specific that cannot be inferred from the checkpoint.

    Preprocessing fields describe *training-time conventions* (e.g. ProtT5's
    ``[UZOB] -> X`` substitution) which are recorded nowhere in the tokenizer
    artifacts. They are declared here and then **verified** by calibration.
    """

    key: str
    hf_id: str
    revision: str | None = None
    adapter: str | None = None

    # --- loading -------------------------------------------------------
    encoder_only: bool | None = None  # None -> infer from config.is_encoder_decoder
    trust_remote_code: bool = False
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"

    # --- declared preprocessing (verified, never inferred) -------------
    residue_map: Mapping[str, str] = field(default_factory=dict)
    uppercase: bool = False
    space_join: bool = False
    prefix_text: str = ""
    suffix_text: str = ""
    mask_token: str | None = None
    max_residues: int | None = None

    # --- declared layout (cross-checked against calibration) -----------
    n_prefix: int | None = None
    n_suffix: int | None = None

    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def map_residues(self, sequence: str) -> str:
        """Apply the declared residue-level substitutions (e.g. ``[UZOB] -> X``)."""
        seq = sequence.upper() if self.uppercase else sequence
        if self.residue_map:
            seq = "".join(self.residue_map.get(c, c) for c in seq)
        return seq

    def preprocess(self, sequence: str) -> tuple[str, str]:
        """Apply the declared conventions.

        Returns ``(text, residues)`` where ``text`` is fed to the tokenizer and
        ``residues`` is the mapped residue string -- the ground truth that
        calibration and the per-batch invariant check against.
        """
        residues = self.map_residues(sequence)
        body = " ".join(residues) if self.space_join else residues
        return f"{self.prefix_text}{body}{self.suffix_text}", residues
