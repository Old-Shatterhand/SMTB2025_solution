"""Adapter contract: sequences in, per-layer representations out, plus truncation."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Sequence

import torch

from plmlayer.types import ModelSpec, Pooling, TokenLayout

__all__ = ["EncodedBatch", "LayerStack", "PLMAdapter"]


@dataclass(slots=True)
class EncodedBatch:
    """A tokenized batch plus the proven mapping back to residues."""

    model_inputs: dict[str, torch.Tensor]
    residue_index: torch.Tensor  # (B, R_max) int64, -1 padded
    lengths: torch.Tensor  # (B,) int64 -- residue counts
    ids: list[str]

    @property
    def batch_size(self) -> int:
        return len(self.ids)

    def to(self, device: torch.device) -> "EncodedBatch":
        return EncodedBatch(
            {k: v.to(device) for k, v in self.model_inputs.items()},
            self.residue_index.to(device),
            self.lengths.to(device),
            self.ids,
        )


@dataclass(slots=True)
class LayerStack:
    """All layers of one batch, always shaped ``(S, B, T, D)``."""

    states: torch.Tensor
    residue_index: torch.Tensor
    lengths: torch.Tensor

    @property
    def n_states(self) -> int:
        return self.states.shape[0]

    def residue_mask(self) -> torch.Tensor:
        """(B, T) bool -- True exactly at residue positions."""
        b, t = self.states.shape[1], self.states.shape[2]
        mask = torch.zeros(b, t, dtype=torch.bool, device=self.states.device)
        idx = self.residue_index
        mask.scatter_(1, idx.clamp(min=0), idx >= 0)
        return mask


class PLMAdapter(abc.ABC):
    """Knows how to preprocess, extract every layer, and truncate one model family."""

    #: matched against ``config.model_type``; highest priority wins
    model_types: tuple[str, ...] = ()
    priority: int = 0

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.spec = spec
        self.device = torch.device(device)
        self._dtype = dtype
        self.cache_dir = cache_dir
        self._loaded = False

    # --- lifecycle -----------------------------------------------------
    @abc.abstractmethod
    def load(self) -> "PLMAdapter": ...

    def unload(self) -> None:
        self._loaded = False

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"{type(self).__name__}.load() has not been called")

    # --- introspection (valid after load) ------------------------------
    @property
    @abc.abstractmethod
    def n_states(self) -> int:
        """Number of extractable representations, ``n_layers + 1``."""

    @property
    def n_layers(self) -> int:
        return self.n_states - 1

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def layout(self) -> TokenLayout: ...

    @property
    def supports_residue_level(self) -> bool:
        return self.layout.residue_level

    # --- the work ------------------------------------------------------
    @abc.abstractmethod
    def encode(self, sequences: Sequence[str], ids: Sequence[str]) -> EncodedBatch: ...

    @abc.abstractmethod
    def forward(self, batch: EncodedBatch) -> LayerStack: ...

    @abc.abstractmethod
    def truncate(self, layer: int):
        """Return a standalone model whose output equals layer ``layer``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = f"{self.n_states} states" if self._loaded else "unloaded"
        return f"{type(self).__name__}({self.spec.hf_id!r}, {state})"
