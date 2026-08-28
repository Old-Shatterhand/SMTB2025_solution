"""Batched extraction of every layer into one contiguous in-memory array.

The research code writes one pickle per protein per layer -- 30,232 files x 35
layers for a single model/dataset, and ~10 TB across the paper. That layout is
unnecessary here: because the tool subsamples before embedding, all ``L+1``
layers for a few thousand sequences fit comfortably in RAM, and a single forward
pass produces all of them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from plmlayer.adapters.base import PLMAdapter
from plmlayer.types import ResidueLevelUnsupported

__all__ = ["EmbeddingSet", "extract", "estimate_bytes"]

_FP16_MAX = 65504.0


@dataclass(slots=True)
class EmbeddingSet:
    """Per-layer representations for one split. ``values`` is ``(S, N, D)``."""

    ids: list[str]
    values: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.values.shape[0]

    @property
    def n_items(self) -> int:
        return self.values.shape[1]

    def layer(self, index: int) -> np.ndarray:
        """``(N, D)`` float32 view of one layer, ready for a probe."""
        return self.values[index].astype(np.float32, copy=False)

    def nbytes(self) -> int:
        return int(self.values.nbytes)


def estimate_bytes(n_items: int, n_states: int, dim: int, itemsize: int = 2) -> int:
    return n_items * n_states * dim * itemsize


def _batches(lengths: Sequence[int], max_tokens: int, max_size: int) -> list[list[int]]:
    """Length-sorted batches under a token budget.

    Sorting by length keeps padding overhead low and bounds peak activation
    memory far better than a fixed batch size.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        nxt_max = max(cur_max, lengths[i])
        if cur and (len(cur) + 1 > max_size or nxt_max * (len(cur) + 1) > max_tokens):
            out.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur.append(i)
            cur_max = nxt_max
    if cur:
        out.append(cur)
    return out


def extract(
    adapter: PLMAdapter,
    sequences: Sequence[str],
    ids: Sequence[str],
    *,
    positions: Sequence[Sequence[int]] | None = None,
    max_tokens_per_batch: int = 8192,
    max_batch_size: int = 64,
    memory_budget_bytes: int | None = 8 * 1024**3,
    progress: bool = False,
) -> EmbeddingSet:
    """Embed ``sequences`` and return every layer.

    With ``positions``, one row is produced per selected residue; otherwise one
    mean-pooled row per sequence.
    """
    if positions is not None and not adapter.supports_residue_level:
        raise ResidueLevelUnsupported(
            f"{adapter.spec.hf_id} uses a subword tokenizer, so residues do not "
            "map 1:1 onto tokens. Residue-level extraction is unavailable; use "
            "whole-protein pooling instead."
        )

    n_out = sum(len(p) for p in positions) if positions is not None else len(sequences)
    need = estimate_bytes(n_out, adapter.n_states, adapter.hidden_size)
    if memory_budget_bytes is not None and need > memory_budget_bytes:
        raise MemoryError(
            f"extraction needs {need / 1024**3:.1f} GiB "
            f"({n_out} rows x {adapter.n_states} layers x {adapter.hidden_size} dims, "
            f"fp16) but the budget is {memory_budget_bytes / 1024**3:.1f} GiB. "
            "Lower max_train/max_val, or raise memory_budget_bytes."
        )

    out = np.zeros((adapter.n_states, n_out, adapter.hidden_size), dtype=np.float16)
    row_of = np.zeros(len(sequences) + 1, dtype=np.int64)
    if positions is not None:
        row_of[1:] = np.cumsum([len(p) for p in positions])

    batches = _batches([len(s) for s in sequences], max_tokens_per_batch, max_batch_size)
    iterator = batches
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(batches, desc="embedding", unit="batch")
        except ImportError:
            pass

    overflow = False
    for group in iterator:
        batch = adapter.encode([sequences[i] for i in group], [ids[i] for i in group])
        stack = adapter.forward(batch)
        mask = stack.residue_mask()  # (B, T)

        for slot, seq_i in enumerate(group):
            valid = mask[slot]
            states = stack.states[:, slot]  # (S, T, D)
            if positions is None:
                pooled = states[:, valid].float().mean(dim=1)  # (S, D)
                block = pooled.unsqueeze(1)
            else:
                want = torch.as_tensor(list(positions[seq_i]), device=states.device)
                token_idx = batch.residue_index[slot]
                if want.numel() and int(want.max()) >= int((token_idx >= 0).sum()):
                    raise IndexError(
                        f"sequence {ids[seq_i]!r}: position {int(want.max())} is "
                        f"outside its {int((token_idx >= 0).sum())} residues"
                    )
                block = states[:, token_idx[want]].float()  # (S, P, D)

            if float(block.abs().max()) > _FP16_MAX:
                overflow = True
            lo = row_of[seq_i] if positions is not None else seq_i
            hi = lo + block.shape[1]
            out[:, lo:hi] = block.to(torch.float16).cpu().numpy()

    if overflow:
        warnings.warn(
            "activations exceeded the float16 range and were clipped to inf; "
            "re-run with a float32 store for this model.",
            RuntimeWarning,
            stacklevel=2,
        )

    row_ids = (
        [f"{ids[i]}:{p}" for i in range(len(ids)) for p in positions[i]]
        if positions is not None
        else list(ids)
    )
    return EmbeddingSet(
        ids=row_ids,
        values=out,
        meta={
            "hf_id": adapter.spec.hf_id,
            "revision": adapter.spec.revision,
            "n_states": adapter.n_states,
            "hidden_size": adapter.hidden_size,
            "layout": adapter.layout,
            "final_state_is_normed": getattr(adapter, "final_state_is_normed", None),
            "residue_level": positions is not None,
        },
    )
