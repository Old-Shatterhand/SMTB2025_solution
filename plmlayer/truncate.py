"""Build, verify and save the truncated model.

This is the tool's actual deliverable: not a layer index, but an encoder the
user can drop into their pipeline with a plain ``AutoModel.from_pretrained``.

Every save is gated on a numerical check that the truncated model reproduces the
representations the probe actually scored. That guard matters because the naive
version of this is wrong: ``hidden_states[k]`` is the raw block output, while a
model truncated to ``k`` blocks emits ``final_norm(h_k)`` -- on ESM-2 the two
differ by ~32 in absolute magnitude. See :mod:`plmlayer.adapters.hf`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from plmlayer.adapters.base import PLMAdapter

__all__ = ["TruncationCheck", "verify_truncation", "save_truncated"]

_PROBE_SEQS = ("MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY")


@dataclass(slots=True)
class TruncationCheck:
    layer: int
    max_abs_diff: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.tolerance


def verify_truncation(
    adapter: PLMAdapter,
    model,
    layer: int,
    *,
    sequences: Sequence[str] = _PROBE_SEQS,
    tolerance: float = 1e-4,
) -> TruncationCheck:
    """Assert the truncated model reproduces the probed layer."""
    batch = adapter.encode(list(sequences), [f"_check{i}" for i in range(len(sequences))])
    reference = adapter.forward(batch).states[layer]
    with torch.no_grad():
        got = model(**batch.model_inputs).last_hidden_state
    diff = float((got.float() - reference.float()).abs().max())
    return TruncationCheck(layer=layer, max_abs_diff=diff, tolerance=tolerance)


def _verify_reload(
    adapter: PLMAdapter, path: Path, layer: int, tolerance: float
) -> TruncationCheck:
    """Load the saved directory back and re-check it against the probed layer."""
    from transformers import AutoModel

    reloaded = AutoModel.from_pretrained(path, add_pooling_layer=False).eval()
    return verify_truncation(adapter, reloaded, layer, tolerance=tolerance)


def _model_card(result, check: TruncationCheck, adapter: PLMAdapter) -> str:
    gain = result.gain_over_last
    gain_txt = "n/a" if gain != gain else f"{gain:+.1%}"
    kept, total = check.layer, adapter.n_layers
    return f"""---
library_name: transformers
tags: [protein-language-model, plmlayer, truncated]
base_model: {adapter.spec.hf_id}
---

# {adapter.spec.hf_id} truncated to {kept} of {total} layers

Produced by [`plmlayer`](https://github.com/Old-Shatterhand/SMTB2025_solution),
which selects the most informative layer of a protein language model for a
specific downstream dataset instead of defaulting to the last one.

## What this model is

The first **{kept}** transformer blocks of `{adapter.spec.hf_id}`, with the
model's final normalisation applied. Its `last_hidden_state` is exactly the
representation that layer {kept} of the full model produces -- verified
numerically at save time (max abs diff `{check.max_abs_diff:.2e}`).

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("{{path}}")
tok = AutoTokenizer.from_pretrained("{{path}}")
```

## Why layer {kept}

| | |
|---|---|
| dataset | `{result.dataset}` |
| task | {result.task} |
| metric | {result.metric} ({result.probe} probe) |
| score at layer {kept} | **{result.best_score:.4f}** |
| score at layer {total} (the usual choice) | {result.last_layer_score:.4f} |
| gain over the last layer | **{gain_txt}** |
| probing data | {result.n_train} train / {result.n_val} validation |
| agreement across resampling seeds | {result.seed_agreement:.0%} |

Layers kept: {kept}/{total} ({kept / total if total else 0:.0%} of depth), so
inference is correspondingly cheaper.

## Caveats

- Layer choice is **dataset-specific**. The paper this implements found the
  dataset matters more than the task: deep-mutational-scanning sets peak early,
  diverse natural-protein sets peak late. Re-run `plmlayer` for a new dataset.
- Selection used a subsample ({result.n_train} training sequences), justified by
  Figure 3E-H of the paper: 15-20% of the data suffices to find a layer within
  5% of the best.
- Scores come from a {result.probe} probe on frozen embeddings. Fine-tuning
  changes the picture -- it makes performance increase monotonically with depth.

## Citation

> Joeres, Senatorov, Kolchina, Klakow & Kalinina. *Task- and dataset-specific
> information in protein language models.* arXiv:2608.12090.
"""


def save_truncated(
    adapter: PLMAdapter,
    result,
    out_dir: str | Path,
    *,
    layer: int | None = None,
    tolerance: float = 1e-4,
) -> Path:
    """Truncate, verify, and write a loadable model directory."""
    layer = result.best_layer if layer is None else layer
    out = Path(out_dir)

    model = adapter.truncate(layer)
    check = verify_truncation(adapter, model, layer, tolerance=tolerance)
    if not check.passed:
        raise RuntimeError(
            f"refusing to save: the truncated model does not reproduce layer "
            f"{layer} (max abs diff {check.max_abs_diff:.3e} > {tolerance:.1e}). "
            "This means the final-norm handling is wrong for this architecture."
        )

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    adapter.tokenizer.save_pretrained(out)

    # Round-trip through from_pretrained before declaring success. Saving a model
    # that cannot be loaded back is the one failure this tool must never ship,
    # and config/state-dict drift (e.g. depth-dependent auxiliary heads) is
    # invisible until you actually reload.
    reload_check = _verify_reload(adapter, out, layer, tolerance)
    if not reload_check.passed:
        raise RuntimeError(
            f"the model saved to {out} does not reload faithfully "
            f"(max abs diff {reload_check.max_abs_diff:.3e} > {tolerance:.1e})"
        )

    (out / "README.md").write_text(_model_card(result, check, adapter).replace("{path}", str(out)))
    (out / "plmlayer.json").write_text(
        json.dumps(
            {
                **result.to_dict(),
                "base_model": adapter.spec.hf_id,
                "revision": adapter.spec.revision,
                "layers_kept": layer,
                "layers_total": adapter.n_layers,
                "verification_max_abs_diff": check.max_abs_diff,
                "reload_max_abs_diff": reload_check.max_abs_diff,
            },
            indent=2,
            default=str,
        )
    )
    return out
