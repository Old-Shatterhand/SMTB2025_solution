"""Preset model specs and the adapter/spec resolution machinery.

The presets are the models benchmarked in Joeres, Senatorov et al. Layer counts
here are documentation only -- the authoritative count always comes from a
warm-up forward pass (``len(outputs.hidden_states) - 1``).
"""

from __future__ import annotations

import re
from typing import Iterable

from plmlayer.types import ModelSpec

__all__ = ["PRESETS", "register_spec", "get_spec", "list_specs", "resolve_spec"]

# ProtT5 and ProstT5 were pretrained with the rare residues mapped to X.
_T5_RESIDUE_MAP = {"U": "X", "Z": "X", "O": "X", "B": "X"}

_ESM2 = {
    "esm2_8m": ("facebook/esm2_t6_8M_UR50D", 6),
    "esm2_35m": ("facebook/esm2_t12_35M_UR50D", 12),
    "esm2_150m": ("facebook/esm2_t30_150M_UR50D", 30),
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D", 33),
    "esm2_3b": ("facebook/esm2_t36_3B_UR50D", 36),
}

# Names used in the paper's repo (src/viz/constants.py), kept as aliases so
# existing scripts and figures keep resolving.
ALIASES = {
    "esm_t6": "esm2_8m",
    "esm_t12": "esm2_35m",
    "esm_t30": "esm2_150m",
    "esm_t33": "esm2_650m",
    "esm_t36": "esm2_3b",
}

PRESETS: dict[str, ModelSpec] = {}
_LAYER_HINTS: dict[str, int] = {}


def register_spec(spec: ModelSpec, *, n_layers_hint: int | None = None) -> ModelSpec:
    """Register a spec under its key. Users may call this for custom models."""
    PRESETS[spec.key] = spec
    if n_layers_hint is not None:
        _LAYER_HINTS[spec.key] = n_layers_hint
    return spec


for _key, (_hf, _n) in _ESM2.items():
    register_spec(
        ModelSpec(
            key=_key,
            hf_id=_hf,
            mask_token="<mask>",
            max_residues=1022,  # max_position_embeddings 1026 minus <cls>/<eos>
            n_prefix=1,
            n_suffix=1,
            notes="BERT-style encoder; final norm is encoder.emb_layer_norm_after.",
        ),
        n_layers_hint=_n,
    )

for _key, _hf, _n in [
    ("prott5", "Rostlab/prot_t5_xl_uniref50", 24),
    ("prostt5", "Rostlab/ProstT5", 24),
]:
    register_spec(
        ModelSpec(
            key=_key,
            hf_id=_hf,
            encoder_only=True,
            dtype="bfloat16",  # T5 overflows in fp16
            residue_map=_T5_RESIDUE_MAP,
            uppercase=True,
            space_join=True,
            # ProstT5 needs the translation-direction token; it is an ordinary
            # added token, so special_tokens_mask does NOT hide it.
            prefix_text="<AA2fold> " if _key == "prostt5" else "",
            mask_token="<extra_id_0>",
            n_prefix=1 if _key == "prostt5" else 0,
            n_suffix=1,
            notes="T5 encoder only. No BOS; </s> is appended.",
        ),
        n_layers_hint=_n,
    )

for _key, _hf in [("ankh_base", "ElnaggarLab/ankh-base"), ("ankh_large", "ElnaggarLab/ankh-large")]:
    register_spec(
        ModelSpec(
            key=_key,
            hf_id=_hf,
            encoder_only=True,
            dtype="bfloat16",
            mask_token="<extra_id_0>",
            n_prefix=0,  # NOTE: no BOS. src/plm.py:225 wrongly slices [0, 1:].
            n_suffix=1,
            notes=(
                "Unigram tokenizer, no BOS, </s> suffix. The research code's "
                "[0, 1:] slice drops residue 0 and keeps </s>; correct is [0, :-1]."
            ),
        ),
        n_layers_hint=48,
    )

for _key, _hf, _n in [
    ("progen2_small", "hugohrban/progen2-small", 12),
    ("progen2_medium", "hugohrban/progen2-medium", 27),
    ("progen2_large", "hugohrban/progen2-large", 32),
]:
    register_spec(
        ModelSpec(
            key=_key,
            hf_id=_hf,
            adapter="progen2",
            trust_remote_code=True,
            prefix_text="1",  # N->C direction flag; ordinary vocab id, not special
            suffix_text="2",
            n_prefix=1,
            n_suffix=1,
            notes=(
                "ProGenConfig has no attribute_map: use n_layer/embed_dim. "
                "Pass 2-D input_ids -- 1-D input collapses the final hidden state."
            ),
        ),
        n_layers_hint=_n,
    )

register_spec(
    ModelSpec(
        key="protgpt2",
        hf_id="nferruz/ProtGPT2",
        n_prefix=0,
        n_suffix=0,
        notes="BPE tokenizer: residue-level extraction is impossible; whole-protein only.",
    ),
    n_layers_hint=36,
)

for _key, _hf, _n in [("esmc_300m", "esmc_300m", 30), ("esmc_600m", "esmc_600m", 36)]:
    register_spec(
        ModelSpec(
            key=_key,
            hf_id=_hf,
            adapter="esmc",
            dtype="bfloat16",
            n_prefix=1,
            n_suffix=1,
            notes="Uses the separate `esm` SDK, not transformers.",
        ),
        n_layers_hint=_n,
    )


def list_specs() -> list[str]:
    return sorted(PRESETS)


def get_spec(key: str) -> ModelSpec | None:
    key = key.strip()
    key = ALIASES.get(key, key)
    return PRESETS.get(key)


def layer_hint(key: str) -> int | None:
    return _LAYER_HINTS.get(ALIASES.get(key, key))


def resolve_spec(model: str | ModelSpec, **overrides) -> ModelSpec:
    """Turn a user-supplied model identifier into a :class:`ModelSpec`.

    Accepts a preset key (``"esm2_650m"``), a repo alias (``"esm_t30"``), a raw
    HuggingFace id, a local path, or an explicit spec. Unknown identifiers get a
    bare spec whose conventions are then established by calibration.
    """
    if isinstance(model, ModelSpec):
        spec = model
    elif (found := get_spec(model)) is not None:
        spec = found
    else:
        # Unknown checkpoint: infer nothing, declare nothing, let calibration
        # work the layout out empirically.
        key = re.sub(r"[^A-Za-z0-9]+", "_", model.strip("/").split("/")[-1]).lower()
        spec = ModelSpec(key=key, hf_id=model)

    if overrides:
        from dataclasses import replace

        spec = replace(spec, **{k: v for k, v in overrides.items() if v is not None})
    return spec
