"""The generic HuggingFace adapter.

Covers ESM-2, ProtT5, ProstT5, Ankh, ProGen2 and ProtGPT2 through one code path.
Model-specific knowledge lives in :class:`~plmlayer.types.ModelSpec` (declarative)
and in the calibrated :class:`~plmlayer.types.TokenLayout` (empirical), not in
per-family branches.

The one subtlety worth reading before editing: ``hidden_states[i]`` for ``i < L``
is the **raw** block output, while ``hidden_states[L]`` has already had the
model's final norm applied. A model truncated to ``k`` blocks emits
``final_norm(h_k)``. So to make "probe layer k" and "ship a k-block model" mean
the same thing, we apply the final norm to every intermediate layer at
extraction time. See :meth:`HFAdapter._normalize_states`.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence

import torch
from torch import nn

from plmlayer.adapters.base import EncodedBatch, LayerStack, PLMAdapter
from plmlayer.calibrate import calibrate
from plmlayer.types import ModelSpec, TokenLayout

__all__ = ["HFAdapter"]

# Where transformer blocks and the trailing norm live, per architecture. Probed
# in order; the first hit wins.
_BLOCK_OWNERS = ("", "encoder", "transformer", "model")
_BLOCK_ATTRS = ("layer", "block", "h", "layers")
_FINAL_NORM_ATTRS = (
    "emb_layer_norm_after",  # ESM-2
    "final_layer_norm",  # T5
    "ln_f",  # GPT-2 / ProGen2
    "layer_norm",
    "norm",
)
# Config keys that record depth, across the families we support.
_DEPTH_KEYS = ("num_hidden_layers", "num_layers", "n_layer", "n_layers")


def _resolve(root: nn.Module, path: str) -> nn.Module | None:
    obj = root
    for part in filter(None, path.split(".")):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def locate_blocks(model: nn.Module) -> tuple[nn.Module, str, nn.ModuleList]:
    """Find the ``ModuleList`` of transformer blocks and the module owning it."""
    for owner_path in _BLOCK_OWNERS:
        owner = _resolve(model, owner_path)
        if owner is None:
            continue
        for attr in _BLOCK_ATTRS:
            blocks = getattr(owner, attr, None)
            if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
                return owner, attr, blocks
    raise RuntimeError(
        f"cannot locate transformer blocks on {type(model).__name__}; "
        "subclass HFAdapter and override locate_blocks for this architecture"
    )


def locate_final_norm(owner: nn.Module) -> nn.Module | None:
    """Find the norm applied after the last block, if the architecture has one."""
    for attr in _FINAL_NORM_ATTRS:
        mod = getattr(owner, attr, None)
        if isinstance(mod, nn.Module):
            return mod
    return None


def _set_depth(config, depth: int) -> None:
    """Write the new block count to every depth key the config actually has."""
    written = False
    for key in _DEPTH_KEYS:
        if key in config.__dict__:
            setattr(config, key, depth)
            written = True
    if not written:  # configs exposing depth only via attribute_map
        for key in _DEPTH_KEYS:
            try:
                getattr(config, key)
            except AttributeError:
                continue
            setattr(config, key, depth)
            written = True
            break
    if not written:
        raise RuntimeError(f"cannot record depth on {type(config).__name__}")


class HFAdapter(PLMAdapter):
    """Generic adapter over ``transformers``."""

    model_types = ()
    priority = 0

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _resolve_dtype(self, config) -> torch.dtype:
        if self._dtype is not None:
            return self._dtype
        declared = self.spec.dtype
        if declared != "auto":
            return getattr(torch, declared)
        # T5-family activations overflow in fp16; they were trained in bf16.
        if getattr(config, "model_type", "") == "t5":
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float32
        if self.device.type == "cuda":
            return torch.float16
        return torch.float32

    def _load_model(self, config):
        from transformers import AutoModel, AutoModelForTextEncoding

        kw = dict(
            trust_remote_code=self.spec.trust_remote_code,
            cache_dir=self.cache_dir,
            torch_dtype=self._resolve_dtype(config),
        )
        if self.spec.revision:
            kw["revision"] = self.spec.revision

        encoder_only = self.spec.encoder_only
        if encoder_only is None:
            encoder_only = bool(getattr(config, "is_encoder_decoder", False))

        if encoder_only:
            # Loads T5EncoderModel, so the (discarded) decoder is never
            # materialised -- ProtT5-XL drops from ~3B to ~1.2B params.
            try:
                return AutoModelForTextEncoding.from_pretrained(self.spec.hf_id, **kw)
            except (KeyError, ValueError):
                model = AutoModel.from_pretrained(self.spec.hf_id, **kw)
                return getattr(model, "encoder", model)

        try:
            # Suppresses the randomly-initialised pooler that AutoModel would
            # otherwise attach to ESM -- random weights must never be shipped.
            return AutoModel.from_pretrained(
                self.spec.hf_id, add_pooling_layer=False, **kw
            )
        except TypeError:
            return AutoModel.from_pretrained(self.spec.hf_id, **kw)

    def load(self) -> "HFAdapter":
        from transformers import AutoConfig, AutoTokenizer

        common = dict(
            trust_remote_code=self.spec.trust_remote_code, cache_dir=self.cache_dir
        )
        if self.spec.revision:
            common["revision"] = self.spec.revision

        self.config = AutoConfig.from_pretrained(self.spec.hf_id, **common)
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id, **common)
        self.model = self._load_model(self.config).to(self.device).eval()

        report = calibrate(self.tokenizer, self.spec)
        self._layout = report.layout
        self.calibration = report

        self._blocks_owner, self._blocks_attr, self._blocks = locate_blocks(self.model)
        self._final_norm = locate_final_norm(self._blocks_owner)

        self._loaded = True
        self._warm_up()
        return self

    def _warm_up(self) -> None:
        """One forward pass to establish ground truth about the model's outputs."""
        batch = self.encode(["ACDEFGHIKLMNPQRSTVWY"], ["_warmup"])
        with torch.no_grad():
            out = self.model(**batch.model_inputs, output_hidden_states=True)

        self._n_states = len(out.hidden_states)
        self._hidden_size = int(out.hidden_states[0].shape[-1])

        # Is hidden_states[-1] already the post-final-norm state? True on
        # transformers 4.57.x for every family here, but upstream has moved ESM
        # hidden-state capture onto EsmLayer, which would put emb_layer_norm_after
        # outside the recorded states and silently change the last layer.
        last = getattr(out, "last_hidden_state", None)
        self.final_state_is_normed = bool(
            last is not None
            and last.shape == out.hidden_states[-1].shape
            and torch.allclose(last.float(), out.hidden_states[-1].float(), atol=1e-4)
        )
        if not self.final_state_is_normed and self._final_norm is not None:
            warnings.warn(
                f"{self.spec.hf_id}: hidden_states[-1] is not the post-final-norm "
                "state on this transformers version. plmlayer will apply the final "
                "norm to it for consistency; numbers will differ from a run where "
                "it was already applied.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Cross-check the config, but never trust it over the measurement.
        for key in _DEPTH_KEYS:
            depth = getattr(self.config, key, None)
            if isinstance(depth, int):
                if depth + 1 != self._n_states:
                    warnings.warn(
                        f"{self.spec.hf_id}: config.{key}={depth} implies "
                        f"{depth + 1} states but the model returned "
                        f"{self._n_states}; using the measured value.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                break

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        self._require_loaded()
        return self._n_states

    @property
    def hidden_size(self) -> int:
        self._require_loaded()
        return self._hidden_size

    @property
    def layout(self) -> TokenLayout:
        self._require_loaded()
        return self._layout

    @property
    def max_residues(self) -> int | None:
        return self.spec.max_residues

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def encode(self, sequences: Sequence[str], ids: Sequence[str]) -> EncodedBatch:
        cap = self.spec.max_residues
        texts, residue_counts = [], []
        for seq in sequences:
            if cap is not None:
                seq = seq[:cap]
            text, residues = self.spec.preprocess(seq)
            texts.append(text)
            residue_counts.append(len(residues))

        enc = self.tokenizer(
            texts, add_special_tokens=True, padding=True, return_tensors="pt"
        )
        model_inputs = {
            k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")
        }

        lay = (
            self._layout
            if self._loaded
            else calibrate(self.tokenizer, self.spec).layout
        )
        n = len(texts)
        r_max = max(residue_counts) if residue_counts else 0
        residue_index = torch.full((n, max(r_max, 1)), -1, dtype=torch.long)
        if lay.residue_level:
            for i, count in enumerate(residue_counts):
                start = lay.n_prefix
                residue_index[i, :count] = torch.arange(start, start + count)

        batch = EncodedBatch(
            model_inputs=model_inputs,
            residue_index=residue_index,
            lengths=torch.tensor(residue_counts, dtype=torch.long),
            ids=list(ids),
        )
        if self._loaded and lay.residue_level:
            self._assert_alignment(batch, residue_counts)
        return batch.to(self.device)

    def _assert_alignment(self, batch: EncodedBatch, counts: Sequence[int]) -> None:
        """The invariant that makes silent misalignment impossible.

        Right-padding means residue positions are always a prefix-anchored span,
        so a per-row bounds check is sufficient and cheap.
        """
        input_ids = batch.model_inputs["input_ids"]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        for i, count in enumerate(counts):
            hi = self._layout.n_prefix + count
            if hi + self._layout.n_suffix > input_ids.shape[1]:
                raise RuntimeError(
                    f"{self.spec.hf_id}: row {i} needs {hi + self._layout.n_suffix} "
                    f"tokens for {count} residues but got {input_ids.shape[1]}"
                )
            if pad_id is not None and bool((input_ids[i, :hi] == pad_id).any()):
                raise RuntimeError(
                    f"{self.spec.hf_id}: padding found inside the residue span of "
                    f"row {i}; the tokenizer must right-pad"
                )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _normalize_states(
        self, hidden_states: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Make every layer equal what a model truncated there would emit.

        Applies the final norm to each intermediate state, and to the last state
        too if the model did not already apply it.
        """
        states = list(hidden_states)
        if self._final_norm is not None:
            last = len(states) - 1
            for i in range(len(states)):
                if i == last and self.final_state_is_normed:
                    continue
                states[i] = self._final_norm(states[i])
        return torch.stack(states, dim=0)

    def forward(self, batch: EncodedBatch) -> LayerStack:
        self._require_loaded()
        with torch.no_grad():
            out = self.model(**batch.model_inputs, output_hidden_states=True)
            # Kept inside no_grad: the final norm carries learned parameters, so
            # applying it outside would attach grad to every extracted state.
            states = self._normalize_states(out.hidden_states)
        return LayerStack(
            states=states,
            residue_index=batch.residue_index,
            lengths=batch.lengths,
        )

    # ------------------------------------------------------------------
    # truncation
    # ------------------------------------------------------------------
    def truncate(self, layer: int):
        """Return a standalone model whose ``last_hidden_state`` is layer ``layer``.

        ``layer`` indexes the same way as extraction: 0 is the embedding output
        (a model with no transformer blocks), ``n_layers`` is the full model.
        """
        self._require_loaded()
        if not 0 <= layer <= self.n_layers:
            raise ValueError(f"layer must be in [0, {self.n_layers}], got {layer}")

        truncated = copy.deepcopy(self.model)
        owner, attr, blocks = locate_blocks(truncated)
        setattr(owner, attr, nn.ModuleList(list(blocks)[:layer]))
        _set_depth(truncated.config, layer)
        self._resize_layer_dependent_heads(truncated, layer)
        truncated.eval()
        return truncated

    @staticmethod
    def _resize_layer_dependent_heads(model: nn.Module, layer: int) -> None:
        """Shrink auxiliary heads whose input width is a function of depth.

        ESM's ``contact_head`` consumes the attention maps of every layer, so its
        regression is ``Linear(n_layers * n_heads, 1)``. Deep-copying it unchanged
        writes a checkpoint that no longer matches the config we just rewrote, and
        ``from_pretrained`` fails with a size mismatch. Attention maps are ordered
        layer-major, so the first ``layer * n_heads`` columns are exactly the
        retained layers' weights.
        """
        head = getattr(model, "contact_head", None)
        regression = getattr(head, "regression", None)
        if regression is None:
            return
        n_heads = getattr(model.config, "num_attention_heads", None)
        if not isinstance(n_heads, int):
            return
        keep = layer * n_heads
        if keep == regression.in_features:
            return
        resized = nn.Linear(
            keep, regression.out_features, bias=regression.bias is not None
        )
        with torch.no_grad():
            resized.weight.copy_(regression.weight[:, :keep])
            if regression.bias is not None:
                resized.bias.copy_(regression.bias)
        head.regression = resized
        if hasattr(head, "in_features"):
            head.in_features = keep
