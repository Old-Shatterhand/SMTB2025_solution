"""Adapter + truncation contract, exercised against the smallest real ESM-2.

Marked ``weights`` because it needs the checkpoint; it is ~8M params and runs on
CPU in a couple of seconds.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from plmlayer.adapters import HFAdapter  # noqa: E402
from plmlayer.registry import resolve_spec  # noqa: E402

pytestmark = pytest.mark.weights

CKPT = "esm2_8m"
SEQS = ["MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY"]


@pytest.fixture(scope="module")
def adapter():
    return HFAdapter(resolve_spec(CKPT)).load()


def test_introspection_is_measured_not_parsed(adapter):
    assert adapter.n_states == 7  # 6 blocks + embedding output
    assert adapter.n_layers == 6
    assert adapter.hidden_size == 320
    assert adapter.supports_residue_level


def test_calibrated_layout_matches_the_declared_one(adapter):
    assert (adapter.layout.n_prefix, adapter.layout.n_suffix) == (1, 1)
    assert adapter.calibration.declared_matches is True


def test_no_random_pooler_is_attached(adapter):
    """AutoModel would bolt on a randomly-initialised pooler; it must not."""
    assert getattr(adapter.model, "pooler", None) is None


def test_forward_shape(adapter):
    stack = adapter.forward(adapter.encode(SEQS, ["a", "b"]))
    s, b, _, d = stack.states.shape
    assert (s, b, d) == (adapter.n_states, 2, adapter.hidden_size)


def test_residue_mask_selects_exactly_the_residues(adapter):
    batch = adapter.encode(SEQS, ["a", "b"])
    mask = adapter.forward(batch).residue_mask()
    assert mask.sum(dim=1).tolist() == [len(s) for s in SEQS]


@pytest.mark.parametrize("layer", range(7))
def test_truncated_model_reproduces_the_probed_layer(adapter, layer):
    """The contract the whole tool rests on.

    "Probe layer k" and "ship a k-block model" must mean the same thing. They do
    only because the final norm is applied to every intermediate layer during
    extraction -- without it this is off by ~32 in absolute magnitude.
    """
    batch = adapter.encode(SEQS, ["a", "b"])
    reference = adapter.forward(batch).states[layer]

    truncated = adapter.truncate(layer)
    assert truncated.config.num_hidden_layers == layer
    with torch.no_grad():
        got = truncated(**batch.model_inputs).last_hidden_state

    assert torch.allclose(got, reference, atol=1e-5), (
        f"layer {layer}: max|diff| = {(got - reference).abs().max():.3e}"
    )


def test_raw_hidden_states_would_not_match(adapter):
    """Pin the trap itself, so a refactor cannot quietly reintroduce it."""
    batch = adapter.encode(SEQS, ["a", "b"])
    with torch.no_grad():
        raw = adapter.model(**batch.model_inputs, output_hidden_states=True).hidden_states
        got = adapter.truncate(4)(**batch.model_inputs).last_hidden_state
    assert not torch.allclose(got, raw[4], atol=1e-3)
    assert (got - raw[4]).abs().max() > 1.0


def test_truncate_rejects_out_of_range(adapter):
    with pytest.raises(ValueError, match=r"\[0, 6\]"):
        adapter.truncate(7)
