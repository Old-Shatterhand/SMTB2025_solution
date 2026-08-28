"""Saving a truncated model: verification, reload, and provenance."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoModel  # noqa: E402

from plmlayer.adapters import HFAdapter  # noqa: E402
from plmlayer.registry import resolve_spec  # noqa: E402
from plmlayer.select import SelectionResult  # noqa: E402
from plmlayer.truncate import save_truncated, verify_truncation  # noqa: E402

pytestmark = pytest.mark.weights


@pytest.fixture(scope="module")
def adapter():
    return HFAdapter(resolve_spec("esm2_8m")).load()


def _result(layer: int) -> SelectionResult:
    return SelectionResult(
        model="facebook/esm2_t6_8M_UR50D", dataset="synthetic", task="regression",
        metric="pearson", probe="knn",
        best_layer=layer, best_score=0.60, last_layer=6, last_layer_score=0.50,
        curve={layer: 0.60, 6: 0.50}, n_train=100, n_val=50,
        seed_layers=[layer] * 3, seed_agreement=1.0,
    )


@pytest.mark.parametrize("layer", [0, 1, 2, 4, 6])
def test_saved_model_reloads_and_reproduces_the_layer(adapter, tmp_path, layer):
    """The deliverable must survive a round-trip through from_pretrained.

    ESM's contact_head is Linear(n_layers * n_heads, 1), so a truncated model
    whose head was left at full width writes a checkpoint that no longer matches
    its own config -- which only surfaces on reload.
    """
    out = save_truncated(adapter, _result(layer), tmp_path / f"l{layer}")

    reloaded = AutoModel.from_pretrained(out, add_pooling_layer=False).eval()
    assert reloaded.config.num_hidden_layers == layer

    check = verify_truncation(adapter, reloaded, layer)
    assert check.passed, f"reloaded model drifted: {check.max_abs_diff:.3e}"


def test_contact_head_matches_the_retained_depth(adapter):
    n_heads = adapter.model.config.num_attention_heads
    for layer in (0, 2, 6):
        head = adapter.truncate(layer).contact_head.regression
        assert head.in_features == layer * n_heads


def test_provenance_is_recorded(adapter, tmp_path):
    out = save_truncated(adapter, _result(2), tmp_path / "prov")
    meta = json.loads((out / "plmlayer.json").read_text())
    assert meta["base_model"] == "facebook/esm2_t6_8M_UR50D"
    assert meta["layers_kept"] == 2 and meta["layers_total"] == 6
    assert meta["reload_max_abs_diff"] <= 1e-4

    card = (out / "README.md").read_text()
    assert "truncated to 2 of 6 layers" in card
    assert "plmlayer" in card and meta["dataset"] in card


def test_truncated_model_is_smaller(adapter, tmp_path):
    small = AutoModel.from_pretrained(
        save_truncated(adapter, _result(2), tmp_path / "s"), add_pooling_layer=False
    )
    n_small = sum(p.numel() for p in small.parameters())
    n_full = sum(p.numel() for p in adapter.model.parameters())
    assert n_small < n_full


def test_verification_rejects_a_mismatched_model(adapter):
    """A model that does not reproduce the layer must never be saved."""
    check = verify_truncation(adapter, adapter.truncate(6), layer=2)
    assert not check.passed and check.max_abs_diff > 1.0
