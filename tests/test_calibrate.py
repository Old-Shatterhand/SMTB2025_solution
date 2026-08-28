"""Calibration contract. Requires no model weights."""

from __future__ import annotations

import pytest

from plmlayer.calibrate import calibrate
from plmlayer.registry import resolve_spec
from plmlayer.types import CalibrationError, ModelSpec
from tests import fake_tokenizers as fk


def _layout(tok, key: str):
    return calibrate(tok, resolve_spec(key)).layout


def test_esm_layout_is_cls_eos():
    lay = _layout(fk.esm_like(), "esm2_8m")
    assert (lay.n_prefix, lay.n_suffix) == (1, 1)
    assert lay.residue_level


def test_prott5_has_no_bos():
    lay = _layout(fk.prott5_like(), "prott5")
    assert (lay.n_prefix, lay.n_suffix) == (0, 1)


def test_prostt5_direction_token_is_counted_as_prefix():
    """The bug special_tokens_mask cannot see.

    <AA2fold> is an ordinary added token, so a mask-based approach reports it as
    a residue and shifts every index by one. Calibration must find n_prefix=1.
    """
    tok = fk.prostt5_like()
    lay = _layout(tok, "prostt5")
    assert (lay.n_prefix, lay.n_suffix) == (1, 1)

    # Demonstrate the failure being avoided: the mask sees only </s>.
    ids = tok("<AA2fold> M K", add_special_tokens=True)["input_ids"]
    visible = [i for i in ids if i in tok.all_special_ids]
    assert len(visible) == 1, "mask would hide only </s>, missing <AA2fold>"


def test_progen2_direction_flags_are_ordinary_tokens():
    tok = fk.progen2_like()
    lay = _layout(tok, "progen2_small")
    assert (lay.n_prefix, lay.n_suffix) == (1, 1)
    assert tok.all_special_ids == [], "mask would be all zeros here"


def test_ankh_has_no_bos_so_the_research_slice_is_wrong():
    """src/plm.py:225 slices [0, 1:], which assumes a BOS Ankh does not have."""
    lay = _layout(fk.ankh_like(), "ankh_base")
    assert (lay.n_prefix, lay.n_suffix) == (0, 1)
    assert (lay.n_prefix, lay.n_suffix) != (1, 0)


def test_subword_tokenizer_disables_residue_level():
    rep = calibrate(fk.protgpt2_like(), resolve_spec("protgpt2"))
    assert rep.layout.residue_level is False
    assert "subword" in rep.detail


def test_unknown_residue_raises_with_actionable_message():
    tok = fk.FakeTokenizer(("<cls>",), ("<eos>",), vocab="ACDEFGHIKLMNPQRSTVWY")
    with pytest.raises(CalibrationError, match=r"<unk>"):
        calibrate(tok, ModelSpec(key="t", hf_id="t"))


def test_declared_layout_mismatch_raises():
    """A tokenizer change under a pinned spec must fail loudly, not silently."""
    spec = ModelSpec(key="t", hf_id="t", n_prefix=0, n_suffix=1)
    with pytest.raises(CalibrationError, match="declared layout"):
        calibrate(fk.esm_like(), spec)
