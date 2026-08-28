"""Extraction: batching, pooling, budgets and bounds."""

from __future__ import annotations

import numpy as np
import pytest

from plmlayer.extract import _batches, estimate_bytes


class TestBatching:
    def test_every_item_appears_exactly_once(self):
        lengths = [10, 200, 5, 60, 5, 300, 12]
        flat = [i for b in _batches(lengths, 1000, 4) for i in b]
        assert sorted(flat) == list(range(len(lengths)))

    def test_token_budget_is_respected(self):
        lengths = [50] * 20
        for batch in _batches(lengths, 200, 64):
            assert max(lengths[i] for i in batch) * len(batch) <= 200

    def test_batch_size_cap_is_respected(self):
        assert all(len(b) <= 3 for b in _batches([1] * 20, 10_000, 3))

    def test_batches_are_length_sorted_to_limit_padding(self):
        lengths = [100, 1, 100, 1]
        first = _batches(lengths, 10_000, 2)[0]
        assert {lengths[i] for i in first} == {1}

    def test_an_oversized_item_still_gets_a_batch(self):
        assert len(_batches([10_000], 100, 8)) == 1


class TestMemoryEstimate:
    def test_matches_the_documented_arithmetic(self):
        # ankh_large: 49 states x 1536 dims, fp16 -> ~147 KiB per sequence
        assert estimate_bytes(1, 49, 1536) == 49 * 1536 * 2
        assert estimate_bytes(1, 49, 1536) / 1024 == pytest.approx(147, abs=1)

    def test_scales_linearly(self):
        assert estimate_bytes(100, 7, 320) == 100 * estimate_bytes(1, 7, 320)


def test_embedding_set_layer_view_is_float32():
    from plmlayer.extract import EmbeddingSet

    es = EmbeddingSet(ids=["a", "b"], values=np.zeros((3, 2, 4), dtype=np.float16))
    assert es.n_states == 3 and es.n_items == 2
    assert es.layer(0).dtype == np.float32
    assert es.layer(0).shape == (2, 4)
