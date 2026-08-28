"""Layer selection: the plateau rule, gain reporting, and stability flags."""

from __future__ import annotations

import numpy as np
import pytest

from plmlayer.data import Dataset, Split
from plmlayer.extract import EmbeddingSet
from plmlayer.select import SelectionResult, _choose, layer_schedule, select_layer


class TestSchedule:
    def test_full_covers_every_layer(self):
        assert layer_schedule(6) == [0, 1, 2, 3, 4, 5, 6]

    def test_coarse_always_includes_first_and_last(self):
        for n in (6, 12, 24, 33, 48):
            sched = layer_schedule(n, "coarse")
            assert sched[0] == 0 and sched[-1] == n
            assert len(sched) < n + 1 or n <= 8

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="unknown schedule"):
            layer_schedule(6, "nonsense")


class TestPlateauRule:
    # The real fluorescence curve from esm2_8m, where layers 2 and 4 are a
    # coin flip and the raw argmax disagreed with 3/3 resampling seeds.
    CURVE = {0: 0.4774, 1: 0.5057, 2: 0.5999, 3: 0.5845, 4: 0.6031, 5: 0.5227, 6: 0.5802}

    def test_zero_tolerance_is_plain_argmax(self):
        assert _choose(self.CURVE, "pearson", 0.0) == (4, 4, [4])

    def test_prefers_the_shallowest_indistinguishable_layer(self):
        """Same score, half the blocks -- and a far more stable choice."""
        chosen, peak, plateau = _choose(self.CURVE, "pearson", 0.02)
        assert (chosen, peak) == (2, 4)
        assert plateau == [2, 4]

    def test_wider_tolerance_widens_the_plateau(self):
        assert _choose(self.CURVE, "pearson", 0.05)[2] == [2, 3, 4, 6]

    def test_lower_is_better_metrics_invert(self):
        curve = {0: 5.0, 1: 1.0, 2: 1.01, 3: 9.0}
        chosen, peak, plateau = _choose(curve, "rmse", 0.02)
        assert peak == 1 and chosen == 1 and plateau == [1, 2]

    def test_nan_layers_are_skipped(self):
        curve = {0: float("nan"), 1: 0.4, 2: 0.8}
        assert _choose(curve, "pearson", 0.0)[0] == 2

    def test_all_nan_raises(self):
        with pytest.raises(RuntimeError, match="degenerate"):
            _choose({0: float("nan")}, "pearson", 0.0)


class TestGainReporting:
    def _result(self, **kw):
        base = dict(
            model="m", dataset="d", task="regression", metric="pearson", probe="knn",
            best_layer=2, best_score=0.60, last_layer=6, last_layer_score=0.50,
            curve={2: 0.60, 6: 0.50}, n_train=100, n_val=50,
        )
        base.update(kw)
        return SelectionResult(**base)

    def test_gain_over_last_layer(self):
        assert self._result().gain_over_last == pytest.approx(0.2)

    def test_depth_fraction(self):
        assert self._result().depth_fraction == pytest.approx(2 / 6)

    def test_zero_last_layer_score_is_not_a_division_error(self):
        assert np.isnan(self._result(last_layer_score=0.0).gain_over_last)

    def test_lower_is_better_gain_inverts(self):
        r = self._result(metric="rmse", best_score=1.0, last_layer_score=2.0)
        assert r.gain_over_last == pytest.approx(1.0)

    def test_summary_mentions_the_headline_numbers(self):
        text = self._result().summary()
        assert "best layer" in text and "gain over last" in text and "+20.0%" in text


class TestSelectLayer:
    """End-to-end selection on synthetic embeddings with a known best layer."""

    def _fixture(self, n_states=5, best=1, n=200, dim=8, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.normal(size=n)
        values = np.zeros((n_states, n, dim), dtype=np.float16)
        for layer in range(n_states):
            # signal peaks at `best` and decays away from it
            strength = 1.0 / (1.0 + 2.0 * abs(layer - best))
            signal = np.outer(y, rng.normal(size=dim)) * strength
            values[layer] = (signal + rng.normal(scale=0.3, size=(n, dim))).astype(np.float16)
        ids = [f"P{i}" for i in range(n)]
        emb = EmbeddingSet(ids=ids, values=values)
        cut = n // 2
        ds = Dataset(
            name="synthetic", task="regression",
            train=Split(ids[:cut], ["A"] * cut, y[:cut]),
            val=Split(ids[cut:], ["A"] * (n - cut), y[cut:]),
        )
        train = EmbeddingSet(ids[:cut], values[:, :cut])
        val = EmbeddingSet(ids[cut:], values[:, cut:])
        return ds, train, val, emb

    def test_finds_the_planted_layer(self):
        ds, train, val, _ = self._fixture(best=1)
        res = select_layer(ds, train, val, n_seeds=0, tolerance=0.0)
        assert res.best_layer == 1

    def test_curve_covers_every_layer(self):
        ds, train, val, _ = self._fixture()
        res = select_layer(ds, train, val, n_seeds=0)
        assert sorted(res.curve) == [0, 1, 2, 3, 4]

    def test_notes_when_the_last_layer_wins(self):
        ds, train, val, _ = self._fixture(best=4)
        res = select_layer(ds, train, val, n_seeds=0, tolerance=0.0)
        assert res.best_layer == 4
        assert any("last layer won" in n for n in res.notes)

    def test_seed_agreement_is_reported(self):
        ds, train, val, _ = self._fixture(best=1)
        res = select_layer(ds, train, val, n_seeds=3, tolerance=0.0)
        assert len(res.seed_layers) == 3
        assert 0.0 <= res.seed_agreement <= 1.0

    def test_coarse_schedule_is_flagged(self):
        ds, train, val, _ = self._fixture()
        res = select_layer(ds, train, val, n_seeds=0, schedule="coarse")
        assert any("coarse" in n for n in res.notes)

    def test_result_serialises(self):
        ds, train, val, _ = self._fixture()
        d = select_layer(ds, train, val, n_seeds=0).to_dict()
        assert "gain_over_last" in d and "curve" in d
        import json

        json.loads(json.dumps(d, default=str))
