"""Dataset loading, task inference and subsampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmlayer.data import DEEPLOC_LABELS, infer_task, load_dataset


def _write(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _frame(n=200, **cols):
    base = {
        "ID": [f"P{i:05d}" for i in range(n)],
        "sequence": ["ACDEFGHIKLMNPQRSTVWY"[: 6 + i % 12] for i in range(n)],
    }
    base.update(cols)
    return pd.DataFrame(base)


class TestInferTask:
    def test_continuous_is_regression(self):
        rng = np.random.default_rng(0)
        assert infer_task(rng.normal(size=500), ["label"]) == "regression"

    def test_two_values_is_binary(self):
        assert infer_task(np.array([0, 1] * 50), ["label"]) == "binary"

    def test_dense_integer_codes_are_multi_class(self):
        assert infer_task(np.repeat(np.arange(7), 30), ["class_"]) == "multi-class"

    def test_sparse_integer_codes_are_still_multi_class(self):
        """SCOPe's `sampled` filter leaves gaps in the fold codes.

        An earlier density-based rule called these regression; cardinality
        relative to row count is the robust signal.
        """
        codes = np.repeat(np.arange(0, 1200, 3), 8)  # 400 classes, gaps of 3
        assert infer_task(codes, ["fold"]) == "multi-class"

    def test_identifier_like_column_is_not_a_class(self):
        assert infer_task(np.arange(500), ["ID"]) == "regression"

    def test_several_columns_is_multi_label(self):
        assert infer_task(np.zeros((10, 3)), ["a", "b", "c"]) == "multi-label"


class TestLoadDataset:
    def test_requires_id_and_sequence(self, tmp_path):
        path = _write(tmp_path, "bad.csv", pd.DataFrame({"seq": ["AC"], "label": [1]}))
        with pytest.raises(ValueError, match="missing required column"):
            load_dataset(path)

    def test_reports_missing_label_column(self, tmp_path):
        path = _write(tmp_path, "nolabel.csv", _frame())
        with pytest.raises(ValueError, match="no label column"):
            load_dataset(path)

    def test_labels_column_is_accepted(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(labels=np.arange(200) * 0.5))
        assert load_dataset(path).task == "regression"

    @pytest.mark.parametrize("val_name", ["valid", "val"])
    def test_both_validation_split_spellings(self, tmp_path, val_name):
        """HF-derived sets say 'valid'; locally split ones say 'val'."""
        split = ["train"] * 150 + [val_name] * 50
        path = _write(tmp_path, f"{val_name}.csv", _frame(label=np.arange(200) * 1.0, split=split))
        ds = load_dataset(path, max_train=None, max_val=None)
        assert (len(ds.train), len(ds.val)) == (150, 50)

    def test_split_column_without_validation_level_raises(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(label=np.arange(200) * 1.0, split=["train"] * 100 + ["test"] * 100))
        with pytest.raises(ValueError, match="no .* level"):
            load_dataset(path)

    def test_splits_are_generated_when_absent(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(label=np.arange(200) * 1.0))
        ds = load_dataset(path, max_train=None, max_val=None, val_fraction=0.25)
        assert len(ds.val) == 50 and len(ds.train) == 150

    def test_subsampling_respects_the_budget(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(1000, label=np.arange(1000) * 1.0))
        ds = load_dataset(path, max_train=100, max_val=40)
        assert len(ds.train) <= 100 and len(ds.val) <= 40

    def test_stratified_subsampling_keeps_every_class(self, tmp_path):
        """Rare classes must survive the cut, or the probe silently degrades."""
        labels = np.concatenate([np.repeat(np.arange(20), 49), np.arange(20)])
        path = _write(tmp_path, "d.csv", _frame(len(labels), label=labels))
        ds = load_dataset(path, max_train=200, max_val=100, task="multi-class")
        assert len(np.unique(ds.train.y)) == 20

    def test_sampled_column_is_honoured(self, tmp_path):
        sampled = [True] * 60 + [False] * 140
        path = _write(tmp_path, "d.csv", _frame(label=np.arange(200) * 1.0, sampled=sampled))
        ds = load_dataset(path, max_train=None, max_val=None)
        assert len(ds.train) + len(ds.val) == 60

    def test_deeploc_columns_become_multi_label(self, tmp_path):
        cols = {c: np.random.default_rng(0).integers(0, 2, 200) for c in DEEPLOC_LABELS}
        path = _write(tmp_path, "deeploc.csv", _frame(**cols))
        ds = load_dataset(path)
        assert ds.task == "multi-label" and ds.n_classes == 10
        assert ds.train.y.shape[1] == 10

    def test_residue_level_requires_positions(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(label=np.arange(200) * 1.0))
        with pytest.raises(ValueError, match="positions"):
            load_dataset(path, residue_level=True)

    def test_positions_parsed_without_eval(self, tmp_path):
        """The research code runs eval() on this column; literal_eval is safe."""
        path = _write(tmp_path, "d.csv", _frame(20, label=np.arange(20) * 1.0, positions=["[0, 1, 2]"] * 20))
        ds = load_dataset(path, residue_level=True, max_train=None, max_val=None)
        assert ds.train.positions[0] == [0, 1, 2]

    def test_malicious_positions_do_not_execute(self, tmp_path):
        path = _write(tmp_path, "d.csv", _frame(20, label=np.arange(20) * 1.0,
                                                positions=["__import__('os').system('touch /tmp/pwned')"] * 20))
        with pytest.raises((ValueError, SyntaxError)):
            load_dataset(path, residue_level=True)
