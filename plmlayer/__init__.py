"""plmlayer -- pick the best layer of a protein language model for your dataset.

Implements the practical consequence of Joeres, Senatorov et al.,
*Task- and dataset-specific information in protein language models*
(arXiv:2608.12090): the last layer is almost never the best one, so probe a
subsample of your data and ship a truncated encoder instead.
"""

from plmlayer.api import load_adapter, suggest_layer
from plmlayer.data import Dataset, load_dataset
from plmlayer.registry import PRESETS, list_specs, register_spec, resolve_spec
from plmlayer.select import SelectionResult, select_layer
from plmlayer.truncate import save_truncated, verify_truncation

__version__ = "0.1.0"

__all__ = [
    "suggest_layer", "load_adapter",
    "load_dataset", "Dataset",
    "select_layer", "SelectionResult",
    "save_truncated", "verify_truncation",
    "list_specs", "register_spec", "resolve_spec", "PRESETS",
    "__version__",
]
