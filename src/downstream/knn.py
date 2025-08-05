from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--function', type=str, required=True, choices=["lr", "xgb"])
parser.add_argument('--task', type=str, default="regression", choices=["regression", "classification"])
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument("--binary", action='store_true', default=False, help="Indicator for binary classification")
args = parser.parse_args()

