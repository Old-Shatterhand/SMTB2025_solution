import pickle
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM
from pathlib import Path
import torch
from tqdm import tqdm


def save_embeddings(embeddings: np.ndarray, aa_level: bool, fp) -> None:
    if aa_level:
        pickle.dump(embeddings, fp)
    else:
        pickle.dump(embeddings.mean(axis=0), fp)


def run_esm(
    model_name: str,
    data_path: Path,
    output_path: Path,
    aa_level: bool = False,
    empty: bool = False,
    force: bool = False,
) -> None:
    """
    Run ESM model to extract embeddings for sequences in the given data path.

    Args:
        model_name: Name of the ESM model to use.
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the extracted embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Whether to use an untrained model.
        force: Whether to overwrite existing embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(int(model_name.split("_")[1][1:]) + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not empty:
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    else:
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_name)).to(device).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if not force and (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        inputs = tokenizer(seq[:1022])
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor(inputs["input_ids"]).reshape(1, -1).to(device),
                attention_mask=torch.tensor(inputs["attention_mask"]).reshape(1, -1).to(device),
                output_attentions=False,
                output_hidden_states=True,
            )
            del inputs
        for i, layer in enumerate(outputs.hidden_states):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                save_embeddings(layer[0, 1:-1].cpu().numpy(), aa_level, f)
        del outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ESM model to extract embeddings.")
    parser.add_argument("model_name", type=str, help="Name of the ESM model to use.")
    parser.add_argument("data_path", type=Path, help="Path to the CSV file containing sequences.")
    parser.add_argument("output_path", type=Path, help="Path to save the extracted embeddings.")
    parser.add_argument(
        "--aa_level",
        action="store_true",
        help="Whether to save amino acid level embeddings or mean pooled embeddings.",
    )
    parser.add_argument("--empty", action="store_true", help="Whether to use an untrained model.")
    parser.add_argument("--force", action="store_true", help="Whether to overwrite existing embeddings.")
    args = parser.parse_args()

    run_esm(
        model_name=args.model_name,
        data_path=args.data_path,
        output_path=args.output_path,
        aa_level=args.aa_level,
        empty=args.empty,
        force=args.force,
    )
