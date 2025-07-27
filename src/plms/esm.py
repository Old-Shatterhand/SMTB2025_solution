from pathlib import Path
from argparse import ArgumentParser
import pickle

from transformers import AutoModel, AutoTokenizer, T5EncoderModel
import torch
import pandas as pd
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_esm(model_name: str, data_path: Path, output_path: Path):
    """
    Run ESM model to extract embeddings for sequences in the given data path.

    :param model_name: Name of the ESM model to use.
    :param num_layers: Number of layers to extract embeddings from.
    :param data_path: Path to the CSV file containing sequences.
    :param output_path: Path to save the extracted embeddings.
    """
    num_layers = int(model_name.split("_")[1][1:])
    (out := Path(output_path)).mkdir(parents=True, exist_ok=True)
    for i in range(num_layers + 1):
        (out / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        inputs = tokenizer(seq)
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor(inputs["input_ids"]).reshape(1, -1).to(DEVICE),
                attention_mask=torch.tensor(inputs["attention_mask"]).reshape(1, -1).to(DEVICE),
                output_attentions=False,
                output_hidden_states=True,
            )
        for i, layer in enumerate(outputs.hidden_states):
            with open(out / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                pickle.dump(layer[0, 1: -1].cpu().numpy().mean(axis=0), f)
    print(f"Embedded all sequences with {model_name}")


def run_ankh(model_name: str, data_path: Path, output_path: Path):
    data = pd.read_csv(data_path)
    num_layers = 48
    (out := Path(output_path)).mkdir(parents=True, exist_ok=True)
    for i in range(num_layers + 1):
        (out / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        tokens = tokenizer.encode(seq, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = model(tokens, output_hidden_states=True)

        for i in range(0, num_layers + 1):
            with open(out / f"{idx}.pkl", "wb") as f:
                pickle.dump(embeddings.hidden_states[i][0, 1:].mean(axis=0), f)


def run_esm_batched(model_name: str, num_layers: int, data_path: str, output_path: str, batch_size: int = 16):
    """Not in use, but kept for reference."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(num_layers + 1):
        (out / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    ids = df["ID"].tolist()
    sequences = df["sequence"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    for start in tqdm(range(0, len(sequences), batch_size), desc="Embedding batches"):
        end = start + batch_size
        batch_ids = ids[start:end]
        batch_seqs = sequences[start:end]

        batch_tokens = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch_tokens["input_ids"].to(DEVICE)
        attention_mask = batch_tokens["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
            )

        hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors

        for b_idx, sample_id in enumerate(batch_ids):
            trimmed = [layer[b_idx, 1:-1].cpu().numpy() for layer in hidden_states]
            for i, layer in enumerate(trimmed):
                with open(out / f"layer_{i}" / f"{sample_id}.pkl", "wb") as f:
                    pickle.dump(layer, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="data/esm_embeddings")
    args = parser.parse_args()

    if "ankh" in args.model_name:
        run_ankh(args.model_name, args.data_path, args.output_path)
    else:
        run_esm(args.model_name, args.data_path, args.output_path)
