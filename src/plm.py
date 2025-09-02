import re
import pickle
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, BertTokenizer, T5EncoderModel, T5Model, T5Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AA_OHE = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7,
    "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19
}

PLM_MODELS = {
    "esm_t6": "facebook/esm2_t6_8M_UR50D",
    "esm_t12": "facebook/esm2_t12_35M_UR50D",
    "esm_t30": "facebook/esm2_t30_150M_UR50D",
    "esm_t33": "facebook/esm2_t33_650M_UR50D",
    "esm_t36": "facebook/esm2_t36_3B_UR50D",
    "ankh-base": "ElnaggarLab/ankh-base",
    "ankh-large": "ElnaggarLab/ankh-large",
}


def run_esm(model_name: str, data_path: Path, output_path: Path):
    """
    Run ESM model to extract embeddings for sequences in the given data path.

    :param model_name: Name of the ESM model to use.
    :param num_layers: Number of layers to extract embeddings from.
    :param data_path: Path to the CSV file containing sequences.
    :param output_path: Path to save the extracted embeddings.
    """
    for i in range(int(model_name.split("_")[1][1:]) + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        inputs = tokenizer(seq[:1022])
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor(inputs["input_ids"]).reshape(1, -1).to(DEVICE),
                attention_mask=torch.tensor(inputs["attention_mask"]).reshape(1, -1).to(DEVICE),
                output_attentions=False,
                output_hidden_states=True,
            )
            del inputs
        for i, layer in enumerate(outputs.hidden_states):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                pickle.dump(layer[0, 1: -1].cpu().numpy().mean(axis=0), f)
        del outputs
    print(f"Embedded all sequences with {model_name}")


def run_esmc(model_name: str, data_path: Path, output_path: Path):
    data = pd.read_csv(data_path)
    num_layers = 30 if "300m" in model_name else 36
    for i in range(num_layers + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    model = ESMC.from_pretrained(model_name).to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        with torch.no_grad():
            logits_output = model.logits(
                model.encode(ESMProtein(sequence=seq[:1022])),
                LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
            )
        for i in range(0, num_layers):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                pickle.dump(logits_output.hidden_states[i, 0, 1:-1].cpu().float().numpy().mean(axis=0), f)
        with open(output_path / f"layer_{num_layers}" / f"{idx}.pkl", "wb") as f:
            pickle.dump(logits_output.embeddings[0, 1:-1].cpu().float().numpy().mean(axis=0), f)
        del logits_output
    print(f"Embedded all sequences with {model_name}")


def run_ankh(model_name: str, data_path: Path, output_path: Path):
    data = pd.read_csv(data_path)
    num_layers = 48
    for i in range(num_layers + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        tokens = tokenizer.encode(seq[:1022], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = model(tokens, output_hidden_states=True)
            del tokens

        for i in range(0, num_layers + 1):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                pickle.dump(embeddings.hidden_states[i][0, 1:].cpu().numpy().mean(axis=0), f)
        del embeddings


def run_prostt5(data_path: Path, output_path: Path):
    data = pd.read_csv(data_path)
    num_layers = 24
    for i in range(num_layers + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)
    
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5")
    model = T5Model.from_pretrained("Rostlab/ProstT5").to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        seq = "<AA2fold>" + " " + " ".join([aa.upper() for aa in re.sub(r"[UZOB]", "X", seq[:1022])])
        tokens = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding="longest", return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            embeddings = model.encoder(tokens.input_ids, attention_mask=tokens.attention_mask, output_attentions=False, output_hidden_states=True)
            del tokens

        for i in range(0, num_layers + 1):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                pickle.dump(embeddings.hidden_states[i][0, 1:-1].cpu().numpy().mean(axis=0), f)
        del embeddings


def run_prott5(data_path: Path, output_path: Path):
    data = pd.read_csv(data_path)
    num_layers = 24
    for i in range(num_layers + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        seq = " ".join(list(re.sub(r"[UZOB]", "X", seq[:1022])))
        tokens = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            embedding_repr = model.encoder(tokens.input_ids, attention_mask=tokens.attention_mask, output_attentions=False, output_hidden_states=True)
            del tokens

        for i in range(0, num_layers + 1):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                pickle.dump(embedding_repr.hidden_states[i][0, 1:-1].cpu().numpy().mean(axis=0), f)
        del embedding_repr


def run_ohe(data_path: Path, output_path: Path):
    """
    One-hot encode sequences and save them to the output path.
    """
    (out := (output_path / "layer_0")).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    sequences = data["sequence"].tolist()
    ids = data["ID"].tolist()

    for idx, seq in tqdm(zip(ids, sequences), total=len(sequences)):
        one_hot = np.zeros((len(seq), 21), dtype=np.float32)
        for i, aa in enumerate(seq[:1022]):
            if aa == "-":
                continue
            one_hot[i, AA_OHE.get(aa, 20)] = 1
        with open(out / f"{idx}.pkl", "wb") as f:
            pickle.dump(one_hot.mean(axis=0), f)


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
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    if "esmc" in args.model_name:
        run_esmc(args.model_name, args.data_path, args.output_path)
    elif "esm" in args.model_name:
        run_esm(PLM_MODELS[args.model_name], args.data_path, args.output_path)
    elif "ankh" in args.model_name:
        run_ankh(PLM_MODELS[args.model_name], args.data_path, args.output_path)
    elif "prott5" in args.model_name:
        run_prott5(args.data_path, args.output_path)
    elif "prostt5" in args.model_name:
        run_prostt5(args.data_path, args.output_path)
    elif "ohe" in args.model_name:
        run_ohe(args.data_path, args.output_path)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}. Supported models are: {list(PLM_MODELS.keys())} or 'ohe'.")