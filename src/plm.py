import os

# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/s8rojoer/.cache"
# os.environ.setdefault("LOGNAME", "s8rojoer")

import re
import pickle
from pathlib import Path
from argparse import ArgumentParser
import sys

from tokenizers import Tokenizer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM

from src.viz.constants import LAYERS, PLM_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AA_OHE = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}


def save_embeddings(embeddings: np.ndarray, aa_level: bool, fp, positions: list | None = None) -> None:
    """
    Save embeddings to a file, either at amino acid level or mean pooled.
    
    Args:
        embeddings: The embeddings to save.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        fp: The file pointer to save the embeddings to.
        positions: Optional list of positions to select from the embeddings (only used if aa_level is True).
    """
    if positions is not None and aa_level is False:
        raise ValueError("Positions can only be used when aa_level is True.")
    if aa_level:
        if positions is not None:
            embeddings = embeddings[positions]
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
    data = pd.read_csv(data_path)
    if "positions" in data.columns:
        data["positions"] = data["positions"].apply(eval)

    for i in range(int(model_name.split("_")[1][1:]) + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not empty:
        model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    else:
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_name)).to(DEVICE).eval()

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing sequences"):
        if not force and (output_path / "layer_0" / f"{row['ID']}.pkl").exists():
            continue
        inputs = tokenizer(row["sequence"], truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor(inputs["input_ids"]).reshape(1, -1).to(DEVICE),
                attention_mask=torch.tensor(inputs["attention_mask"]).reshape(1, -1).to(DEVICE),
                output_attentions=False,
                output_hidden_states=True,
            )
            del inputs
        for i, layer in enumerate(outputs.hidden_states):
            with open(output_path / f"layer_{i}" / f"{row['ID']}.pkl", "wb") as f:
                save_embeddings(layer[0, 1:-1].cpu().numpy(), aa_level, f, row.get("positions", None))
        del outputs


def run_esmc(
    model_name: str, data_path: Path, output_path: Path, aa_level: bool = False, empty: bool = False, force: bool = False
) -> None:
    """
    Run ESMC model to extract embeddings for sequences in the given data path.

    Args:
        model_name: Name of the ESMC model to use.
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the extracted embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Whether to use an untrained model.
        force: Whether to overwrite existing embeddings.
    """
    data = pd.read_csv(data_path)
    num_layers = 30 if "300m" in model_name else 36
    for i in range(num_layers + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    if not empty:
        model = ESMC.from_pretrained(model_name.replace("-", "_")).to(DEVICE).eval()
    else:
        if "300m" in model_name:
            model = ESMC(n_layers=30, n_heads=15, d_model=960, tokenizer=EsmSequenceTokenizer()).to(DEVICE).eval()
        else:
            model = ESMC(n_layers=36, n_heads=18, d_model=1152, tokenizer=EsmSequenceTokenizer()).to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if not force and (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        with torch.no_grad():
            logits_output = model.logits(
                model.encode(ESMProtein(sequence=seq[:1022])),
                LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True),
            )
        for i in range(0, num_layers):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                save_embeddings(logits_output.hidden_states[i, 0, 1:-1].cpu().float().numpy(), aa_level, f)
        with open(output_path / f"layer_{num_layers}" / f"{idx}.pkl", "wb") as f:
            save_embeddings(logits_output.embeddings[0, 1:-1].cpu().float().numpy(), aa_level, f)
        del logits_output


def run_ankh(
    model_name: str, data_path: Path, output_path: Path, aa_level: bool = False, empty: bool = False, force: bool = False
) -> None:
    """
    Run Ankh model to extract embeddings for sequences in the given data path.

    Args:
        model_name: Name of the Ankh model to use.
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the extracted embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Whether to use an untrained model.
        force: Whether to overwrite existing embeddings.
    """
    data = pd.read_csv(data_path)
    if "positions" in data.columns:
        data["positions"] = data["positions"].apply(eval)
    
    for i in range(49):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not empty:
        model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    else:
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_name)).to(DEVICE).eval()

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing sequences"):
        if not force and (output_path / "layer_0" / f"{row['ID']}.pkl").exists():
            continue
        tokens = tokenizer.encode(row["sequence"][:1022].replace("<mask>", "<extra_id_0>"), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = model.encoder(tokens, output_hidden_states=True)
            del tokens

        for i in range(0, 49):
            with open(output_path / f"layer_{i}" / f"{row['ID']}.pkl", "wb") as f:
                save_embeddings(embeddings.hidden_states[i][0, 1:].cpu().numpy(), aa_level, f, row.get("positions", None))
        del embeddings


def run_prostt5(data_path: Path, output_path: Path, aa_level: bool = False, empty: bool = False, force: bool = False) -> None:
    """
    Run ProstT5 model to extract embeddings for sequences in the given data path.

    Args:
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the extracted embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Whether to use an untrained model.
        force: Whether to overwrite existing embeddings.
    """
    data = pd.read_csv(data_path)
    if "positions" in data.columns:
        data["positions"] = data["positions"].apply(eval)

    for i in range(25):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(PLM_MODELS["prostt5"])
    if not empty:
        model = AutoModel.from_pretrained(PLM_MODELS["prostt5"]).to(DEVICE).eval()
    else:
        model = AutoModel.from_config(AutoConfig.from_pretrained(PLM_MODELS["prostt5"])).to(DEVICE).eval()

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing sequences"):
        if not force and (output_path / "layer_0" / f"{row['ID']}.pkl").exists():
            continue
        seq = "<AA2fold>" + " " + " ".join([aa.upper() for aa in re.sub(r"[UZOB]", "X", row["sequence"][:1022])])
        tokens = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            embeddings = model.encoder(
                tokens.input_ids, attention_mask=tokens.attention_mask, output_attentions=False, output_hidden_states=True
            )
            del tokens

        for i in range(0, 25):
            with open(output_path / f"layer_{i}" / f"{row['ID']}.pkl", "wb") as f:
                save_embeddings(embeddings.hidden_states[i][0, 1:-1].cpu().numpy(), aa_level, f, row.get("positions", None))
        del embeddings


def run_prott5(data_path: Path, output_path: Path, aa_level: bool = False, empty: bool = False, force: bool = False) -> None:
    """
    Run ProtT5 model to extract embeddings for sequences in the given data path.

    Args:
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the extracted embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Whether to use an untrained model.
        force: Whether to overwrite existing embeddings.
    """
    data = pd.read_csv(data_path)
    if "positions" in data.columns:
        data["positions"] = data["positions"].apply(eval)

    for i in range(25):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(PLM_MODELS["prott5"])
    if not empty:
        model = AutoModel.from_pretrained(PLM_MODELS["prott5"]).to(DEVICE).eval()
    else:
        model = AutoModel.from_config(AutoConfig.from_pretrained(PLM_MODELS["prott5"])).to(DEVICE).eval()

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing sequences"):
        if not force and (output_path / "layer_0" / f"{row['ID']}.pkl").exists():
            continue
        seq = " ".join([x if x != "?" else "<extra_id_0>" for x in re.sub(r"[UZOB]", "X", row["sequence"][:1022].replace("<mask>", "?").upper())])
        tokens = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            embedding_repr = model.encoder(
                tokens.input_ids, attention_mask=tokens.attention_mask, output_attentions=False, output_hidden_states=True
            )
            del tokens

        for i in range(0, 25):
            with open(output_path / f"layer_{i}" / f"{row['ID']}.pkl", "wb") as f:
                save_embeddings(embedding_repr.hidden_states[i][0, :-1].cpu().numpy(), aa_level, f, row.get("positions", None))
        del embedding_repr


def run_progen2(
    model_name: str, data_path: Path, output_path: Path, aa_level: bool = False, empty: bool = False, force: bool = False
) -> None:
    """
    Run ProGen2 model to extract embeddings for sequences in the given data path.

    Args:
        model_name: Name of the ProGen2 model to use.
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the extracted embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Whether to use an untrained model.
        force: Whether to overwrite existing embeddings.
    """
    data = pd.read_csv(data_path)
    if "positions" in data.columns:
        data["positions"] = data["positions"].apply(eval)

    for i in range(LAYERS[model_name] + 1):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_pretrained(PLM_MODELS[model_name])
    if not empty:
        model = (
            AutoModelForCausalLM.from_pretrained(
                PLM_MODELS[model_name], trust_remote_code=True  # , cache_dir="/home/s8rojoer/.cache/huggingface/"
            ).to(DEVICE).eval()
        )
    else:
        raise NotImplementedError("Empty ProGen2 model is not implemented.")

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing sequences"):
        if not force and (output_path / "layer_0" / f"{row['ID']}.pkl").exists():
            continue
        input_ids = torch.tensor(tokenizer.encode(f"1{row['sequence'][:1022]}2").ids).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            del input_ids
        for i, layer in enumerate(outputs.hidden_states[:-1]):
            with open(output_path / f"layer_{i}" / f"{row['ID']}.pkl", "wb") as f:
                save_embeddings(layer[0, 1:-1].cpu().numpy(), aa_level, f)

        # treat last layer differently, because ProGen2 developers are crazy!?
        with open(output_path / f"layer_{LAYERS[model_name]}" / f"{row['ID']}.pkl", "wb") as f:
            save_embeddings(outputs.hidden_states[-1][1:-1].cpu().numpy(), aa_level, f, row.get("positions", None))

        del outputs


def run_protgpt2(data_path: Path, output_path: Path, empty: bool = False, force: bool = False) -> None:
    """
    Run ProtGPT2 model to extract embeddings for sequences in the given data path.

    Args:
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the extracted embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Whether to use an untrained model.
        force: Whether to overwrite existing embeddings.
    """
    data = pd.read_csv(data_path)

    for i in range(37):
        (output_path / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2", trust_remote_code=True)
    if not empty:
        model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", trust_remote_code=True).to(DEVICE).eval()
    else:
        raise NotImplementedError("Empty ProtGPT2 model is not implemented.")

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        if not force and (output_path / "layer_0" / f"{idx}.pkl").exists():
            continue
        tokens = torch.tensor(tokenizer.encode(seq[:1022])).to(DEVICE).reshape(1, -1)
        with torch.no_grad():
            outputs = model(
                input_ids=tokens,
                output_hidden_states=True,
            )
            del tokens
        for i, layer in enumerate(outputs.hidden_states):
            with open(output_path / f"layer_{i}" / f"{idx}.pkl", "wb") as f:
                save_embeddings(layer[0].cpu().numpy(), False, f)
        del outputs


def run_ohe(data_path: Path, output_path: Path, aa_level: bool = False, empty: bool = False, force: bool = False) -> None:
    """
    One-hot encode sequences and save them to the output path.

    Args:
        data_path: Path to the CSV file containing sequences.
        output_path: Path to save the one-hot encoded embeddings.
        aa_level: Whether to save amino acid level embeddings or mean pooled embeddings.
        empty: Not used, included for consistency.
        force: Whether to overwrite existing embeddings.
    """
    (out := (output_path / "layer_0")).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    sequences = data["sequence"].tolist()
    ids = data["ID"].tolist()

    for idx, seq in tqdm(zip(ids, sequences), total=len(sequences)):
        if not force and (out / f"{idx}.pkl").exists():
            continue
        one_hot = np.zeros((len(seq), 21), dtype=np.float32)
        for i, aa in enumerate(seq[:1022]):
            if aa == "-":
                continue
            one_hot[i, AA_OHE.get(aa, 20)] = 1
        with open(out / f"{idx}.pkl", "wb") as f:
            save_embeddings(one_hot, aa_level, f)


def run_esm_batched(model_name: str, num_layers: int, data_path: str, output_path: str, batch_size: int = 16) -> None:
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
    parser.add_argument("--aa-level", action="store_true")
    parser.add_argument("--empty", action="store_true")
    parser.add_argument("--force", action="store_true", help="Whether to overwrite existing embeddings.")
    args = parser.parse_args()

    print(sys.argv[1:])

    if "esmc" in args.model_name:
        run_esmc(args.model_name, args.data_path, args.output_path, args.aa_level, args.empty, args.force)
    elif "esm" in args.model_name:
        run_esm(PLM_MODELS[args.model_name], args.data_path, args.output_path, args.aa_level, args.empty, args.force)
    elif "ankh" in args.model_name:
        run_ankh(PLM_MODELS[args.model_name], args.data_path, args.output_path, args.aa_level, args.empty, args.force)
    elif "prott5" in args.model_name:
        run_prott5(args.data_path, args.output_path, args.aa_level, args.empty, args.force)
    elif "prostt5" in args.model_name:
        run_prostt5(args.data_path, args.output_path, args.aa_level, args.empty, args.force)
    elif "progen" in args.model_name:
        run_progen2(args.model_name, args.data_path, args.output_path, args.aa_level, args.empty, args.force)
    elif "protgpt2" in args.model_name:
        run_protgpt2(args.data_path, args.output_path, args.empty, args.force)
    elif "ohe" in args.model_name:
        run_ohe(args.data_path, args.output_path, args.aa_level, args.empty, args.force)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}. Supported models are: {list(PLM_MODELS.keys())} or 'ohe'.")
