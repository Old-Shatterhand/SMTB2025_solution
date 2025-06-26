from pathlib import Path
from argparse import ArgumentParser
import pickle

from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_esm(model_name: str, input_file: str, output_path: str):
    (out := Path(output_path)).mkdir(parents=True, exist_ok=True)
    for i in range(6):
        (out / f"layer_{i}").mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_file)

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name).to(DEVICE)

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
                pickle.dump(f, layer[:, 1: -1].cpu().numpy())
    print(f"Embedded all sequences with {model_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/esm_embeddings")
    args = parser.parse_args()

    run_esm(args.model_name, args.input_file, args.output_path)
