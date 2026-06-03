import pandas as pd
import numpy as np


def sample_dataset(
    sequences=pd.Series(dtype=str),
    mask_token="<mask>",
    mask_prob=0.15,
    n_samples=1000,
    train_size=0.8,
    val_size=0.1,
    random_state=42,
) -> pd.DataFrame:
    np.random.seed(random_state)
    assert 0 <= mask_prob <= 1, "mask_prob must be between 0 and 1"
    assert 0 <= n_samples <= len(sequences), "n_samples must be less than or equal to the number of sequences"
    data = []
    while len(data) < n_samples:
        seq = sequences.sample(n=1).iloc[0]
        if len(seq) > 1022:
            continue
        result = ""
        old_labels = ""
        positions = []
        for i, token in enumerate(seq):
            if np.random.rand() < mask_prob:
                result += mask_token
                old_labels += token
                positions.append(i)
            else:
                result += token
        if len(old_labels) > 0:
            data.append(
                {
                    "sequence": result,
                    "labels": old_labels,
                    "positions": positions,
                    "split": np.random.choice(["train", "val", "test"], p=[train_size, val_size, 1 - train_size - val_size]),
                }
            )

    df = pd.DataFrame(data)
    df["ID"] = [f"P{i:05d}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Sample a dataset for masked language modeling.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
    parser.add_argument("--sequence_column", type=str, default="sequence", help="Column containing sequences.")
    parser.add_argument("--mask_token", type=str, default="<mask>", help="Token to use for masking.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking a token.")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of training data.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of validation data.")
    args = parser.parse_args()
    dataset = pd.read_csv(args.input_file)
    sample_df = sample_dataset(
        dataset[args.sequence_column],
        mask_token=args.mask_token,
        n_samples=args.n_samples,
        mask_prob=args.mask_prob,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.output_file, index=False)
