# plmlayer

**The last layer of a protein language model is almost never the best one. This
tool finds the layer that is, and hands you back a truncated model.**

Implements the practical consequence of [*Task- and dataset-specific information
in protein language models*](https://arxiv.org/abs/2608.12090) (Joeres,
Senatorov, Kolchina, Klakow & Kalinina), which probed 13 PLMs across 15
downstream tasks and found the deepest layer won in only **17.9%** of cases.

```bash
pip install -e .

plmlayer suggest \
    --model facebook/esm2_t33_650M_UR50D \
    --data my_dataset.csv \
    --task regression \
    --out ./my-esm-truncated
```

```
best layer      22 of 33 (67% depth)   pearson = 0.7241
last layer      33                     pearson = 0.6615
gain over last  +9.5%
seed agreement  100% (layers chosen across seeds: [22])
```

`./my-esm-truncated` is a normal HuggingFace model directory holding the first
22 blocks. It loads anywhere the original did, runs ~1.5x faster, and scores
better on your task:

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("./my-esm-truncated")
tok = AutoTokenizer.from_pretrained("./my-esm-truncated")
```

## Why this is cheap

The paper's Figure 3E-H shows **15-20% of your training data is enough** to
identify a layer within 5% of the best achievable performance, and that the same
layer is picked across seeds. So `plmlayer` subsamples first, embeds once
(a single forward pass yields every layer), and probes each layer with a kNN
probe. Minutes on a GPU, not a cluster job.

## Input format

A CSV with `ID`, `sequence`, and a label column:

| column | required | notes |
|---|---|---|
| `ID` | yes | unique identifier |
| `sequence` | yes | amino acid sequence |
| `label` / `labels` | yes | the target; DeepLoc2.0's 10 class columns are also recognised |
| `split` | no | `train` / `valid` (or `val`); generated if absent |
| `positions` | residue-level only | 0-based residue indices to probe |

`--task` is inferred from the label column and can be set explicitly with
`regression`, `binary`, `multi-class` or `multi-label`.

## Models

Any HuggingFace protein language model works — pass its id to `--model`. The
models benchmarked in the paper have tested presets:

```bash
plmlayer models                # list presets
plmlayer models --model ankh_base   # introspect any checkpoint
```

`esm2_8m/35m/150m/650m/3b`, `prott5`, `prostt5`, `ankh_base`, `ankh_large`,
`progen2_small/medium/large`, `protgpt2`. The repo's own names (`esm_t30`, …)
also resolve. ProtGPT2 uses a BPE tokenizer, so residues do not map 1:1 onto
tokens and residue-level probing is rejected rather than silently mis-pooled.

## How it differs from the paper's pipeline

`plmlayer` is a reimplementation, not a wrapper around `src/`. Three deliberate
differences:

1. **The final norm is applied to every layer.** Every architecture here applies
   a trailing norm (`emb_layer_norm_after`, `final_layer_norm`, `ln_f`) to the
   last block's output, so `hidden_states[i]` is raw for `i < L` but normed at
   `i = L`. A model truncated to layer `k` emits `final_norm(h_k)` — on ESM-2
   that differs from raw `h_k` by ~32 in absolute magnitude. Probing the raw
   state would mean shipping a model the probe never scored, so `plmlayer`
   probes exactly what it ships, and verifies it numerically before saving.
   A consequence: curves here will not reproduce the paper's exactly, which
   compare a normed last layer against un-normed intermediates.
2. **Token alignment is calibrated, not hardcoded.** Residue positions are
   established by tokenizing probe sequences and proving a 1:1 alignment, rather
   than per-family slices. `special_tokens_mask` cannot do this: ProstT5's
   `<AA2fold>` and ProGen2's `1`/`2` flags are ordinary tokens it does not see.
3. **Metrics are computed in the pipeline.** No prediction pickles, no
   recomputation at figure-drawing time.

## Development

```bash
pytest tests -q                      # full suite
pytest tests -q -m "not weights"     # no model downloads needed
```

## Citation

```bibtex
@article{joeres2026plmlayers,
  title  = {Task- and dataset-specific information in protein language models},
  author = {Joeres, Roman and Senatorov, Ilya and Kolchina, Anastasia and
            Klakow, Dietrich and Kalinina, Olga V.},
  journal = {arXiv preprint arXiv:2608.12090},
  year   = {2026}
}
```
