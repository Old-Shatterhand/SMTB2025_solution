"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_suggest(sub) -> None:
    p = sub.add_parser(
        "suggest",
        help="find the best layer for a dataset and optionally save it truncated",
    )
    p.add_argument("--model", required=True,
                   help="preset key (esm2_650m), repo alias (esm_t30), HF id, or local path")
    p.add_argument("--data", required=True, type=Path,
                   help="CSV with ID, sequence and a label column")
    p.add_argument("--out", type=Path, default=None,
                   help="directory to write the truncated model into")
    p.add_argument("--task", default=None,
                   choices=["regression", "binary", "multi-class", "multi-label"],
                   help="default: inferred from the label column")
    p.add_argument("--label-col", default=None, help="override label column detection")
    p.add_argument("--probe", default="knn", choices=["knn", "lr"])
    p.add_argument("-k", type=int, default=10, help="neighbours for the kNN probe")
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-val", type=int, default=1000)
    p.add_argument("--schedule", default="full", choices=["full", "coarse"])
    p.add_argument("--residue-level", action="store_true",
                   help="probe individual residues; needs a 'positions' column")
    p.add_argument("--seeds", type=int, default=3, help="resampling stability checks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    p.add_argument("--json", type=Path, default=None, help="also write the result as JSON")
    p.add_argument("--quiet", action="store_true")


def _add_models(sub) -> None:
    p = sub.add_parser("models", help="list presets, or introspect one checkpoint")
    p.add_argument("--model", default=None, help="load this checkpoint and report its layers")
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=None)


def _curve_plot(result) -> str:
    scores = [v for v in result.curve.values() if v == v]
    if not scores:
        return ""
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    lines = ["", f"layer performance ({result.metric}):"]
    for layer, score in sorted(result.curve.items()):
        if score != score:
            lines.append(f"  {layer:>3}      nan")
            continue
        bar = "#" * max(1, int((score - lo) / span * 46))
        tag = "  <- best" if layer == result.best_layer else ("  (last)" if layer == result.last_layer else "")
        lines.append(f"  {layer:>3} {score:+.4f} {bar}{tag}")
    return "\n".join(lines)


def _run_suggest(args) -> int:
    from plmlayer.api import suggest_layer

    result = suggest_layer(
        data=args.data,
        model=args.model,
        task=args.task,
        label_col=args.label_col,
        probe=args.probe,
        k=args.k,
        max_train=args.max_train,
        max_val=args.max_val,
        schedule=args.schedule,
        residue_level=args.residue_level,
        n_seeds=args.seeds,
        seed=args.seed,
        device=args.device,
        cache_dir=args.cache_dir,
        out=args.out,
        progress=not args.quiet,
    )
    if not args.quiet:
        print(result.summary())
        print(_curve_plot(result))
        if args.out:
            print(f"\ntruncated model written to {args.out}")
            print(f"  load it with: AutoModel.from_pretrained({str(args.out)!r})")
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


def _run_models(args) -> int:
    from plmlayer.registry import PRESETS, layer_hint, list_specs

    if args.model is None:
        print(f"{'key':16s} {'layers':>7s}  checkpoint")
        for key in list_specs():
            hint = layer_hint(key)
            print(f"{key:16s} {str(hint or '?'):>7s}  {PRESETS[key].hf_id}")
        print("\nAny HuggingFace protein language model also works; pass its id to --model.")
        return 0

    from plmlayer.api import load_adapter

    adapter = load_adapter(args.model, device=args.device, cache_dir=args.cache_dir)
    lay = adapter.layout
    print(f"checkpoint         {adapter.spec.hf_id}")
    print(f"layers             {adapter.n_layers} ({adapter.n_states} extractable states)")
    print(f"hidden size        {adapter.hidden_size}")
    print(f"token layout       {lay.n_prefix} prefix / {lay.n_suffix} suffix (calibrated)")
    print(f"residue-level      {'yes' if adapter.supports_residue_level else 'no (subword tokenizer)'}")
    print(f"final state normed {adapter.final_state_is_normed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plmlayer",
        description=(
            "Pick the most informative layer of a protein language model for your "
            "dataset, and get back a truncated model."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_suggest(sub)
    _add_models(sub)

    args = parser.parse_args(argv)
    try:
        return _run_suggest(args) if args.command == "suggest" else _run_models(args)
    except (ValueError, RuntimeError, NotImplementedError, MemoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
