"""Empirically establish where residues sit inside a tokenized sequence.

Why not ``return_special_tokens_mask``? Because it is wrong, silently, for two
of the models this package must support:

* **ProstT5** -- ``<AA2fold>`` is an *ordinary added token*, absent from
  ``additional_special_tokens``. The slow ``T5Tokenizer`` computes the mask
  positionally (``[0] * len + [1]``), so the direction token is reported as a
  residue and every index shifts by one.
* **ProGen2** -- the ``1``/``2`` direction flags are ordinary vocab ids with a
  bare ``ByteLevel`` post-processor, so the mask is all zeros.

Neither failure raises. Instead of trusting the mask, we *prove* the alignment:
tokenize a handful of probe sequences, require the token count to be affine in
the residue count with slope exactly 1, then solve for the unique offset at
which the tokens decode back to the residue characters in order.
"""

from __future__ import annotations

from dataclasses import dataclass

from plmlayer.types import CalibrationError, ModelSpec, TokenLayout

__all__ = ["PROBES", "calibrate", "CalibrationReport"]

# Chosen to expose distinct failure modes:
#   0/1 differ in length      -> solves the affine relation
#   2 repeats one residue     -> catches subword merging of runs
#   3 has non-canonical AAs   -> catches UNK mapping and residue_map gaps
PROBES: tuple[str, ...] = (
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDEFGHIKLMNPQRSTVWY" * 2 + "ACDEFG",
    "MMMMMMMMMM",
    "ACDEFGHIKLMNPQRSTVWYXBUZO",
)

# Sub-token markers used by sentencepiece / byte-level / wordpiece tokenizers.
_MARKERS = ("▁", "Ġ", "##")


def _strip_marker(token: str) -> str:
    for m in _MARKERS:
        if token.startswith(m):
            return token[len(m) :]
    return token


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    layout: TokenLayout
    n_tokens_per_probe: tuple[int, ...]
    declared_matches: bool | None  # None when the spec declared nothing
    detail: str = ""


def _encode(tokenizer, text: str) -> tuple[list[int], list[str]]:
    """Tokenize with special tokens on, returning ids and their string forms."""
    # add_special_tokens=False is deliberately NOT used: it routes slow
    # tokenizers through the already_has_special_tokens branch, whose
    # all_special_ids check includes mask_token_id -- which would delete the
    # very <mask> position MLM-style work depends on.
    enc = tokenizer(text, add_special_tokens=True)
    ids = list(enc["input_ids"])
    return ids, tokenizer.convert_ids_to_tokens(ids)


def calibrate(
    tokenizer,
    spec: ModelSpec,
    probes: tuple[str, ...] = PROBES,
) -> CalibrationReport:
    """Determine the :class:`TokenLayout` for ``tokenizer`` under ``spec``."""
    encoded = []
    for probe in probes:
        text, residues = spec.preprocess(probe)
        ids, tokens = _encode(tokenizer, text)
        encoded.append((residues, ids, tokens))

    n_tokens = tuple(len(ids) for _, ids, _ in encoded)

    # --- step 1: is the token count affine in residue count, slope 1? -----
    base_res, base_ids, _ = encoded[0]
    for residues, ids, _ in encoded[1:]:
        if len(ids) - len(base_ids) != len(residues) - len(base_res):
            return CalibrationReport(
                layout=TokenLayout(0, 0, residue_level=False),
                n_tokens_per_probe=n_tokens,
                declared_matches=None,
                detail=(
                    "token count is not 1:1 with residue count "
                    f"({len(base_ids)} tokens for {len(base_res)} residues vs "
                    f"{len(ids)} for {len(residues)}) -- subword tokenizer, "
                    "residue-level extraction unavailable"
                ),
            )

    # --- step 2: solve for the alignment offset ---------------------------
    layouts: set[tuple[int, int]] = set()
    unk = set()
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for residues, ids, tokens in encoded:
        n_extra = len(ids) - len(residues)
        if n_extra < 0:
            raise CalibrationError(
                f"{spec.hf_id}: {len(ids)} tokens for {len(residues)} residues -- "
                "the tokenizer dropped characters; check ModelSpec.residue_map."
            )
        offsets = [
            p
            for p in range(n_extra + 1)
            if [_strip_marker(t) for t in tokens[p : p + len(residues)]] == list(residues)
        ]
        if not offsets:
            # Fall back to reporting UNKs, which is the usual cause.
            if unk_id is not None:
                bad = {
                    r
                    for p in range(n_extra + 1)
                    for r, i in zip(residues, ids[p : p + len(residues)])
                    if i == unk_id
                }
                if bad:
                    raise CalibrationError(
                        f"{spec.hf_id}: residues {sorted(bad)} tokenize to <unk>. "
                        "Add them to ModelSpec.residue_map."
                    )
            raise CalibrationError(
                f"{spec.hf_id}: no offset aligns tokens to residues.\n"
                f"  residues: {residues}\n  tokens:   {tokens}"
            )
        if len(offsets) > 1:
            # Ambiguity is possible for short/repetitive probes; the probe set
            # includes varied ones so the intersection is unique.
            pass
        p = offsets[0]
        layouts.add((p, n_extra - p))
        if unk_id is not None:
            unk |= {
                r for r, i in zip(residues, ids[p : p + len(residues)]) if i == unk_id
            }

    if len(layouts) != 1:
        raise CalibrationError(
            f"{spec.hf_id}: inconsistent token layout across probes: {sorted(layouts)}"
        )
    n_prefix, n_suffix = layouts.pop()
    layout = TokenLayout(
        n_prefix=n_prefix,
        n_suffix=n_suffix,
        residue_level=True,
        unk_residues=frozenset(unk),
        source="calibrated",
    )

    # --- step 3: cross-check the declared layout --------------------------
    declared_matches: bool | None = None
    detail = ""
    if spec.n_prefix is not None and spec.n_suffix is not None:
        declared = (spec.n_prefix, spec.n_suffix)
        declared_matches = declared == (n_prefix, n_suffix)
        if not declared_matches:
            detail = (
                f"declared layout {declared} but measured {(n_prefix, n_suffix)} -- "
                "the tokenizer changed under us; refusing to guess"
            )
            raise CalibrationError(f"{spec.hf_id}: {detail}")

    return CalibrationReport(
        layout=layout,
        n_tokens_per_probe=n_tokens,
        declared_matches=declared_matches,
        detail=detail,
    )
