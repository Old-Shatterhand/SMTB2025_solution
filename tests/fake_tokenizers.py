"""Synthetic tokenizers reproducing each real model's token layout.

These let the calibration contract be tested without downloading weights, and
pin the exact failure modes that ``return_special_tokens_mask`` gets wrong.

Two distinct mechanisms are modelled, because conflating them is precisely the
bug in the research code:

``auto_prefix`` / ``auto_suffix``
    Tokens the *tokenizer* inserts (ESM's ``<cls>``/``<eos>``, T5's ``</s>``).
``added_tokens``
    Multi-character vocab entries recognised inside the *text*, which the caller
    put there (ProstT5's ``<AA2fold>``). These are ordinary tokens, so they do
    not appear in ``all_special_ids``.
"""

from __future__ import annotations


class FakeTokenizer:
    def __init__(
        self,
        auto_prefix: tuple[str, ...] = (),
        auto_suffix: tuple[str, ...] = (),
        *,
        marker: str = "",
        special: tuple[str, ...] = (),
        added_tokens: tuple[str, ...] = (),
        vocab: str = "ACDEFGHIKLMNPQRSTVWYXBUZO",
        merge_runs: bool = False,
    ) -> None:
        self.auto_prefix, self.auto_suffix = auto_prefix, auto_suffix
        self.marker, self.merge_runs = marker, merge_runs
        self.added_tokens = added_tokens
        toks = [
            "<unk>",
            *auto_prefix,
            *auto_suffix,
            *added_tokens,
            *(marker + c for c in vocab),
        ]
        self._itos = list(dict.fromkeys(toks))
        self._stoi = {t: i for i, t in enumerate(self._itos)}
        self.unk_token_id = 0
        self.all_special_ids = [self._stoi[t] for t in special if t in self._stoi]

    def _pieces(self, text: str) -> list[str]:
        """Greedily match added tokens, then fall back to characters."""
        out, i = [], 0
        while i < len(text):
            if text[i] == " ":
                i += 1
                continue
            for at in self.added_tokens:
                if text.startswith(at, i):
                    out.append(at)  # added tokens carry no sub-token marker
                    i += len(at)
                    break
            else:
                if self.merge_runs:  # crude BPE stand-in: collapse identical runs
                    j = i
                    while j + 1 < len(text) and text[j + 1] == text[i] and j - i < 3:
                        j += 1
                    out.append(self.marker + text[i : j + 1])
                    i = j + 1
                else:
                    out.append(self.marker + text[i])
                    i += 1
        return out

    def __call__(self, text: str, add_special_tokens: bool = True, **kw):
        toks = self._pieces(text)
        if add_special_tokens:
            toks = [*self.auto_prefix, *toks, *self.auto_suffix]
        return {"input_ids": [self._stoi.get(t, self.unk_token_id) for t in toks]}

    def convert_ids_to_tokens(self, ids):
        return [self._itos[i] for i in ids]


def esm_like() -> FakeTokenizer:
    return FakeTokenizer(("<cls>",), ("<eos>",), special=("<cls>", "<eos>"))


def prott5_like() -> FakeTokenizer:
    return FakeTokenizer((), ("</s>",), marker="▁", special=("</s>",))


def prostt5_like() -> FakeTokenizer:
    # <AA2fold> is an ORDINARY added token -> absent from `special`.
    return FakeTokenizer(
        (), ("</s>",), marker="▁", special=("</s>",), added_tokens=("<AA2fold>",)
    )


def ankh_like() -> FakeTokenizer:
    # Unigram, no BOS at all.
    return FakeTokenizer((), ("</s>",), special=("</s>",))


def progen2_like() -> FakeTokenizer:
    # "1"/"2" direction flags are ordinary vocab entries; nothing is special.
    return FakeTokenizer(
        special=(), vocab="ACDEFGHIKLMNPQRSTVWYXBUZO12"
    )


def protgpt2_like() -> FakeTokenizer:
    return FakeTokenizer(merge_runs=True, special=())
