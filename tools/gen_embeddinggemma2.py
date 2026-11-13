#!/usr/bin/env python3
"""Build the embeddinggemma-2 data files from its Hugging Face tokenizer.json.

    python tools/gen_embeddinggemma2.py path/to/tokenizer.json

(huggingface.co/google/embeddinggemma-2). Writes
crates/toktok/data/{eg2.vocab, eg2.special}.

Gemma's tokenizer is a SentencePiece-style BPE, not a byte-level one:

- spaces become U+2581 and there is no split pattern, so the whole text is one
  word. Tokens are stored here with U+2581 turned back into a space, so text
  goes in as it is (Rust maps a literal U+2581 in the input to a space first).
- the starting symbols are characters; one the vocab lacks becomes its UTF-8
  bytes (<0x..> tokens), which never merge.
- merge priority is the merge list's order, and 160k tokens can be built by
  more than one merge, so rank order can't be read off token ids.

So the vocab ships, after the tokens:
  SPL1  one byte per id: the left part's byte length of the merge that builds
        that token from its own text (0 = BPE never yields it, 255 = a starting
        symbol: a character, or a byte no character token covers)
  MRG1  u32 count, then one u32 per merge in list order: the merged token's id
        | its left part's byte length << 24. The loader works out the order the
        merges fire in (crates/toktok/src/vocab.rs, `fire_order`).
"""

import json
import struct
import sys

OUT = "crates/toktok/data/"
W = "▁"


def load(path):
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    m = t["model"]
    assert m["type"] == "BPE" and m["byte_fallback"] and not m["ignore_merges"]
    assert t["normalizer"] == {"type": "Replace", "pattern": {"String": " "}, "content": W}
    # splitting on " " after spaces became U+2581 never splits anything
    assert t["pre_tokenizer"]["type"] == "Split" and t["pre_tokenizer"]["pattern"] == {
        "String": " "
    }
    vocab = m["vocab"]
    assert sorted(vocab.values()) == list(range(len(vocab)))
    merges = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m["merges"]]
    return t, vocab, merges


class Bpe:
    """Hugging Face's BPE with byte fallback, symbol by symbol (slow, exact)."""

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.rank = {}
        for r, (a, b) in enumerate(merges):
            self.rank.setdefault((vocab[a], vocab[b]), (r, vocab[a + b]))
        self.byte = [vocab[f"<0x{b:02X}>"] for b in range(256)]

    def start(self, s):
        out = []
        for c in s:
            i = self.vocab.get(c)
            out.extend([i] if i is not None else [self.byte[b] for b in c.encode()])
        return out

    def run(self, s):
        """Final ids, and the last merge (left, right) applied, if any."""
        sym = self.start(s)
        last = None
        while len(sym) > 1:
            best = None
            for k in range(len(sym) - 1):
                r = self.rank.get((sym[k], sym[k + 1]))
                if r is not None and (best is None or r[0] < best[0][0]):
                    best = (r, k)
            if best is None:
                break
            (_, new), k = best
            last = (sym[k], sym[k + 1])
            sym[k : k + 2] = [new]
        return sym, last


def to_bytes(tok):
    if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
        return bytes([int(tok[3:5], 16)])
    return tok.replace(W, " ").encode()


def build(path):
    t, vocab, merges = load(path)
    bpe = Bpe(vocab, merges)
    n = len(vocab)
    tok = [None] * n
    for s, i in vocab.items():
        tok[i] = s
    special = {a["id"] for a in t["added_tokens"]}
    byte_ids = set(bpe.byte)

    # SPL1: how each token builds from its own text
    left = bytearray(n)
    split = [None] * n
    for i, s in enumerate(tok):
        if i in special:
            continue
        if i in byte_ids:
            b = to_bytes(s)[0]
            # a byte a character token covers never comes out as a byte
            if b >= 0x80 or chr(b) not in vocab:
                left[i] = 255
            continue
        if len(s) == 1:
            left[i] = 255
            continue
        ids, last = bpe.run(s)
        if ids == [i]:
            a, b = last
            left[i] = len(to_bytes(tok[a]))
            split[i] = last
            assert to_bytes(tok[a]) + to_bytes(tok[b]) == to_bytes(s)
    for i in range(n):
        assert left[i] < 255 or len(to_bytes(tok[i])) <= 4

    rows, seen = [], set()
    for a, b in merges:
        if (a, b) in seen:
            continue  # a repeated merge never fires at its later rank
        seen.add((a, b))
        la = len(to_bytes(a))
        assert vocab[a + b] < 1 << 24 and 0 < la < 255
        rows.append(vocab[a + b] | la << 24)
    return t, tok, left, rows


def write(t, tok, left, order):
    with open(OUT + "eg2.vocab", "wb") as f:
        f.write(struct.pack("<I", len(tok)))
        for i, s in enumerate(tok):
            b = to_bytes(s)
            assert 0 < len(b) <= 255
            f.write(struct.pack("<H", len(b)) + b + struct.pack("<I", i))
        f.write(b"SPL1" + bytes(left))
        f.write(b"MRG1" + struct.pack("<I", len(order)))
        f.write(struct.pack(f"<{len(order)}I", *order))
    added = sorted((a["id"], a["content"].encode()) for a in t["added_tokens"])
    with open(OUT + "eg2.special", "wb") as f:
        f.write(struct.pack("<I", len(added)))
        for i, s in added:
            f.write(struct.pack("<IH", i, len(s)) + s)
    reach = sum(1 for x in left if x)
    print(f"{len(tok)} tokens ({reach} reachable), {len(order)} merges, {len(added)} added tokens")


def selfcheck(path):
    """The slow BPE above agrees with `tokenizers` (it is what the tables model)."""
    import random

    from tokenizers import Tokenizer

    _, vocab, merges = load(path)
    bpe, hf = Bpe(vocab, merges), Tokenizer.from_file(path)
    rnd = random.Random(1)
    pool = [
        chr(c)
        for c in list(range(32, 127)) * 3 + list(range(0xA0, 0x500)) + list(range(0x4E00, 0x4F00))
    ]
    pool += ["\U0001f600", "▁", "\n", "\t", "\U00020000"]
    for _ in range(3000):
        s = "".join(rnd.choice(pool) for _ in range(rnd.randint(1, 30)))
        want = hf.encode(s, add_special_tokens=False).ids
        got = bpe.run(s.replace(" ", W))[0]
        assert got == want, (s, got, want)
    print("slow BPE matches tokenizers on 3000 strings")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        selfcheck(sys.argv[2])
    else:
        write(*build(sys.argv[1]))
