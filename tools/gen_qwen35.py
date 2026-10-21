#!/usr/bin/env python3
"""Build the qwen3.5 data files from bigpickle's Hugging Face tokenizer.json.

bigpickle is a Qwen3.5 model. Its tokenizer.json has Qwen3.5's vocab and merges
but Qwen2's split pattern: combining marks are not joined to letters, so Hindi,
Tamil, Thai and the like split into more pieces than with Qwen's own
tokenizer.json. This matches bigpickle; feed Qwen's file
(huggingface.co/Qwen/Qwen3.5-0.8B) to match Qwen instead. Run from the repo
root, with `tokenizers` installed:

    python tools/gen_qwen35.py path/to/tokenizer.json

Writes crates/toktok/data/{qwen35.vocab, qwen35.special, uniclass_qwen35.bin,
nfc_unsafe.bin}.

The vocab's ids follow its merge order, so merge priority is rank order. The
merges themselves ride along after the tokens (one left-part length per id),
because for some newer tokens they aren't the ones rank order implies.

Either pattern runs on the Qwen2 scanner: Qwen's joins \\p{M} to letters, which
is just L = [\\p{L}\\p{M}] in the class table. The classes are read off
`tokenizers` itself, one codepoint at a time: its regex engine has its own
Unicode version, and a newer one (Python's `regex`) calls some unassigned
codepoints letters.
"""

import json
import struct
import sys
import unicodedata

from tokenizers import Tokenizer

OUT = "crates/toktok/data/"


def byte_decoder():
    # GPT-2's byte <-> printable-char table, as ByteLevel uses it
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def classes(path):
    """L, N and S per codepoint, as Qwen3.5's own pre-tokenizer splits them."""
    pre = Tokenizer.from_file(path).pre_tokenizer

    def pieces(s):
        return len(pre.pre_tokenize_str(s))

    lc, nc, sc = set(), set(), set()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if pieces("a" + c) == 1:  # joins a letter run
            lc.add(cp)
        elif pieces(c + c) == 2:  # one piece per digit
            nc.add(cp)
        elif pieces("!" + c + "!") > 1:  # breaks a punctuation run
            sc.add(cp)
    return lc, nc, sc


def nfc_unsafe():
    """Codepoints NFC can change or compose onto what precedes them: a text
    without any is NFC already (see crates/toktok/src/nfc.rs)."""
    second = {0x1161 + i for i in range(21)} | {0x11A8 + i for i in range(27)}  # Hangul V, T
    for cp in range(0x110000):
        d = unicodedata.decomposition(chr(cp)).split()
        if len(d) == 2 and not d[0].startswith("<"):
            second.add(int(d[1], 16))
    out = set()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if unicodedata.combining(c) or unicodedata.normalize("NFC", c) != c or cp in second:
            out.add(cp)
    assert min(out) >= 0x300, "the Rust check skips bytes below 0xCC"
    return out


def ranges(cps):
    out, lo = [], None
    for cp in range(0x110000):
        hit = cp in cps
        if hit and lo is None:
            lo = cp
        if not hit and lo is not None:
            out.append((lo, cp - 1))
            lo = None
    if lo is not None:
        out.append((lo, 0x10FFFF))
    return struct.pack("<I", len(out)) + b"".join(struct.pack("<II", a, b) for a, b in out)


def main(path):
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    m = t["model"]
    assert m["type"] == "BPE" and t["normalizer"] == {"type": "NFC"}
    dec = byte_decoder()
    vocab = {bytes(dec[c] for c in tok): i for tok, i in m["vocab"].items()}
    assert sorted(vocab.values()) == list(range(len(vocab)))
    merges = [x.split(" ") if isinstance(x, str) else x for x in m["merges"]]
    ids = [m["vocab"][a + b] for a, b in merges]
    assert ids == sorted(ids), "merge order must follow ids for rank merging"

    with open(OUT + "qwen35.vocab", "wb") as f:
        f.write(struct.pack("<I", len(vocab)))
        for b, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write(struct.pack("<H", len(b)) + b + struct.pack("<I", i))
        # rank order alone picks other merges for some newer tokens, and 201
        # tokens have none: ship each token's left-part length (0 = no merge)
        left = bytearray(len(vocab))
        for a, _ in merges:
            left[m["vocab"][a + _]] = len(bytes(dec[c] for c in a))
        f.write(b"SPL1" + bytes(left))

    added = sorted((a["id"], a["content"].encode()) for a in t["added_tokens"])
    with open(OUT + "qwen35.special", "wb") as f:
        f.write(struct.pack("<I", len(added)))
        for i, s in added:
            f.write(struct.pack("<IH", i, len(s)) + s)

    with open(OUT + "uniclass_qwen35.bin", "wb") as f:
        f.write(b"".join(ranges(c) for c in classes(path)))
    with open(OUT + "nfc_unsafe.bin", "wb") as f:
        f.write(ranges(nfc_unsafe()))
    print(f"{len(vocab)} tokens, {len(added)} added tokens")


if __name__ == "__main__":
    main(sys.argv[1])
