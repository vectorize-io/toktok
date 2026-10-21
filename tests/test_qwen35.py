"""qwen3.5: ids against Hugging Face `tokenizers` (tests/vectors_qwen3.5.bin,
from tools/gen_vectors.py), and counts a bigpickle server reported."""

import os
import struct
import unicodedata

import toktok

HERE = os.path.dirname(__file__)

# usage.input_tokens from a bigpickle server, minus the 38 an empty state with
# one short question costs
SERVER = [
    ("hello world", 2),
    ("Hello, World!", 4),
    ("trailing\n\n", 3),
    ("日本語のテキストです。", 5),
    ("Привет, как дела?", 7),
    ("emoji 🚀🔥👍🏽 test", 14),
    ("nai\u0308ve cafe\u0301", 4),  # decomposed: counted as its NFC form
    ("हिन्दी पाठ", 8),
    ("ในภาษาไทยมีสระ", 5),
    ("<|endoftext|> and <think>", 4),  # special strings are one token each
]


def vectors():
    with open(os.path.join(HERE, "vectors_qwen3.5.bin"), "rb") as f:
        blob = f.read()
    (n,), p = struct.unpack_from("<I", blob), 4
    for _ in range(n):
        (tl,) = struct.unpack_from("<I", blob, p)
        text = blob[p + 4 : p + 4 + tl].decode()
        p += 4 + tl
        (ni,) = struct.unpack_from("<I", blob, p)
        yield text, list(struct.unpack_from(f"<{ni}I", blob, p + 4))
        p += 4 + 4 * ni


def test_matches_hf_tokenizers():
    enc = toktok._encoding("qwen3.5")
    cases = list(vectors())
    assert len(cases) > 2000
    for text, ids in cases:
        assert enc.encode_ordinary(text) == ids, repr(text)
    texts = [t for t, _ in cases]
    assert toktok.batch_count(texts, "qwen3.5") == [len(i) for _, i in cases]
    assert toktok.batch_count(texts, "qwen3.5", threads=1) == [len(i) for _, i in cases]


def test_server_counts():
    texts = [t for t, _ in SERVER]
    assert toktok.batch_count(texts, "qwen3.5", with_special=True) == [n for _, n in SERVER]
    # without with_special a special string counts as its text: only ever more
    plain = toktok.batch_count(texts, "qwen3.5")
    assert all(p >= n for p, (_, n) in zip(plain, SERVER, strict=True))


def test_nfc():
    composed, decomposed = "naïve café", "nai\u0308ve cafe\u0301"
    assert toktok.batch_count([decomposed], "qwen3.5") == toktok.batch_count([composed], "qwen3.5")
    assert toktok.truncate(decomposed, 100, "qwen3.5") == (composed, 4)
    assert toktok.truncate(composed, 100, "qwen3.5")[0] is composed  # NFC already: no copy
    # the core's check flags every text NFC would change, and few others
    texts = [t for t, _ in vectors()] + [decomposed, "한국어", "e\u0301", "\u1100\u1161", "Å"]
    flagged = set(toktok._encoding("qwen3.5").unnormalized(texts))
    assert {i for i, t in enumerate(texts) if not unicodedata.is_normalized("NFC", t)} <= flagged
    kept = {texts[i] for i in flagged}
    assert composed not in kept and "한국어" not in kept


def test_truncate():
    docs = ["hello world", "हिन्दी पाठ " * 20, "", "日本語のテキストです。" * 10]
    for d in docs:
        total = toktok.batch_count([d], "qwen3.5")[0]
        for n in (0, 1, 3, total, total + 1):
            got, t = toktok.truncate(d, n, "qwen3.5")
            assert t == total and d.startswith(got)
            assert toktok.batch_count([got], "qwen3.5")[0] <= max(n, 0) or not got
    want = [toktok.truncate(d, 5, "qwen3.5") for d in docs]
    assert toktok.batch_truncate(docs, 5, "qwen3.5") == want
    assert toktok.batch_truncate(docs, 5, "qwen3.5", threads=1) == want
