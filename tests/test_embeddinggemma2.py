"""embeddinggemma-2: ids against Hugging Face `tokenizers` (tests/vectors_
embeddinggemma-2.bin, from tools/gen_vectors.py), and what Hindsight's
embedding server feeds the model."""

import os
import struct

import toktok

HERE = os.path.dirname(__file__)
ENC = "embeddinggemma-2"

# ids read off the embedding server's own tokenizer (GemmaTokenizer,
# transformers 5.22, tokenizers 0.23)
SERVER_IDS = [
    ("hello world", [23391, 1902]),
    ("  two  spaces", [138, 13498, 138, 35220]),
    ("12345 numbers", [236770, 236778, 236800, 236812, 236810, 4945]),
]


def vectors():
    with open(os.path.join(HERE, "vectors_embeddinggemma-2.bin"), "rb") as f:
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
    enc = toktok._encoding(ENC)
    cases = list(vectors())
    assert len(cases) > 2000
    for text, ids in cases:
        assert enc.encode_ordinary(text) == ids, repr(text)
    texts = [t for t, _ in cases]
    assert toktok.batch_count(texts, ENC) == [len(i) for _, i in cases]
    assert toktok.batch_count(texts, ENC, threads=1) == [len(i) for _, i in cases]


def test_server_ids():
    enc = toktok._encoding(ENC)
    for text, ids in SERVER_IDS:
        assert enc.encode_ordinary(text) == ids


def test_server_document_prompt():
    # the server sends plain text as <bos> "title: none | text: " + text <eos>;
    # no token crosses the space after "text:", so that is 8 + count(" " + text)
    for text in ["hello world", "  indented", "東京", "", "x" * 50]:
        full = "<bos>title: none | text: " + text + "<eos>"
        want = toktok.batch_count([full], ENC, with_special=True)[0]
        assert want == 8 + toktok.batch_count([" " + text], ENC)[0]


def test_specials():
    # tokenizers always reads special strings as specials; with_special matches it
    assert toktok.batch_count(["a<bos>b <|turn>x"], ENC, with_special=True) == [6]
    assert toktok.batch_count(["a<bos>b <|turn>x"], ENC)[0] > 6


def test_lower_block_is_a_space():
    text = "a▁b ▁▁c"
    assert toktok.batch_count([text], ENC) == toktok.batch_count(["a b   c"], ENC)
    got, total = toktok.truncate(text, 2, ENC)
    assert text.startswith(got) and got == "a▁b" and total == 4


def test_long_whitespace_runs():
    enc = toktok._encoding(ENC)
    assert enc.encode_ordinary(" " * 32) == [167, 236743]
    assert toktok.batch_count([" " * 1000, "\n" * 1000, "\t" * 1000], ENC) == [33, 33, 33]


def test_truncate():
    docs = ["hello world", "def f(x):\n    return x\n" * 20, "", "日本語のテキストです。" * 10]
    for d in docs:
        total = toktok.batch_count([d], ENC)[0]
        for n in (0, 1, 3, total, total + 1):
            got, t = toktok.truncate(d, n, ENC)
            assert t == total and d.startswith(got)
            assert toktok.batch_count([got], ENC)[0] <= max(n, 0) or not got
    want = [toktok.truncate(d, 5, ENC) for d in docs]
    assert toktok.batch_truncate(docs, 5, ENC) == want
    assert toktok.batch_truncate(docs, 5, ENC, threads=1) == want
