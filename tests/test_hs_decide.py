"""hs-decide-1-flash: ids against Hugging Face `tokenizers` with the model's own
tokenizer.json (tests/vectors_hs-decide-1-flash.bin, from tools/gen_vectors.py)."""

import toktok
from test_qwen35 import vectors

ENC = "hs-decide-1-flash"


def test_matches_hf_tokenizers():
    cases = list(vectors(ENC))
    assert len(cases) > 2000
    enc = toktok._encoding(ENC)
    for text, ids in cases:
        assert enc.encode_ordinary(text) == ids, repr(text)
    assert toktok.batch_count([t for t, _ in cases], ENC) == [len(i) for _, i in cases]
    # the fixture covers where its split differs from qwen3.5's
    q = toktok._encoding("qwen3.5")
    assert any(q.encode_ordinary(t) != ids for t, ids in cases)


def test_marks_join_letters():
    hi = "हिन्दी पाठ"
    assert toktok.batch_count([hi], ENC) < toktok.batch_count([hi], "qwen3.5")


def test_nfc_and_specials():
    assert toktok.batch_count(["naïve"], ENC) == toktok.batch_count(["naïve"], ENC)
    assert toktok.batch_count(["<think>"], ENC, with_special=True) == [1]
    assert toktok.batch_count(["<|audio_start|>"], ENC, with_special=True) != [1]
