"""The Python surface: decode, offsets, batch, numpy helpers, model lookup."""

import pytest

import toktok

TEXT = "Hello, 日本語 world! 123\n\n  indented\ttext 🚀"


@pytest.fixture(scope="module")
def enc():
    return toktok.get_encoding("cl100k_base")


def test_roundtrip(enc):
    assert enc.decode(enc.encode_ordinary(TEXT)) == TEXT
    assert enc.decode_bytes(enc.encode_ordinary(TEXT)) == TEXT.encode()


def test_count_matches_encode(enc):
    assert enc.count(TEXT) == len(enc.encode_ordinary(TEXT))


def test_encoding_is_cached(enc):
    assert toktok.get_encoding("cl100k_base") is enc
    assert enc.name == "cl100k_base"
    assert repr(enc) == "<toktok.Tokenizer 'cl100k_base'>"


def test_single_token_helpers(enc):
    tid = enc.encode_single_token(" hello")
    assert enc.decode_single_token_bytes(tid) == b" hello"
    assert enc.encode_single_token(b" hello") == tid
    assert enc.decode_single_token_bytes(enc.eot_token) == b"<|endoftext|>"
    assert enc.is_special_token(enc.eot_token)
    assert not enc.is_special_token(tid)
    with pytest.raises(KeyError):
        enc.encode_single_token("this is definitely not one token")


def test_token_byte_values(enc):
    vals = enc.token_byte_values()
    assert len(vals) == 100256
    assert vals[enc.encode_single_token(" hello")] == b" hello"


def test_offsets_byte_and_char(enc):
    ids, spans = enc.encode_with_offsets(TEXT, "byte")
    raw = TEXT.encode()
    assert spans[0][0] == 0 and spans[-1][1] == len(raw)
    for tid, (a, b) in zip(ids, spans):
        assert enc.decode_single_token_bytes(tid) == raw[a:b]

    ids2, cspans = enc.encode_with_offsets(TEXT, "char")
    assert ids2 == ids
    assert cspans[-1][1] == len(TEXT)
    # char spans slice the str into the same pieces the byte spans do
    assert "".join(TEXT[a:b] for a, b in cspans) == TEXT

    with pytest.raises(ValueError):
        enc.encode_with_offsets(TEXT, "nope")


def test_encode_with_special(enc):
    ids = enc.encode_with_special("a<|endoftext|>b")
    assert enc.eot_token in ids
    assert enc.decode(ids) == "a<|endoftext|>b"


def test_batch_matches_sequential(enc):
    docs = [f"doc {i}: hello 日本語 {i}" for i in range(200)]
    flat, offsets = toktok.encode_batch_to_numpy(enc, docs, threads=4)
    assert len(offsets) == len(docs) + 1
    for i, d in enumerate(docs):
        assert list(flat[offsets[i] : offsets[i + 1]]) == enc.encode_ordinary(d)


def test_count_batch(enc):
    docs = ["one two three", "", "日本語のテキスト", "x" * 100]
    counts = toktok.count_batch(enc, docs)
    assert list(counts) == [enc.count(d) for d in docs]


def test_encode_to_numpy(enc):
    arr = toktok.encode_to_numpy(enc, TEXT)
    assert arr.dtype.name == "uint32"
    assert list(arr) == enc.encode_ordinary(TEXT)


def test_decode_batch(enc):
    docs = ["alpha", "日本語", ""]
    ids = [enc.encode_ordinary(d) for d in docs]
    assert enc.decode_batch(ids) == docs


def test_encode_bytes_invalid_utf8(enc):
    for bad in [b"\xe4\xb8", b"\x80\x80", b"abc\xf0\x9f\x9a"]:
        assert enc.decode_bytes(enc.encode_bytes(bad)) == bad


def test_encoding_for_model():
    assert toktok.encoding_for_model("gpt-4").name == "cl100k_base"
    assert toktok.encoding_for_model("gpt-4o-mini").name == "o200k_base"
    assert toktok.encoding_for_model("openai/gpt-oss-20b").name == "o200k_harmony"
    with pytest.raises(KeyError):
        toktok.encoding_for_model("not-a-model")


def test_unknown_encoding_raises():
    with pytest.raises(RuntimeError, match="unknown encoding"):
        toktok.get_encoding("does_not_exist")


def test_harmony_specials():
    h = toktok.get_encoding("o200k_harmony")
    assert "<|message|>" in h.special_tokens_set
    assert h.encode_ordinary("hello") == toktok.get_encoding("o200k_base").encode_ordinary("hello")
