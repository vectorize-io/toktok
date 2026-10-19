"""The 'jev' estimate against counts TypeSafe's API reported (tests/jev_counts.json),
and the same surface the exact encodings have: batch_count, truncate, batch_truncate."""

import json
import os
import unicodedata

import pytest
import toktok

CASES = [
    " ",
    "\n",
    "  \n\n\t mixed \r\n whitespace  ",
    "it's I'M they'RE won't we'll you've",
    "def f(x):\n    return {'a': [1, 2]}  # comment\n",
    "0 1 12 123 1234 3.14159 1e-9 1_000_000",
    "Привет, мир! العربية نص 日本語のテキストです。 한국어 테스트",
    "🚀🎉😀 👨\u200d👩\u200d👧\u200d👦 🇺🇸🇯🇵 ＡＢＣ",
    "MiXeD CamelCase SCREAMING_SNAKE https://example.com/a/b?c=d#e",
    "a" * 500,
    "word " * 200,
    "\x00\x01 \ud7ff \ue000 \U0010ffff",
]
with open(os.path.join(os.path.dirname(__file__), "jev_counts.json"), encoding="utf-8") as f:
    JEV = json.load(f)["cases"]


def test_never_under_jev():
    est = toktok.batch_count([c["text"] for c in JEV], "jev")
    under = [
        (c["source"], c["text"][:30], c["jev"], e)
        for c, e in zip(JEV, est, strict=True)
        if e < c["jev"]
    ]
    assert not under


def test_overcount_stays_modest():
    long = [c for c in JEV if c["jev"] >= 100 and not c["source"].startswith("edge")]
    est = toktok.batch_count([c["text"] for c in long], "jev")
    assert sum(est) / sum(c["jev"] for c in long) < 1.3


def test_nfd_only_counts_higher():
    # Jev NFC-normalizes first; the estimate doesn't, so decomposed text only counts higher
    nfc = "한국어 텍스트입니다. 안녕하세요! " * 10
    nfd = unicodedata.normalize("NFD", nfc)
    assert toktok.batch_count([nfd], "jev")[0] >= toktok.batch_count([nfc], "jev")[0]


def test_threads_agree():
    texts = [c["text"] for c in JEV] + CASES
    assert toktok.batch_count(texts, "jev", threads=1) == toktok.batch_count(texts, "jev")


def test_empty_and_cases():
    assert toktok.batch_count([], "jev") == []
    assert toktok.batch_count([""], "jev") == [0]
    assert all(n > 0 for n in toktok.batch_count(CASES, "jev"))


@pytest.mark.parametrize("max_tokens", [0, 1, 5, 20, 100, 1000])
def test_truncate_fits_its_budget(max_tokens):
    texts = [c["text"] for c in JEV[::7]] + CASES
    totals = toktok.batch_count(texts, "jev")
    cuts = toktok.batch_truncate(texts, max_tokens, "jev")
    for text, total, (kept, reported) in zip(texts, totals, cuts, strict=True):
        assert reported == total
        assert text.startswith(kept)
        assert toktok.batch_count([kept], "jev")[0] <= max_tokens or kept == ""
        if total <= max_tokens:
            assert kept == text
        assert toktok.truncate(text, max_tokens, "jev") == (kept, total)


def test_truncate_returns_same_object_when_it_fits():
    text = "hello world, nothing to cut here"
    assert toktok.truncate(text, 10_000, "jev")[0] is text


def test_no_ids_for_jev():
    with pytest.raises(KeyError):
        toktok._encoding("jev")
    with pytest.raises(KeyError, match="jev"):
        toktok._encoding("nope")
