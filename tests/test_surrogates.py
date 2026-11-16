"""Lone UTF-16 surrogates (legal in Python and JSON) are counted, not raised on (#8).

They become U+FFFD the way tiktoken does it, so counts match tiktoken."""

import pytest
import toktok

LONE = "hi \ud83e there"
FIXED = LONE.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def test_lone_surrogate_counts_like_tiktoken():
    tiktoken = pytest.importorskip("tiktoken")
    want = len(tiktoken.get_encoding("o200k_base").encode(LONE))
    assert toktok.batch_count([LONE], "o200k_base", threads=1) == [want]


def test_one_bad_text_does_not_fail_the_batch():
    assert toktok.batch_count(["ok", LONE, "ok"], "cl100k_base") == toktok.batch_count(
        ["ok", FIXED, "ok"], "cl100k_base"
    )


def test_every_entry_point():
    for enc in ("cl100k_base", "qwen3.5", "embeddinggemma-2", "jev"):
        assert toktok.batch_count([LONE], enc) == toktok.batch_count([FIXED], enc)
        assert toktok.truncate(LONE, 1, enc)[1] == toktok.truncate(FIXED, 1, enc)[1]
        assert toktok.batch_truncate([LONE], 1, enc) == toktok.batch_truncate([FIXED], 1, enc)
    t = toktok._encoding("o200k_base")
    assert t.encode(LONE) == t.encode(FIXED)
    assert t.count(LONE) == t.count(FIXED)


def test_split_pair_joins_back():
    # two halves of one emoji, as separate code points, read as the emoji
    assert toktok.batch_count(["🤔"], "o200k_base") == toktok.batch_count(["🤔"], "o200k_base")
