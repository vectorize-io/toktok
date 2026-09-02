"""Byte-exactness vs tiktoken: fixed cases, a randomized fuzz, and (when the
benchmark corpora have been fetched) the full 25 MB files."""

import glob
import os
import random

import pytest
import toktok

tiktoken = pytest.importorskip("tiktoken")

ENCODINGS = ["cl100k_base", "o200k_base"]

CASES = [
    "",
    " ",
    "\n",
    "hello world",
    "  \n\n\t mixed \r\n whitespace  ",
    "it's I'M they'RE won't we'll you've",
    "'s 'S 'll 'LL 've 'd 'm 'x '",
    "def f(x):\n    return {'a': [1, 2]}  # comment\n",
    "0 1 12 123 1234 3.14159 1e-9 1_000_000",
    "ÀÉÎÕÜ àéîõü ĄĆĘŁŃÓŚŹŻ",
    "Привет, мир!",
    "العربية نص",
    "日本語のテキストです。",
    "中文测试，标点。",
    "한국어 테스트",
    "亚洲AV 中文AB混合Test",
    "🚀🎉😀 👨‍👩‍👧‍👦 🇺🇸🇯🇵",
    "ＡＢＣ ｆｕｌｌｗｉｄｔｈ",
    "MiXeD CamelCase SCREAMING_SNAKE",
    "https://example.com/a/b?c=d#e",
    "a" * 500,
    "日本" * 200,
    "word " * 200,
]

ALPHABET = "abcXYZ 0123\n\t\r.,!?'-/_:;()[]{}éü中日あ한\U0001f600́A"


@pytest.fixture(scope="module", params=ENCODINGS)
def pair(request):
    name = request.param
    return name, toktok._encoding(name), tiktoken.get_encoding(name)


def test_fixed_cases(pair):
    _, a, b = pair
    for s in CASES:
        assert a.encode_ordinary(s) == b.encode_ordinary(s), repr(s)


def test_fuzz(pair):
    _, a, b = pair
    rng = random.Random(1234)
    for _ in range(4000):
        s = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 80)))
        assert a.encode_ordinary(s) == b.encode_ordinary(s), repr(s)


def test_random_unicode_fuzz(pair):
    """Codepoints drawn from the whole BMP + astral planes, not just a curated set."""
    _, a, b = pair
    rng = random.Random(99)
    for _ in range(1500):
        n = rng.randint(0, 40)
        s = "".join(
            chr(
                rng.choice(
                    [
                        rng.randint(1, 0x2FFF),
                        rng.randint(0x3000, 0xD7FF),
                        rng.randint(0xE000, 0xFFFF),
                        rng.randint(0x10000, 0x10FFFF),
                    ]
                )
            )
            for _ in range(n)
        )
        assert a.encode_ordinary(s) == b.encode_ordinary(s), repr(s)


def test_n_vocab_and_specials(pair):
    _, a, b = pair
    assert a.n_vocab == b.n_vocab
    assert a.max_token_value == b.max_token_value
    assert a.special_tokens_set == b.special_tokens_set
    assert a.eot_token == b.eot_token


def test_encode_with_allowed_special(pair):
    _, a, b = pair
    text = "hi <|endoftext|> there"
    assert a.encode(text, allowed_special="all") == b.encode(text, allowed_special="all")
    assert a.encode(text, allowed_special={"<|endoftext|>"}) == b.encode(
        text, allowed_special={"<|endoftext|>"}
    )


def test_disallowed_special_raises(pair):
    _, a, _b = pair
    with pytest.raises(ValueError, match="disallowed special token"):
        a.encode("hi <|endoftext|> there")
    # ...and is silent when the check is disabled
    assert a.encode("hi <|endoftext|> there", disallowed_special=()) == a.encode_ordinary(
        "hi <|endoftext|> there"
    )


CORPORA = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), "..", "bench", "corpus", "*.txt"))
)


@pytest.mark.skipif(not CORPORA, reason="run bench/fetch_corpus.py for the corpus test")
@pytest.mark.parametrize("path", CORPORA, ids=lambda p: os.path.basename(p))
def test_corpus(pair, path):
    _, a, b = pair
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    assert a.encode_ordinary(text) == b.encode_ordinary(text)
