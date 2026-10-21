"""toktok — a fast, exact BPE tokenizer (Rust core).

The public API is one function: `batch_count`. Give it your texts and an
encoding, get back a token count per text. Ids are never materialized.

    import toktok

    counts = toktok.batch_count(["hello world", "how many tokens?"], "cl100k_base")
    # [2, 4]

`encoding` is one of the bundled encodings: cl100k_base, o200k_base,
o200k_harmony, qwen3.5 — or 'jev', an upper-bound estimate for TypeSafe's Jev.

Encodings are loaded once and cached, and counting releases the GIL and runs
across threads, so calling this per request is fine.
"""

from collections.abc import Iterable
from unicodedata import is_normalized, normalize

from ._toktok import BUILTIN_ENCODINGS as _BUILTIN
from ._toktok import Tokenizer as _Tokenizer
from ._toktok import __version__ as __version__  # re-exported
from ._toktok import jev_count_batch as _jev_count_batch
from ._toktok import jev_truncate as _jev_truncate
from ._toktok import jev_truncate_batch as _jev_truncate_batch

# The vocabularies are compiled into the extension module, so there are no data
# files to find, ship or download.
_CACHE = {}


def _encoding(name: str, data_dir: str = "") -> _Tokenizer:
    """The loaded (and cached) tokenizer for an encoding.

    Private: the supported API is `batch_count` and `truncate`. This is here for
    tests and for anyone who knowingly wants the full encode/decode surface."""
    if name == "jev":
        raise KeyError("'jev' is an estimate with no ids: use batch_count / truncate")
    key = (name, data_dir)
    if key not in _CACHE:
        if not data_dir and name not in _BUILTIN:
            raise KeyError(f"unknown encoding {name!r}; available: {', '.join(_BUILTIN)}, jev")
        _CACHE[key] = _Tokenizer(name, data_dir)
    return _CACHE[key]


def _nfc(text: str) -> str:
    # Qwen's tokenizer NFC-normalizes first; the Rust core doesn't, so do it here
    return text if is_normalized("NFC", text) else normalize("NFC", text)


def _nfc_batch(texts: Iterable[str], threads: int) -> list[str]:
    # the core finds the few texts NFC could change, in parallel
    texts = list(texts)
    for i in _encoding("qwen3.5").unnormalized(texts, threads):
        texts[i] = _nfc(texts[i])
    return texts


def batch_count(
    texts: Iterable[str],
    encoding: str = "cl100k_base",
    threads: int = 0,
    with_special: bool = False,
) -> list[int]:
    """Count the tokens in each of `texts`. Returns one int per text, in order.

    Args:
        texts: the strings to count.
        encoding: one of 'cl100k_base' (GPT-3.5, GPT-4, text-embedding-3),
            'o200k_base' (GPT-4o, o-series, GPT-4.1, GPT-5), 'o200k_harmony'
            (GPT-OSS) or 'qwen3.5' (Qwen3.5's vocab as bigpickle uses it;
            text is NFC-normalized first). An unknown name raises
            KeyError. 'jev' (TypeSafe's
            Jev) is an upper-bound estimate, ~12-19% over on average: safe for
            splitting a payload under Jev's limits, not an exact count.
            `threads` applies; `with_special` doesn't (Jev has none).
        threads: worker threads; 0 (default) uses every core. Counting releases
            the GIL, so this scales.
        with_special: if True, a special-token string such as '<|endoftext|>'
            counts as the single token it is, instead of as ordinary text.

    Counting never builds the token ids: one scratch buffer per thread is reused
    across texts, so a batch of any size allocates O(threads), not O(tokens).

        >>> toktok.batch_count(["hello world"], "o200k_base")
        [2]
    """
    if encoding == "jev":
        return _jev_count_batch(list(texts), threads)
    if encoding == "qwen3.5":
        texts = _nfc_batch(texts, threads)
    return _encoding(encoding).count_batch(list(texts), threads, with_special)


def truncate(
    text: str,
    max_tokens: int,
    encoding: str = "cl100k_base",
) -> tuple[str, int]:
    """Cut `text` down to at most `max_tokens` tokens.

    Returns `(text, total_tokens)`. `total_tokens` is the count of the *whole*
    input, truncated or not, so you can report how much was dropped:

        >>> toktok.truncate("hello world", 1, "o200k_base")
        ('hello', 2)

    One pass over the input, no token ids built, and nothing decoded — the cut
    is a byte offset into the string you passed in. When nothing needs cutting
    the original string object is returned as-is.

    The cut always lands on a character boundary. Byte-level BPE can split one
    character across tokens ("🧠" is three tokens), so cutting at the token
    boundary — which is what `decode(encode(text)[:n])` does — can leave a
    partial character that decodes to U+FFFD. This drops that partial character
    instead.

    With 'jev' the counts are the upper-bound estimate, and the cut lands where
    the kept text's estimate fits `max_tokens`. With 'qwen3.5' the text is
    NFC-normalized first, so the kept text is a prefix of its NFC form.
    """
    if encoding == "jev":
        return _jev_truncate(text, max_tokens)
    if encoding == "qwen3.5":
        text = _nfc(text)
    return _encoding(encoding).truncate(text, max_tokens)


def batch_truncate(
    texts: Iterable[str],
    max_tokens: int,
    encoding: str = "cl100k_base",
    threads: int = 0,
) -> list[tuple[str, int]]:
    """`truncate` over many texts, in parallel. See `truncate` and `batch_count`."""
    if encoding == "jev":
        return _jev_truncate_batch(list(texts), max_tokens, threads)
    if encoding == "qwen3.5":
        texts = _nfc_batch(texts, threads)
    return _encoding(encoding).truncate_batch(list(texts), max_tokens, threads)


__all__ = ["batch_count", "batch_truncate", "truncate"]
