"""toktok — a fast, exact BPE tokenizer (Rust core).

The public API is one function: `batch_count`. Give it your texts and an
encoding, get back a token count per text. Ids are never materialized.

    import toktok

    counts = toktok.batch_count(["hello world", "how many tokens?"], "cl100k_base")
    # [2, 4]

`encoding` takes an encoding name (cl100k_base, o200k_base, o200k_harmony) or a
model name (gpt-4o, gpt-4, openai/gpt-oss-20b, text-embedding-3-small).

Encodings are loaded once and cached, and counting releases the GIL and runs
across threads, so calling this per request is fine.
"""

from typing import Iterable, List

from ._toktok import BUILTIN_ENCODINGS as _BUILTIN
from ._toktok import Tokenizer as _Tokenizer
from ._toktok import __version__

# The vocabularies are compiled into the extension module, so there are no data
# files to find, ship or download.
_CACHE = {}

# model name -> encoding. Lookup lowercases and strips an org prefix, so HF-style
# ids like "openai/gpt-oss-20b" resolve too.
_MODEL_TO_ENCODING = {
    "gpt-5": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-oss": "o200k_harmony",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}


def _resolve(name: str) -> str:
    """Encoding name, or the encoding a model name maps to."""
    if name in _BUILTIN:
        return name
    m = name.lower().rsplit("/", 1)[-1]  # "openai/gpt-oss-20b" -> "gpt-oss-20b"
    for prefix, enc in sorted(_MODEL_TO_ENCODING.items(), key=lambda kv: -len(kv[0])):
        if m == prefix or m.startswith(prefix + "-") or m.startswith(prefix + "."):
            return enc
    raise KeyError(
        f"unknown encoding or model {name!r}; bundled encodings are "
        f"{', '.join(_BUILTIN)}"
    )


def _encoding(name: str, data_dir: str = "") -> _Tokenizer:
    """The loaded (and cached) tokenizer behind an encoding or model name.

    Private: the supported API is `batch_count`. This is here for tests and for
    anyone who knowingly wants the full encode/decode surface."""
    key = (name, data_dir)
    if key not in _CACHE:
        _CACHE[key] = _Tokenizer(_resolve(name), data_dir)
    return _CACHE[key]


def batch_count(
    texts: Iterable[str],
    encoding: str = "cl100k_base",
    threads: int = 0,
    with_special: bool = False,
) -> List[int]:
    """Count the tokens in each of `texts`. Returns one int per text, in order.

    Args:
        texts: the strings to count.
        encoding: encoding name ('cl100k_base', 'o200k_base', 'o200k_harmony')
            or a model name ('gpt-4o', 'openai/gpt-oss-20b').
        threads: worker threads; 0 (default) uses every core. Counting releases
            the GIL, so this scales.
        with_special: if True, a special-token string such as '<|endoftext|>'
            counts as the single token it is, instead of as ordinary text.

    Counting never builds the token ids: one scratch buffer per thread is reused
    across texts, so a batch of any size allocates O(threads), not O(tokens).

        >>> toktok.batch_count(["hello world"], "gpt-4o")
        [2]
    """
    return _encoding(encoding).count_batch(list(texts), threads, with_special)


__all__ = ["batch_count"]
