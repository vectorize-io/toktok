"""toktok — a fast, exact BPE tokenizer (Rust core).

Bundled encodings: cl100k_base, o200k_base, o200k_harmony.
Drop-in-shaped for tiktoken:

    import toktok
    enc = toktok.get_encoding("cl100k_base")
    ids = enc.encode("hello world")          # == tiktoken.encode
    text = enc.decode(ids)
"""

import os as _os

from ._toktok import BUILTIN_ENCODINGS, Tokenizer, __version__, _set_datadir

_set_datadir(_os.path.join(_os.path.dirname(__file__), "data"))

_CACHE = {}

# model name -> encoding (the common ones). encoding_for_model lowercases and
# strips an org prefix, so HF-style ids like "openai/gpt-oss-20b" resolve.
MODEL_TO_ENCODING = {
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


def get_encoding(name: str, data_dir: str = "") -> Tokenizer:
    """Load (and cache) a tokenizer by encoding name.

    Bundled: 'cl100k_base', 'o200k_base', 'o200k_harmony' (GPT-OSS)."""
    key = (name, data_dir)
    if key not in _CACHE:
        _CACHE[key] = Tokenizer(name, data_dir)
    return _CACHE[key]


def encoding_for_model(model: str) -> Tokenizer:
    """tiktoken-style: resolve a model name to its encoding."""
    m = model.lower().rsplit("/", 1)[-1]
    for prefix, enc in sorted(MODEL_TO_ENCODING.items(), key=lambda kv: -len(kv[0])):
        if m == prefix or m.startswith(prefix + "-") or m.startswith(prefix + "."):
            return get_encoding(enc)
    raise KeyError(
        f"unknown model {model!r}; pass an encoding name to get_encoding() instead"
    )


def encode_to_numpy(enc: Tokenizer, text: str, **kw):
    """Encode to a uint32 numpy array — the fastest single-encode path from
    Python (no per-token Python int objects)."""
    import numpy as _np

    return _np.frombuffer(enc.encode_to_buffer(text, **kw), dtype=_np.uint32)


def encode_batch_to_numpy(enc: Tokenizer, texts, threads: int = 0, with_special: bool = False):
    """Encode many texts in parallel. Returns (flat uint32 ids, int64 offsets)
    where text i occupies ids[offsets[i]:offsets[i + 1]]."""
    import numpy as _np

    flat, offsets = enc.encode_batch(list(texts), threads, with_special)
    return _np.frombuffer(flat, dtype=_np.uint32), _np.frombuffer(offsets, dtype=_np.int64)


def count_batch(enc: Tokenizer, texts, threads: int = 0, with_special: bool = False):
    """Token counts for many texts, in parallel. Returns a numpy int64 array.
    Faster than len(encode(t)) per text — no Python list of ids is ever built."""
    import numpy as _np

    _flat, offsets = encode_batch_to_numpy(enc, texts, threads, with_special)
    return _np.diff(offsets)


__all__ = [
    "Tokenizer",
    "get_encoding",
    "encoding_for_model",
    "encode_to_numpy",
    "encode_batch_to_numpy",
    "count_batch",
    "BUILTIN_ENCODINGS",
    "MODEL_TO_ENCODING",
    "__version__",
]
