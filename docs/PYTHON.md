# The Python tokenizer behind `batch_count`

`toktok.batch_count` is the supported API. When you need token ids, decoding or
offsets, `toktok._encoding(name)` returns the `Tokenizer` it uses:

```python
import toktok

enc = toktok._encoding("cl100k_base")     # or "o200k_base", "o200k_harmony"
```

It is private on purpose — a much larger surface than `batch_count`, and not
what this package promises to keep stable. If you depend on it, say so and it
can be promoted.

## Encoding

| method | returns |
|---|---|
| `encode(text, allowed_special=None, disallowed_special=None)` | `list[int]` — tiktoken semantics: raises `ValueError` on a stray special token |
| `encode_ordinary(text)` | `list[int]`, specials treated as ordinary text |
| `encode_bytes(data)` | `list[int]` from raw bytes, no UTF-8 validation |
| `encode_with_special(text)` | `list[int]`, every special string becomes its id |
| `encode_to_numpy(text, ...)` | `uint32` numpy array — the fastest single-encode path |
| `encode_batch(texts, threads=0, with_special=False)` | `(ids_buffer, offsets_buffer)` as `bytes` |
| `encode_batch_to_numpy(texts, threads=0, with_special=False)` | `(uint32 ids, int64 offsets)`; text `i` is `ids[offsets[i]:offsets[i+1]]` |
| `encode_with_offsets(text, unit="byte")` | `(ids, spans)`; `unit="char"` gives code-point spans (HF `offset_mapping` shape) |
| `encode_single_token(piece)` | the id of an exact token; `KeyError` if it is not one |
| `count(text)` / `count_batch(texts, threads=0, with_special=False)` | token counts, no ids built |
| `truncate(text, max_tokens)` | `(text, total_tokens)` — see `toktok.truncate` |
| `truncate_batch(texts, max_tokens, threads=0)` | the same, in parallel |

`encode_to_numpy` and the batch paths avoid building a `list[int]`, which is
where most of the time and memory goes on large inputs — a Python list costs
8 bytes of pointer plus a 28-byte object per token.

## Decoding

| method | returns |
|---|---|
| `decode(ids, errors="replace")` | `str` |
| `decode_bytes(ids)` | `bytes` — lossless, even for ids that are not valid UTF-8 on their own |
| `decode_single_token_bytes(id)` | `bytes` for one token |
| `decode_batch(batch, errors="replace")` | `list[str]` |

## Inspection

| attribute | |
|---|---|
| `name` | the encoding name |
| `n_vocab` / `max_token_value` | tiktoken-compatible vocabulary size |
| `eot_token` | id of `<\|endoftext\|>` |
| `special_tokens_set` | `set[str]` |
| `special_tokens()` | `[(string, id), ...]` |
| `is_special_token(id)` | |
| `token_byte_values()` | every base-vocab token's bytes, indexed by id |
| `memory_bytes` | exact live table footprint |

## Thread safety

A `Tokenizer` is immutable and safe to share across threads; encode and count
release the GIL. On free-threaded CPython (3.14t) the module declares
`gil_used = false`, so importing it does not switch the GIL back on.

```python
import toktok
from concurrent.futures import ThreadPoolExecutor

enc = toktok._encoding("o200k_base")
with ThreadPoolExecutor(8) as pool:
    all_ids = list(pool.map(enc.encode_ordinary, docs))
```

For counting, prefer `toktok.batch_count(docs, threads=8)` — it parallelizes
internally and never materializes the ids.

## Custom encodings

`Tokenizer(name, datadir)` loads `<datadir>/<name>.vocab` and friends instead of
the embedded tables, in the binary format under `crates/toktok/data`.
