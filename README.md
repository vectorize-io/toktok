# toktok

A fast, exact BPE tokenizer in **Rust**, with **Python** bindings.

Token ids are **byte-identical to [tiktoken](https://github.com/openai/tiktoken)**
— verified against it on fixed vectors, a randomized Unicode fuzz and three 25 MB
corpora — while encoding **3–7× faster**, using **3–5× less CPU per MB**, and with
a **p99 latency ~4× lower**.

```sh
uv add toktok-rs               # Python — imports as `toktok`
cargo add toktok-rs            # Rust   — the library is `toktok`
```

Wheels for CPython **3.11 – 3.14** on Linux (x86_64, aarch64), macOS
(universal2) and Windows, plus a **free-threaded `cp314t`** wheel: the extension
declares `gil_used = false`, so a no-GIL interpreter keeps the GIL off and
`batch_count` scales across threads without one. (3.13t is not supported — PyO3
rejects the free-threaded build below 3.14.)

- **Exact** — ids match tiktoken byte-for-byte; every benchmark verifies before it times.
- **Self-contained** — vocabularies are compiled in. No data files, no downloads, no runtime dependencies.
- **Fast where it counts** — see [Benchmarks](#benchmarks) and [Resource profile](#resource-profile).
- **Thread-safe** — load once, count or encode from every core.

Bundled encodings: `cl100k_base` (GPT-3.5/GPT-4), `o200k_base` (GPT-4o),
`o200k_harmony` (GPT-OSS).

## Python API

The public API is one function.

```python
import toktok

counts = toktok.batch_count(["hello world", "how many tokens is this?"])
# [2, 6]
```

### `batch_count(texts, encoding="cl100k_base", threads=0, with_special=False) -> list[int]`

Counts the tokens in each text and returns one `int` per text, in order.

| argument | meaning |
|---|---|
| `texts` | any iterable of `str`. |
| `encoding` | an encoding name (`cl100k_base`, `o200k_base`, `o200k_harmony`) **or a model name** (`gpt-4o`, `gpt-4`, `openai/gpt-oss-20b`, `text-embedding-3-small`). |
| `threads` | worker threads. `0` (default) uses every core. Counting releases the GIL, so this scales. |
| `with_special` | when `True`, a special string like `<\|endoftext\|>` counts as the single token it is instead of as ordinary text. |

```python
import toktok

# by model name — resolves to that model's encoding
toktok.batch_count(["hello world"], "gpt-4o")                 # [2]
toktok.batch_count(["hello world"], "openai/gpt-oss-20b")     # [2]

# a big batch, pinned to 4 threads
counts = toktok.batch_count(docs, "text-embedding-3-small", threads=4)

# special tokens counted as tokens rather than as text
toktok.batch_count(["a<|endoftext|>b"], with_special=True)     # [3]
```

Counting **never builds the token ids**: one scratch buffer per thread is reused
across texts, so a batch of any size allocates O(threads), not O(tokens). That is
both faster and dramatically lighter than `len(enc.encode(text))` — see
[Resource profile](#resource-profile).

Encodings are loaded once and cached, so calling `batch_count` per request is
fine; the first call for an encoding pays ~50 ms to build its tables.

An unknown encoding or model name raises `KeyError`.

<details>
<summary>Need ids, decoding or offsets from Python?</summary>

The full engine is one underscore away — `toktok._encoding(name)` returns the
`Tokenizer` that `batch_count` uses, with `encode`, `encode_ordinary`,
`encode_to_numpy`, `encode_batch`, `encode_batch_to_numpy`, `encode_with_offsets`,
`decode`, `decode_bytes`, `count`, `count_batch`, `memory_bytes` and the
tiktoken-compatible properties (`n_vocab`, `eot_token`, `special_tokens_set`).

It is deliberately private: a much larger surface than `batch_count`, and not the
API this package promises to keep stable. If you need ids in production, say so
and it can be promoted.

```python
enc = toktok._encoding("cl100k_base")
ids = enc.encode_ordinary("hello world")     # [15339, 1917]
enc.decode(ids)                              # 'hello world'
arr = enc.encode_to_numpy(big_text)          # uint32 numpy array — fastest path
```
</details>

## Rust API

```toml
[dependencies]
toktok-rs = "0.1"
```

```rust
let tok = toktok::Tokenizer::builtin("cl100k_base")?;      // nothing to download

let ids = tok.encode(b"hello world");
let text = tok.decode(&ids);                  // lossless, even on invalid UTF-8
let n = tok.count(b"how many tokens is this?");

let docs: Vec<&[u8]> = vec![b"first", b"second"];
let counts = tok.count_batch(&docs, 0, false);   // parallel; 0 threads = every core
let batch = tok.encode_batch(&docs, 0, false);

let (ids, bounds) = tok.encode_with_offsets(b"per-token byte spans");
tok.memory_bytes();                              // exact table footprint
```

`Tokenizer` is `Send + Sync`: load it once and share it. Full docs on
[docs.rs](https://docs.rs/toktok-rs). Turn off the default `embedded-data`
feature to load vocabularies from a directory instead of compiling them in.

## Benchmarks

Single thread, three 25 MB corpora, Apple M-series, every encoder's ids verified
identical before timing.

**Against the Rust tokenizers** (`bench/rust`, no Python in the measurement).
Speedups are same-run ratios across both encodings and all three corpora —
absolute MB/s moves with machine load, the ratio holds:

| vs | encode throughput | p99 latency per document |
|---|---|---|
| [bpe-openai](https://crates.io/crates/bpe-openai) | **1.9–3.8× faster** | 1.3–20× lower |
| [tiktoken-rs](https://crates.io/crates/tiktoken-rs) | **3.4–20× faster** | 3.8–27× lower |

Measured by the **Benchmarks** workflow on a GitHub runner (The Pile,
cl100k_base, single thread pinned with `taskset`):

| encoder | MB/s | p50 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| **toktok** | **99.6** | **0.6 µs** | **11.7 µs** | **24.0 µs** |
| bpe-openai | 30.2 | 2.0 µs | 42.6 µs | 90.5 µs |
| tiktoken-rs | 7.4 | 7.7 µs | 164.3 µs | 347.8 µs |

**From Python** (`bench/compare.py`), cl100k_base:

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **toktok (numpy path)** | **62.4** | **105.2** | 45.0 |
| quicktok (C++, the original) | 60.8 | 102.5 | **48.9** |
| tiktoken | 16.2 | 15.4 | 13.5 |

Absolute MB/s is machine- and thermal-dependent (this laptop swings ±15% run to
run); the same-run ratios are the stable signal.

```sh
uv run python bench/fetch_corpus.py                   # 3 × 25 MB, streamed
cargo run --release --manifest-path bench/rust/Cargo.toml -- bench/corpus/pile.txt cl100k_base
uv run python bench/compare.py                        # vs tiktoken / quicktok
uv run python bench/profile.py --threads 8            # memory, CPU, latency
```

Or run the whole set on CI, where it is isolated from whatever else your laptop
is doing: the **Benchmarks** workflow (manual dispatch, weekly, and on release
tags) publishes the tables to its job summary.

## Resource profile

Throughput is only part of the story: what a tokenizer costs in production is RAM
per process, CPU per MB, and tail latency per request. `bench/profile.py` measures
all three, running **each encoder in its own subprocess** so RSS is attributable.
Numbers below: The Pile, 8 threads.

**Memory** — RSS growth of a bare process that only loads the encoding:

| encoder | cl100k_base | o200k_base | ids for 3 255 Common-Crawl docs |
|---|---:|---:|---:|
| **toktok** | **37.5 MiB** (14.0 live) | **72.4 MiB** (27.2 live) | 25.4 MiB |
| tiktoken | 48.3 MiB | 86.5 MiB | 212.6 MiB |
| quicktok (C++) | 50.4 MiB | 99.7 MiB | 20.3 MiB |

`memory_bytes` reports the exact live table bytes (14.0 / 27.2 MiB — the rest of
RSS is construction scratch the allocator hasn't returned). tiktoken's ids cost
~8× more because a `list[int]` is 8 bytes of pointer plus a 28-byte object per
token, where the numpy and batch paths hand back one `uint32` buffer.

**CPU** — CPU-seconds per MB encoded (single thread, lower is better):

| encoder | cl100k_base | o200k_base |
|---|---:|---:|
| **toktok** | **0.0117** | **0.0123** |
| quicktok (C++) | 0.0118 | 0.0132 |
| tiktoken | 0.0598 | 0.0395 |

That is 3–5× less CPU for the same bytes — the number that sets how many cores an
ingestion pipeline burns on tokenization.

**Latency** — per-document encode, 4 000 Pile documents (mean 0.3 KiB), µs:

| encoder | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|
| **toktok** | **1.8** | **10.0** | **32.5** | **103.6** | **167.9** |
| quicktok (C++) | 1.9 | 10.3 | 36.2 | 108.6 | 150.8 |
| tiktoken | 7.2 | 41.8 | 156.3 | 565.2 | 905.9 |

p99 is ~4.8× lower than tiktoken and the tail is tighter still at p99.9 (5.5×).
On larger documents (Common Crawl, mean 7.5 KiB) the ordering holds: p99 1.97 ms
vs tiktoken's 6.31 ms.

**Counting without ids** — what `batch_count` buys you:

| operation (4 000 Pile docs, 8 threads) | throughput | RSS growth |
|---|---:|---:|
| `toktok.batch_count()` | 426 MB/s | **0.2 MiB** |
| toktok `encode_batch()` | 432 MB/s | 4.4 MiB |
| tiktoken `encode_ordinary_batch()` | 12.7 MB/s | 11.5 MiB |

## How it's fast

Same algorithm as [`bpe`](https://github.com/github/rust-gems) (exact backtracking
BPE) — the speed is data-structure engineering, ported from
[quicktok](https://github.com/dmatth1/quicktok)'s C++:

- **2-byte trie** — the longest-match walk reads 2 input bytes per single 8-byte slot load, with a zero-lookup direct table for CJK characters.
- **Dense validity memos** — merge-validity checks hit exactly-keyed caches (2 MB for 17-bit token ids, a wider one for 200k-vocab ids; a bijective mixer means no aliasing, ever).
- **Specialized pretokenizers** — the fixed cl100k/o200k regexes are compiled by hand into SIMD scanners; no general regex engine anywhere.
- **Single-pass product machines** — for ASCII text (most of code and English), one loop owns both the pretokenizer's boundary rules and token emission; only Unicode contact falls back to the general scanner, one piece at a time.

## Layout

```
crates/toktok        the engine + embedded vocabularies (crates.io: toktok-rs)
crates/toktok-py     PyO3 bindings -> toktok._toktok  (PyPI: toktok-rs)
python/toktok        the Python package
bench/               corpus fetcher, throughput (compare.py), resource
                     profile (profile.py), Rust-vs-Rust comparison (rust/)
tools/gen_vectors.py regenerates the exactness fixtures from tiktoken
```

## Development

```sh
uv sync                       # venv + dev deps + builds the extension
uv run pytest                 # parity vs tiktoken + the Python surface
cargo test --release          # exactness fixtures, offsets, batch, invalid UTF-8

uv build                      # wheel + sdist into dist/
```

The Rust extension is compiled by [maturin](https://www.maturin.rs), declared as
the PEP 517 build backend in `pyproject.toml` — uv drives it, so `uv sync`,
`uv run` and `uv build` are the only commands you need. To rebuild after a change
to the Rust sources, `uv sync --reinstall-package toktok-rs`.

Benchmarks need their own extras and the corpora:

```sh
uv sync --group bench
uv run python bench/fetch_corpus.py     # 3 x 25 MB, streamed from source
uv run python bench/compare.py
```

## Releasing

Tag-driven: `git tag v0.1.0 && git push origin v0.1.0` builds every wheel and
publishes to PyPI and crates.io. Both registries use Trusted Publishing, so
publishing rights belong to this repository rather than to a personal token —
setup and the first-publish bootstrap are in [RELEASING.md](RELEASING.md).

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for the upstream attributions
(quicktok, MIT; the vocabulary data derives from tiktoken, MIT).
