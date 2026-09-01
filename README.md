# toktok

A fast, exact BPE tokenizer in **Rust**, with Python wheels (`pip install toktok-rs`, imported as `toktok`).

Token ids are **byte-identical to [tiktoken](https://github.com/openai/tiktoken)** —
verified against it in CI on fixed vectors, a randomized Unicode fuzz, and three
25 MB corpora — and encoding runs **3–7× faster** than tiktoken.

It is a Rust port of [quicktok](https://github.com/dmatth1/quicktok)'s C++ engine:
same algorithm, same data-structure engineering, same throughput (see
[Benchmarks](#benchmarks)).

- **Exact** — ids match tiktoken byte-for-byte; every benchmark is exactness-checked before timing.
- **Drop-in** — tiktoken-style Python API (`get_encoding`, `encode`, `decode`, `encode_ordinary`, …).
- **Self-contained** — no runtime dependencies; cl100k_base, o200k_base and o200k_harmony ship in the wheel.
- **Thread-safe** — load once, call `encode()` from as many threads as you like; `encode_batch()` scales across cores.

## Install

```sh
pip install toktok-rs       # Python ≥ 3.9, Linux / macOS / Windows — imports as `toktok`
```

Rust:

```toml
[dependencies]
toktok-core = "0.1"
```

## Quickstart

```python
import toktok

enc = toktok.get_encoding("cl100k_base")
ids = enc.encode("hello world")            # raises on a stray special, like tiktoken
text = enc.decode(ids)
n = enc.count("how many tokens is this?")

arr = toktok.encode_to_numpy(enc, big_text)          # uint32 array — fastest single-encode path
ids, offsets = toktok.encode_batch_to_numpy(enc, docs, threads=8)   # parallel
counts = toktok.count_batch(enc, docs)               # int64 array of per-doc token counts

enc = toktok.encoding_for_model("gpt-4o")            # -> o200k_base
```

Rust:

```rust
let tok = toktok_core::Tokenizer::load_dir("python/toktok/data", "cl100k_base")?;
let ids = tok.encode("Hello, toktok! 日本語 🚀".as_bytes());
let text = tok.decode(&ids);               // lossless round-trip, even on invalid UTF-8
```

## Benchmarks

Apple M-series laptop, single thread, 25 MB corpora, every output verified
token-for-token identical before timing (`python bench/compare.py`). Throughput
in **MB/s**; `quicktok (C++)` is the original this port is measured against.

**cl100k_base** (GPT-3.5 / GPT-4)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **toktok (numpy)** | **62.4** | **105.2** | 45.0 |
| quicktok (C++) | 60.8 | 102.5 | **48.9** |
| toktok (list[int]) | 54.4 | 81.2 | 39.9 |
| tiktoken | 16.2 | 15.4 | 13.5 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **toktok (numpy)** | **72.9** | **83.8** | **40.8** |
| quicktok (C++) | 68.8 | 80.9 | 35.5 |
| toktok (list[int]) | 55.5 | 60.1 | 38.3 |
| tiktoken | 23.8 | 23.6 | 15.5 |

`encode_to_numpy()` returns a `uint32` array directly, skipping the per-token
Python-int marshalling — from Python it runs at near-native speed. Absolute
MB/s is machine- and thermal-dependent (this laptop swings ±15% run to run);
the same-run ratios are the stable signal.

Native (no Python in the measurement, `cargo run --release --example bench`)
the Rust core lands within run-to-run noise of the C++ original — roughly
90–105% of it depending on corpus and encoding.

Reproduce:

```sh
python bench/fetch_corpus.py     # streams 3 × 25 MB from their real sources
python bench/compare.py          # toktok vs quicktok vs tiktoken, exactness-checked
cargo run --release --example bench -- bench/corpus/pile.txt cl100k_base
```

## Encodings

| name | model family | reference |
|---|---|---|
| `cl100k_base` | GPT-3.5 / GPT-4 / text-embedding-3 | tiktoken (the default) |
| `o200k_base` | GPT-4o, o1/o3/o4-mini, GPT-4.1, GPT-5 | tiktoken |
| `o200k_harmony` | GPT-OSS | tiktoken — o200k_base ranks + harmony specials |

Llama-3, Qwen (with its NFC normalizer), Llama-4 and Mistral Tekken exist in the
upstream C++ and are not ported yet.

## How it's fast

Same algorithm as [`bpe`](https://github.com/github/rust-gems) (exact
backtracking BPE) — the speed is data-structure engineering, carried over from
quicktok:

- **2-byte trie** — the longest-match walk reads 2 input bytes per single 8-byte slot load, with a zero-lookup direct table for CJK characters.
- **Dense validity memos** — merge-validity checks hit exactly-keyed caches (2 MB for 17-bit token ids, a wider one for 200k-vocab ids; a bijective mixer means no aliasing, ever).
- **Specialized pretokenizers** — the fixed cl100k/o200k regexes are compiled by hand into SIMD scanners; no general regex engine anywhere.
- **Single-pass product machines** — for ASCII text (most of code and English), one loop owns both the pretokenizer's boundary rules and token emission; only Unicode contact falls back to the general scanner, one piece at a time.

## Layout

```
crates/toktok-core   the engine (zero dependencies)
crates/toktok-py     PyO3 bindings -> toktok._toktok
python/toktok        the Python package + bundled vocab data
bench/               corpus fetcher + cross-encoder benchmark
tools/gen_vectors.py regenerates the exactness fixtures from tiktoken
```

## Development

```sh
uv venv && uv pip install maturin pytest tiktoken numpy
maturin develop --release
cargo test --release          # exactness fixtures, offsets, batch, invalid UTF-8
pytest                        # parity vs tiktoken + the Python surface
```

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for the upstream attributions
(quicktok, MIT; the vocabulary data derives from tiktoken, MIT).
