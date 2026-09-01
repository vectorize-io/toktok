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
- **Thread-safe** — load once, call `encode()` from as many threads as you like; `encode_batch()` and `count_batch()` scale across cores.
- **Cheap** — smaller tables than tiktoken, ~4× less CPU per MB, and a p99 per-document latency ~4× lower (see [Resource profile](#resource-profile)).

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
counts = enc.count_batch(docs, threads=8)            # counts only — ids never materialized

enc = toktok.encoding_for_model("gpt-4o")            # -> o200k_base
enc.memory_bytes                                     # exact RAM the tables cost
```

Rust:

```rust
let tok = toktok_core::Tokenizer::load_dir("python/toktok/data", "cl100k_base")?;
let ids = tok.encode("Hello, toktok! 日本語 🚀".as_bytes());
let text = tok.decode(&ids);               // lossless round-trip, even on invalid UTF-8
let counts = tok.count_batch(&docs, 8, false);        // parallel, allocates O(threads)
let (bytes, by_table) = (tok.memory_bytes(), tok.vocab().memory_breakdown());
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
python bench/profile.py --threads 8   # memory, CPU and latency percentiles
cargo run --release --example bench -- bench/corpus/pile.txt cl100k_base
```

## Resource profile

Throughput is only part of the story: what a tokenizer costs you in production is
RAM per process, CPU per MB, and tail latency per request. `bench/profile.py`
measures all three, running **each encoder in its own subprocess** so RSS is
attributable to that encoder alone, and checking ids agree before measuring
anything. Numbers below: The Pile, 8-core Apple M-series, `--threads 8`.

**Memory** — RSS growth of a bare process that only loads the encoding:

| encoder | cl100k_base | o200k_base | ids for 3 255 Common-Crawl docs |
|---|---:|---:|---:|
| **toktok** | **37.5 MiB** (14.0 live) | **72.4 MiB** (27.2 live) | 25.4 MiB |
| tiktoken | 48.3 MiB | 86.5 MiB | 212.6 MiB |
| quicktok (C++) | 50.4 MiB | 99.7 MiB | 20.3 MiB |

`enc.memory_bytes` reports the exact live table bytes (14.0 / 27.2 MiB — the rest
of RSS is construction scratch the allocator hasn't returned);
`vocab().memory_breakdown()` breaks it down per table. tiktoken's ids cost ~8×
more because a `list[int]` is 8 bytes of pointer plus a 28-byte object per token,
where `encode_to_numpy` / `encode_batch` hand back one `uint32` buffer.

**CPU** — CPU-seconds per MB encoded (single thread; lower is better):

| encoder | cl100k_base | o200k_base |
|---|---:|---:|
| **toktok (numpy)** | **0.0117** | **0.0123** |
| quicktok (C++) | 0.0118 | 0.0132 |
| toktok (`list[int]`) | 0.0136 | 0.0156 |
| tiktoken | 0.0598 | 0.0395 |

That is **3–5× less CPU for the same bytes** — the number that sets how many
cores an ingestion pipeline burns on tokenization.

**Latency** — per-document encode, 4 000 Pile documents (mean 0.3 KiB), µs:

| encoder | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|
| **toktok** | **1.8** | **10.0** | **32.5** | **103.6** | **167.9** |
| quicktok (C++) | 1.9 | 10.3 | 36.2 | 108.6 | 150.8 |
| tiktoken | 7.2 | 41.8 | 156.3 | 565.2 | 905.9 |

**p99 is ~4.8× lower than tiktoken**, and the tail is tighter still at p99.9
(5.5×) — worth more than mean throughput if tokenization sits in a request path.
On larger documents (Common Crawl, mean 7.5 KiB) the same ordering holds:
p99 1.97 ms vs tiktoken's 6.31 ms.

**Counting without ids** — `count_batch()` never materializes token ids:

| operation (4 000 Pile docs, 8 threads) | throughput | RSS growth |
|---|---:|---:|
| `count_batch()` | 426 MB/s | **0.2 MiB** |
| `encode_batch()` | 432 MB/s | 4.4 MiB |
| tiktoken `encode_ordinary_batch()` | 12.7 MB/s | 11.5 MiB |

Reproduce the whole table set, including the other corpora:

```sh
python bench/profile.py --threads 8
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
bench/               corpus fetcher, throughput benchmark (compare.py),
                     memory/CPU/latency profile (profile.py)
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
