# Benchmarks

Throughput is only part of the story: what a tokenizer costs in production is RAM
per process, CPU per MB, and tail latency per request. All three are measured
here, and **every encoder's ids are verified identical before anything is
timed** — a row only appears if it is byte-exact.

Absolute numbers move with the machine. Ratios within a single run are the
stable signal: measuring on a laptop at load average 44 swung the same benchmark
5× between runs, which is why the headline numbers come from CI.

## Against the Rust tokenizers

`bench/rust`, no Python in the measurement. The Pile, 25 MB, cl100k_base, single
thread pinned with `taskset`, on a GitHub runner:

| encoder | MB/s | p50 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| **toktok** | **99.6** | **0.6 µs** | **11.7 µs** | **24.0 µs** |
| bpe-openai | 30.2 | 2.0 µs | 42.6 µs | 90.5 µs |
| tiktoken-rs | 7.4 | 7.7 µs | 164.3 µs | 347.8 µs |

Across both encodings and all three corpora, same-run ratios:

| vs | encode throughput | p99 latency per document |
|---|---|---|
| bpe-openai | **1.9–3.8× faster** | 1.3–20× lower |
| tiktoken-rs | **3.4–20× faster** | 3.8–27× lower |

## Against tiktoken, from Python

`bench/compare.py`, cl100k_base, three 25 MB corpora (MB/s):

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **toktok (numpy path)** | **62.4** | **105.2** | 45.0 |
| quicktok (C++, the original this is ported from) | 60.8 | 102.5 | **48.9** |
| tiktoken | 16.2 | 15.4 | 13.5 |

## Resource profile

`bench/profile.py` runs **each encoder in its own subprocess**, so RSS is
attributable to that encoder alone. The Pile, 8 threads.

**Memory** — RSS growth of a bare process that only loads the encoding:

| encoder | cl100k_base | o200k_base | ids for 3 255 Common-Crawl docs |
|---|---:|---:|---:|
| **toktok** | **37.5 MiB** (14.0 live) | **72.4 MiB** (27.2 live) | 25.4 MiB |
| tiktoken | 48.3 MiB | 86.5 MiB | 212.6 MiB |
| quicktok (C++) | 50.4 MiB | 99.7 MiB | 20.3 MiB |

`Tokenizer::memory_bytes()` reports the exact live table bytes (14.0 / 27.2 MiB);
the rest of RSS is construction scratch the allocator has not returned.
`vocab().memory_breakdown()` splits it per table. tiktoken's ids cost ~8× more
because a `list[int]` is 8 bytes of pointer plus a 28-byte object per token,
where the numpy and batch paths hand back one `uint32` buffer.

**CPU** — CPU-seconds per MB encoded, single thread (lower is better):

| encoder | cl100k_base | o200k_base |
|---|---:|---:|
| **toktok** | **0.0117** | **0.0123** |
| quicktok (C++) | 0.0118 | 0.0132 |
| tiktoken | 0.0598 | 0.0395 |

3–5× less CPU for the same bytes — the number that sets how many cores an
ingestion pipeline burns on tokenization.

**Latency** — per-document encode, 4 000 Pile documents (mean 0.3 KiB), µs:

| encoder | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|
| **toktok** | **1.8** | **10.0** | **32.5** | **103.6** | **167.9** |
| quicktok (C++) | 1.9 | 10.3 | 36.2 | 108.6 | 150.8 |
| tiktoken | 7.2 | 41.8 | 156.3 | 565.2 | 905.9 |

On larger documents (Common Crawl, mean 7.5 KiB) the ordering holds: p99 1.97 ms
vs tiktoken's 6.31 ms.

**Counting without ids** — what `batch_count` buys:

| operation (4 000 Pile docs, 8 threads) | throughput | RSS growth |
|---|---:|---:|
| `toktok.batch_count()` | 426 MB/s | **0.2 MiB** |
| toktok `encode_batch()` | 432 MB/s | 4.4 MiB |
| tiktoken `encode_ordinary_batch()` | 12.7 MB/s | 11.5 MiB |

## Reproducing

The **Benchmarks** workflow runs the whole set on a runner — manual dispatch,
weekly, and on release tags — and publishes the tables to its job summary. That
is the recommended way: it is isolated from whatever else your machine is doing,
pins single-threaded work with `taskset`, repeats the matrix to expose
run-to-run spread, and records load average before and after.

Locally:

```sh
uv sync --group bench
uv run python bench/fetch_corpus.py     # 3 × 25 MB, streamed from source

# Rust vs Rust
cargo run --release --manifest-path bench/rust/Cargo.toml -- bench/corpus/pile.txt cl100k_base

# vs tiktoken and quicktok, from Python
uv run python bench/compare.py

# memory, CPU and latency percentiles
uv run python bench/profile.py --threads 8
```

The corpora are The Pile (diverse English), GitHub code, and Common Crawl
(multilingual), each cut at 25 MB. Streaming order is fixed, so two fetches
produce identical bytes.
