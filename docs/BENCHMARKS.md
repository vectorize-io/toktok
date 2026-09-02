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

| encoder | MB/s | CPU-s/MB | p50 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|
| **toktok** | **89.7** | **0.0111** | **0.6 µs** | **13.4 µs** | **26.7 µs** |
| bpe-openai | 30.3 | 0.0330 | 1.9 µs | 38.4 µs | 82.0 µs |
| tiktoken-rs | 7.3 | 0.1369 | 7.9 µs | 169.9 µs | 361.3 µs |

Across both encodings and all three corpora, same-run ratios:

| vs | encode throughput | p99 latency per document |
|---|---|---|
| bpe-openai | **1.9–3.8× faster** | 1.3–20× lower |
| tiktoken-rs | **3.4–20× faster** | 3.8–27× lower |

## Against tiktoken, from Python

Measured per interpreter, because they are not interchangeable — and because the
answer to "do I need a new Python for this?" is no. The Pile, cl100k_base,
single thread (MB/s):

| interpreter | toktok (numpy) | toktok (list) | quicktok (C++) | tiktoken | speedup |
|---|---:|---:|---:|---:|---:|
| CPython 3.11 | **85.1** | 53.6 | 52.4 | 11.5 | **7.4×** |
| CPython 3.14 | **68.6** | 46.9 | 47.0 | 12.0 | **5.7×** |
| CPython 3.14t free-threaded | **94.5** | 83.6 | 77.7 | 14.0 | **6.8×** |

Each row ran on its own runner, so compare within a row: tiktoken's own figure
moves 11.5 → 14.0 across the three, which is machine variance rather than an
interpreter effect. The ratio is what holds.

Threaded, 4 vCPU (`--threads 4`), showing where free-threading does and does not
matter:

| interpreter | `batch_count` | `encode_batch` | cores used | tiktoken batch |
|---|---:|---:|---:|---:|
| CPython 3.11 | 220.3 MB/s | 188.9 | 3.58 | 6.1 |
| CPython 3.14 | 191.3 MB/s | 149.9 | 3.28 | 6.3 |
| CPython 3.14t free-threaded | 252.1 MB/s | 203.8 | 3.33 | 16.0 |

toktok already scales on a GIL build — encoding releases the GIL, so ~3.3 of 4
cores stay busy either way. The free-threaded build is not what unlocks the
speed; it mainly helps callers whose *own* Python code around the tokenizer
would otherwise serialize.

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
