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
| **toktok** | **249.8** | **0.0040** | **0.2 µs** | **4.1 µs** | **12.5 µs** |
| bpe-openai | 36.4 | 0.0274 | 1.6 µs | 32.4 µs | 68.0 µs |
| tiktoken-rs | 9.4 | 0.1066 | 6.1 µs | 132.4 µs | 272.9 µs |

Across both encodings and all three corpora, same-run ratios:

| vs | encode throughput | p99 latency per document |
|---|---|---|
| bpe-openai | **2.7–6.9× faster** | 2.4–7.9× lower |
| tiktoken-rs | **9.1–31× faster** | 7.4–32× lower |

## Against tiktoken, from Python

Measured per interpreter, because they are not interchangeable — and because the
answer to "do I need a new Python for this?" is no. The Pile, cl100k_base,
single thread (MB/s):

| interpreter | toktok (numpy) | toktok (list) | quicktok (C++) | tiktoken | speedup |
|---|---:|---:|---:|---:|---:|
| CPython 3.11 | **211.8** | 96.5 | 63.9 | 13.8 | **15.4×** |
| CPython 3.14 | **187.0** | 83.0 | — | 11.3 | **16.6×** |
| CPython 3.14t free-threaded | **174.7** | 111.5 | — | 11.5 | **15.2×** |

Each row ran on its own runner, so compare within a row: tiktoken's own figure
moves 11.3 → 13.8 across the three, which is machine variance rather than an
interpreter effect. The ratio is what holds. quicktok has no wheel past 3.11,
hence the gaps.

Threaded, 4 vCPU (`--threads 4`), showing where free-threading does and does not
matter:

| interpreter | `batch_count` | `encode_batch` | cores used | tiktoken batch |
|---|---:|---:|---:|---:|
| CPython 3.11 | 263.5 MB/s | 151.3 | 3.21 | 8.3 |
| CPython 3.14 | 230.3 MB/s | 147.4 | 3.53 | 7.4 |
| CPython 3.14t free-threaded | 229.2 MB/s | 128.6 | 3.43 | 12.7 |

toktok already scales on a GIL build — encoding releases the GIL, so ~3.3 of 4
cores stay busy either way. The free-threaded build is not what unlocks the
speed; it mainly helps callers whose *own* Python code around the tokenizer
would otherwise serialize.

## Resource profile

`bench/profile.py` runs **each encoder in its own subprocess**, so RSS is
attributable to that encoder alone. The Pile, 4 threads.

**Memory** — RSS growth of a bare process that only loads the encoding:

| encoder | cl100k_base | o200k_base | ids for 4 000 Pile docs |
|---|---:|---:|---:|
| **toktok** | **22.6 MiB** (14.0 live) | **47.3 MiB** (31.2 live) | 0.2 MiB |
| tiktoken | 46.8 MiB | 81.4 MiB | 8.9 MiB |
| quicktok (C++) | 34.3 MiB | 66.4 MiB | 0.2 MiB |

This row is a process that only *loads* the encoding. Encoding additionally
allocates a per-thread cache of recently seen pretokens, capped at 32 MiB for
the whole process however many threads encode, and populated lazily — a caller
that tokenizes one short string never materializes it.

`Tokenizer::memory_bytes()` reports the exact live table bytes (14.0 / 31.2 MiB);
the rest of RSS is construction scratch the allocator has not returned.
`vocab().memory_breakdown()` splits it per table. tiktoken's ids cost ~45× more
because a `list[int]` is 8 bytes of pointer plus a 28-byte object per token,
where the numpy and batch paths hand back one `uint32` buffer.

**CPU** — CPU-seconds per MB encoded, single thread (lower is better):

| encoder | cl100k_base | o200k_base |
|---|---:|---:|
| **toktok** | **0.0054** | **0.0057** |
| quicktok (C++) | 0.0170 | 0.0186 |
| tiktoken | 0.0752 | 0.0517 |

9–14× less CPU for the same bytes — the number that sets how many cores an
ingestion pipeline burns on tokenization.

**Latency** — per-document encode, 4 000 Pile documents (mean 0.3 KiB), µs:

| encoder | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|
| **toktok** | **1.9** | **5.9** | **17.4** | **61.6** | **77.2** |
| quicktok (C++) | 2.5 | 13.3 | 44.3 | 129.6 | 206.8 |
| tiktoken | 9.6 | 53.3 | 187.3 | 688.1 | 1016.7 |

**Counting without ids** — what `batch_count` buys:

| operation (4 000 Pile docs, 4 threads) | throughput | RSS growth |
|---|---:|---:|
| `toktok.batch_count()` | 263.5 MB/s | **0.0 MiB** |
| toktok `encode_batch()` | 151.3 MB/s | 0.2 MiB |
| tiktoken `encode_ordinary_batch()` | 8.3 MB/s | 8.9 MiB |

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
