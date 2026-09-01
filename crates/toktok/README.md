# toktok-rs

A fast, exact BPE tokenizer for OpenAI encodings. Token ids are **byte-identical
to [tiktoken](https://github.com/openai/tiktoken)**, and encoding is
**2–3.8× faster than [`bpe-openai`](https://crates.io/crates/bpe-openai)** and
**3.4–20× faster than [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs)**.

- **Zero dependencies**, and the vocabularies are embedded in the binary — nothing to download or ship alongside your program.
- `cl100k_base` (GPT-3.5/GPT-4), `o200k_base` (GPT-4o), `o200k_harmony` (GPT-OSS).
- Thread-safe: load once, encode from every core.

```toml
[dependencies]
toktok-rs = "0.1"
```

```rust
let tok = toktok::Tokenizer::builtin("cl100k_base")?;

let ids = tok.encode(b"hello world");
assert_eq!(tok.decode(&ids), b"hello world");
assert_eq!(tok.count(b"how many tokens is this?"), 6);

// counting a batch never materializes ids: O(threads) allocation, not O(tokens)
let docs: Vec<&[u8]> = vec![b"first", b"second"];
let counts = tok.count_batch(&docs, 0, false);   // 0 threads = every core
# Ok::<(), toktok::VocabError>(())
```

Note the crate is `toktok-rs` but the library is `toktok` — `use toktok::…`.

## Benchmarks

Single thread, three 25 MB corpora (The Pile, GitHub code, Common Crawl), both
encodings, Apple M-series. Every encoder's ids are verified identical before
timing. Speedup is measured **within a run** — absolute MB/s moves with machine
load, the ratio does not. Full harness:
[`bench/rust`](https://github.com/vectorize-io/toktok/tree/main/bench/rust).

| vs | encode throughput | p99 latency per document |
|---|---|---|
| [bpe-openai](https://crates.io/crates/bpe-openai) | **1.9–3.8× faster** | 1.3–20× lower |
| [tiktoken-rs](https://crates.io/crates/tiktoken-rs) | **3.4–20× faster** | 3.8–27× lower |

Full tables, memory and latency profile: [docs/BENCHMARKS.md](https://github.com/vectorize-io/toktok/blob/main/docs/BENCHMARKS.md). Reproduce:

```sh
cargo run --release --manifest-path bench/rust/Cargo.toml -- bench/corpus/pile.txt cl100k_base
```

Same algorithm as `bpe-openai` (exact backtracking BPE); the speed is
data-structure engineering ported from
[quicktok](https://github.com/dmatth1/quicktok)'s C++ — a 2-byte-radix trie whose
walk consumes two input bytes per single 8-byte load, dense bijectively-mixed
merge-validity memos, hand-compiled SIMD pretokenizers instead of a regex engine,
and a single-pass machine fusing pretokenization with token emission for ASCII.

## Features

`embedded-data` (default) compiles the vocabularies in (~4 MB of `.rodata`).
Turn it off and load them from a directory instead:

```rust
let tok = toktok::Tokenizer::load_dir("data", "cl100k_base")?;
# Ok::<(), toktok::VocabError>(())
```

## License

MIT. See [NOTICE](https://github.com/vectorize-io/toktok/blob/main/NOTICE) for
upstream attributions (quicktok, MIT; vocabulary data derived from tiktoken, MIT).
