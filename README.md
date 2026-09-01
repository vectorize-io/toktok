# toktok

A fast, exact BPE tokenizer for OpenAI encodings — Rust core, Python bindings.
Token ids are **byte-identical to [tiktoken](https://github.com/openai/tiktoken)**.

| encoder | | throughput | CPU per MB | p99 latency |
|---|---|---:|---:|---:|
| **toktok** | Rust | **99.6 MB/s** | **0.0117 s** | **11.7 µs** |
| [bpe-openai](https://crates.io/crates/bpe-openai) | Rust | 30.2 | — | 42.6 µs |
| [tiktoken-rs](https://crates.io/crates/tiktoken-rs) | Rust | 7.4 | — | 164.3 µs |
| [tiktoken](https://github.com/openai/tiktoken) | Python | 16.2 | 0.0598 s | 156.3 µs |

<sub>The Pile, 25 MB, cl100k_base, single thread on a GitHub runner. Every
encoder's ids are verified identical before timing. Full tables, memory figures
and method: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.</sub>

Encodings: `cl100k_base` (GPT-3.5/GPT-4), `o200k_base` (GPT-4o),
`o200k_harmony` (GPT-OSS). They are compiled into the binary — no data files, no
downloads, no runtime dependencies.

## Python

```sh
uv add toktok-rs        # or: pip install toktok-rs
```

One function. Give it your texts and an encoding, get a token count per text:

```python
import toktok

toktok.batch_count(["hello world", "how many tokens is this?"])
# [2, 6]

toktok.batch_count(docs, "gpt-4o")                    # model names work too
toktok.batch_count(docs, "cl100k_base", threads=4)    # 0 = every core (default)
toktok.batch_count(["a<|endoftext|>b"], with_special=True)   # [3]
```

### `batch_count(texts, encoding="cl100k_base", threads=0, with_special=False)`

Returns `list[int]` — one count per text, in the order given.

| parameter | default | what it does |
|---|---|---|
| `texts` | — | any iterable of `str`. An empty iterable returns `[]`; an empty string counts as `0`. |
| `encoding` | `"cl100k_base"` | which tokenizer to count with. Takes an **encoding name** — `cl100k_base`, `o200k_base`, `o200k_harmony` — or a **model name**, which resolves to that model's encoding: `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`, `o1`, `o3`, `text-embedding-3-small`, `openai/gpt-oss-20b`. An org prefix is stripped, so HF-style ids work. Anything unrecognized raises `KeyError`. |
| `threads` | `0` | worker threads. `0` uses every core; `1` counts on the calling thread; any other number caps the pool. Counting releases the GIL, so this scales — including on free-threaded builds. Threads are only worth it for large batches; for a handful of short strings the pool costs more than it saves. |
| `with_special` | `False` | how to treat special-token strings. `False` counts `<\|endoftext\|>` as the ordinary text it looks like (7 tokens); `True` counts it as the single special token it is (1). Use `True` when your text already contains rendered chat/control markup and you want the count the model will see. |

Ids are never built: one scratch buffer per thread is reused across texts, so a
batch of any size allocates O(threads), not O(tokens).

Wheels cover CPython **3.11–3.14** plus free-threaded **3.14t**, on Linux
(x86_64, aarch64), macOS and Windows.

Need ids, decoding or offsets? `toktok._encoding(name)` returns the full
tokenizer — see [docs/PYTHON.md](docs/PYTHON.md).

## Rust

```sh
cargo add toktok-rs     # the crate is toktok-rs, the library is toktok
```

```rust
let tok = toktok::Tokenizer::builtin("cl100k_base")?;   // nothing to download

let ids = tok.encode(b"hello world");
let text = tok.decode(&ids);                 // lossless, even on invalid UTF-8
let n = tok.count(b"how many tokens is this?");

// batch APIs are parallel; 0 threads means every core
let docs: Vec<&[u8]> = vec![b"first", b"second"];
let counts = tok.count_batch(&docs, 0, false);   // no ids materialized
let batches = tok.encode_batch(&docs, 0, false);

let (ids, spans) = tok.encode_with_offsets(b"per-token byte spans");
```

Every batch method takes the same two knobs as the Python API:

| argument | meaning |
|---|---|
| `threads: usize` | `0` uses every core; `1` runs on the calling thread; anything else caps the pool. |
| `with_special: bool` | `false` counts/encodes special strings as ordinary text; `true` maps them to their single special id. |

`Tokenizer` is `Send + Sync` — load it once and share it. Full API on
[docs.rs/toktok-rs](https://docs.rs/toktok-rs).

**Cargo features**

| feature | default | what it does |
|---|---|---|
| `embedded-data` | on | compiles the vocabularies into the binary (~4 MB of `.rodata`), so `Tokenizer::builtin()` needs no files at runtime. Turn it off (`default-features = false`) to drop them and load from a directory with `Tokenizer::load_dir(dir, name)` instead. |

## More

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — full results, memory and latency profile, how to reproduce
- [docs/PYTHON.md](docs/PYTHON.md) — the tokenizer behind `batch_count`
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — layout, tests, how it's fast
- [RELEASING.md](RELEASING.md) — `scripts/release.sh patch|minor|major|X.Y.Z`, and the registry setup

## License

MIT — see [LICENSE](LICENSE). Ported from
[quicktok](https://github.com/dmatth1/quicktok) (MIT); vocabulary data derives
from [tiktoken](https://github.com/openai/tiktoken) (MIT). See [NOTICE](NOTICE).
