# toktok

A fast, exact BPE tokenizer for OpenAI encodings — Rust core, Python bindings.
Token ids are **byte-identical to [tiktoken](https://github.com/openai/tiktoken)**.

The Pile, 25 MB, cl100k_base, single thread, GitHub runners. Ids verified
identical before timing.

#### Python 3.11

| encoder | throughput | CPU per MB | p99 per doc | |
|---|---:|---:|---:|---|
| **toktok** | **85.1 MB/s** | **0.0117 s** | **34.8 µs** | **7.4×** |
| tiktoken | 11.5 MB/s | 0.0873 s | 226.8 µs | 1× |

#### Python 3.14

| encoder | throughput | CPU per MB | p99 per doc | |
|---|---:|---:|---:|---|
| **toktok** | **68.6 MB/s** | **0.0146 s** | **37.5 µs** | **5.7×** |
| tiktoken | 12.0 MB/s | 0.0832 s | 208.6 µs | 1× |

#### Python 3.14t · free-threaded

| encoder | throughput | CPU per MB | p99 per doc | |
|---|---:|---:|---:|---|
| **toktok** | **94.5 MB/s** | **0.0106 s** | **35.1 µs** | **6.8×** |
| tiktoken | 14.0 MB/s | 0.0716 s | 188.8 µs | 1× |

#### Rust

| encoder | throughput | CPU per MB | p99 per doc | |
|---|---:|---:|---:|---|
| **toktok** | **89.7 MB/s** | **0.0111 s** | **13.4 µs** | **12.3×** |
| [bpe-openai](https://crates.io/crates/bpe-openai) | 30.3 MB/s | 0.0330 s | 38.4 µs | 4.2× |
| [tiktoken-rs](https://crates.io/crates/tiktoken-rs) | 7.3 MB/s | 0.1369 s | 169.9 µs | 1× |

<sub>Each table is one machine; the Python ones ran on separate runners, so
compare within a table. Method and full results:
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.</sub>

## Encodings

Pass an encoding name — not a model name. The three bundled encodings are
compiled into the binary, so there are no data files, downloads or runtime
dependencies.

| encoding | used by |
|---|---|
| `cl100k_base` | GPT-3.5, GPT-4, GPT-4 Turbo, `text-embedding-3-*`, `text-embedding-ada-002` |
| `o200k_base` | GPT-4o, GPT-4o mini, GPT-4.1, GPT-5, o1 / o3 / o4-mini |
| `o200k_harmony` | GPT-OSS (o200k_base's merge ranks plus the harmony special tokens) |

An unknown name raises `KeyError` listing the valid ones.

## Python

```sh
uv add toktok-rs        # or: pip install toktok-rs
```

One function. Give it your texts and an encoding, get a token count per text:

```python
import toktok

toktok.batch_count(["hello world", "how many tokens is this?"])
# [2, 6]

toktok.batch_count(docs, "o200k_base")                # GPT-4o and friends
toktok.batch_count(docs, "cl100k_base", threads=4)    # 0 = every core (default)
toktok.batch_count(["a<|endoftext|>b"], with_special=True)   # [3]
```

### `batch_count(texts, encoding="cl100k_base", threads=0, with_special=False)`

Returns `list[int]` — one count per text, in the order given.

| parameter | default | what it does |
|---|---|---|
| `texts` | — | any iterable of `str`. An empty iterable returns `[]`; an empty string counts as `0`. |
| `encoding` | `"cl100k_base"` | which tokenizer to count with — one of the three [encodings](#encodings). Model names are not accepted; look yours up in that table. An unknown name raises `KeyError`. |
| `threads` | `0` | worker threads. `0` uses every core; `1` counts on the calling thread; any other number caps the pool. Counting releases the GIL, so this scales — including on free-threaded builds. Threads are only worth it for large batches; for a handful of short strings the pool costs more than it saves. |
| `with_special` | `False` | how to treat special-token strings. `False` counts `<\|endoftext\|>` as the ordinary text it looks like (7 tokens); `True` counts it as the single special token it is (1). Use `True` when your text already contains rendered chat/control markup and you want the count the model will see. |

Ids are never built: one scratch buffer per thread is reused across texts, so a
batch of any size allocates O(threads), not O(tokens).

Wheels cover CPython **3.11–3.14** plus free-threaded **3.14t**, on Linux
(x86_64, aarch64), macOS and Windows.

### `truncate(text, max_tokens, encoding="cl100k_base")`

Cut text to a token budget. Returns `(text, total_tokens)`, where
`total_tokens` counts the whole input — truncated or not — so you can report how
much was dropped.

```python
toktok.truncate("hello world", 1)          # ('hello', 2)
toktok.truncate("hello world", 50)         # ('hello world', 2) — same object back

toktok.batch_truncate(docs, 8192, "o200k_base", threads=8)   # [(text, total), ...]
```

One pass, no ids built, nothing decoded — the cut is a byte offset into the
string you passed in. About **2× faster** than `decode(encode(text)[:n])`, and
~7× with the batch form.

It also cuts on a **character boundary**. Byte-level BPE can split one character
across tokens (`"🧠"` is three), so a token-boundary cut can leave a partial
character that decodes to `U+FFFD` — `decode(enc.encode("hello 🧠")[:2])` gives
`'hello \ufffd'`. `truncate` gives `'hello '`.

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

// truncation without building ids or decoding: a byte offset back
let t = tok.truncate(text, 8192);
let kept = &text[..t.bytes];        // always a character boundary
t.total_tokens;                     // the whole input, truncated or not
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
