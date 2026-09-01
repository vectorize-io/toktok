# Development

```sh
uv sync                       # venv + dev deps + builds the extension
uv run pytest                 # parity vs tiktoken + the Python surface
cargo test --release          # exactness fixtures, offsets, batch, invalid UTF-8

uv build                      # wheel + sdist into dist/
```

The Rust extension is compiled by [maturin](https://www.maturin.rs), declared as
the PEP 517 build backend in `pyproject.toml` — uv drives it, so `uv sync`,
`uv run` and `uv build` are the only commands you need. After changing the Rust
sources, `uv sync --reinstall-package toktok-rs` rebuilds.

## Layout

```
crates/toktok        the engine + embedded vocabularies (crates.io: toktok-rs)
crates/toktok-py     PyO3 bindings -> toktok._toktok  (PyPI: toktok-rs)
python/toktok        the Python package (batch_count)
bench/               corpus fetcher, throughput (compare.py), resource profile
                     (profile.py), Rust-vs-Rust comparison (rust/)
tools/gen_vectors.py regenerates the exactness fixtures from tiktoken
scripts/release.sh   version bump, tag and push
```

## Tests

- `crates/toktok/tests/vectors.rs` — ids checked against fixtures generated from
  tiktoken, plus offsets tiling, batch equivalence, lossless round-trips on
  invalid UTF-8, and memory accounting.
- `tests/test_parity.py` — fixed cases, a randomized fuzz, a full-Unicode fuzz,
  and all three 25 MB corpora against tiktoken.
- `tests/test_api.py` — the Python surface.
- `tests/test_freethreading.py` — the GIL stays off on 3.14t, concurrent counts
  agree, and work actually runs in parallel (CPU time vs wall time).

Regenerate the exactness fixtures after touching the vocabularies:

```sh
uv run python tools/gen_vectors.py
```

## How it's fast

Same algorithm as [`bpe`](https://github.com/github/rust-gems) (exact
backtracking BPE) — the speed is data-structure engineering, ported from
[quicktok](https://github.com/dmatth1/quicktok)'s C++:

- **2-byte trie** — the longest-match walk reads 2 input bytes per single 8-byte
  slot load, with a zero-lookup direct table for CJK characters.
- **Dense validity memos** — merge-validity checks hit exactly-keyed caches (2 MB
  for 17-bit token ids, a wider one for 200k-vocab ids). The key goes through a
  bijective mixer, so index bits plus tag bits reconstruct it exactly and a slot
  can never alias a different pair.
- **Specialized pretokenizers** — the fixed cl100k/o200k regexes are compiled by
  hand into SIMD scanners; there is no general regex engine anywhere.
- **Single-pass product machines** — for ASCII text (most of code and English),
  one loop owns both the pretokenizer's boundary rules and token emission. Any
  Unicode contact falls back to the exact scalar scanner for one piece, so output
  is byte-exact by construction.

Two things that look like dead weight but are not: the byte trie is dropped after
construction (it only seeds the other tables, and is the largest of them), and it
grows incrementally rather than being sized for the worst case — together that
took cl100k's resident tables from 51 MiB to 37.5 MiB.

## Adding an encoding

The pretokenizer is parameterized over the axes the cl100k/o200k family differs
on (`O200K_WS`, `SINGLE_DIGIT`, `CONTR`), so Llama-3, Qwen and Mistral Tekken
need vocabulary data and a scanner flag rather than new code. Qwen additionally
needs an NFC normalizer, which is not ported yet. See upstream quicktok for the
reference implementations.
