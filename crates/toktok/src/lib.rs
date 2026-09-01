//! **toktok** — a fast, exact BPE tokenizer.
//!
//! Token ids are byte-identical to [tiktoken](https://github.com/openai/tiktoken),
//! and encoding runs ~3.5x faster than [`bpe-openai`](https://crates.io/crates/bpe-openai)
//! and ~10x faster than [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs)
//! (see the repo's `bench/rust`). The bundled encodings are embedded in the
//! binary, so there is nothing to download or ship alongside it.
//!
//! ```
//! let tok = toktok::Tokenizer::builtin("cl100k_base")?;
//!
//! let ids = tok.encode("Hello, toktok! 日本語 🚀".as_bytes());
//! assert_eq!(tok.decode(&ids), "Hello, toktok! 日本語 🚀".as_bytes());
//!
//! assert_eq!(tok.count(b"how many tokens is this?"), 6);
//! # Ok::<(), toktok::VocabError>(())
//! ```
//!
//! One tokenizer is safe to share across threads — load it once:
//!
//! ```
//! # let tok = toktok::Tokenizer::builtin("o200k_base")?;
//! let docs: Vec<&[u8]> = vec![b"first", b"second", b"third"];
//! let counts = tok.count_batch(&docs, 0, false);   // 0 threads = every core
//! let ids = tok.encode_batch(&docs, 0, false);
//! # assert_eq!(counts.len(), 3); assert_eq!(ids.len(), 3);
//! # Ok::<(), toktok::VocabError>(())
//! ```
//!
//! # How it's fast
//!
//! Same algorithm as `bpe-openai` (exact backtracking BPE); the speed comes from
//! data-structure engineering ported from
//! [quicktok](https://github.com/dmatth1/quicktok)'s C++: a 2-byte-radix trie
//! whose walk consumes two input bytes per single 8-byte load, dense
//! bijectively-mixed merge-validity memos, hand-compiled SIMD pretokenizers
//! instead of a regex engine, and a single-pass machine that fuses
//! pretokenization with token emission for ASCII text.

mod builtin;
pub mod mb;
pub mod pretok;
pub mod pretok_o200k;
mod scratch;
pub mod tokenizer;
pub mod vocab;

pub use builtin::BUILTIN_ENCODINGS;
pub use pretok::UClass;
pub use pretok_o200k::UClassO;
pub use tokenizer::{Scanner, Tokenizer};
pub use vocab::{Vocab, VocabError, RANK_MAX};
