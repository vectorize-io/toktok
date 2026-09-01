//! toktok — a fast, exact BPE tokenizer.
//!
//! A Rust port of [quicktok](https://github.com/dmatth1/quicktok)'s C++ engine:
//! same algorithm (exact backtracking BPE), same data-structure engineering
//! (packed 2-byte trie, dense bijectively-mixed merge-validity memos, hand-
//! compiled pretokenizers, single-pass pretok+merge product machines).
//!
//! Token ids are byte-identical to tiktoken.
//!
//! ```no_run
//! let tok = toktok_core::Tokenizer::load_dir("python/toktok/data", "cl100k_base").unwrap();
//! let ids = tok.encode(b"hello world");
//! assert_eq!(tok.decode(&ids), b"hello world");
//! ```

pub mod mb;
pub mod pretok;
pub mod pretok_o200k;
mod scratch;
pub mod tokenizer;
pub mod vocab;

pub use pretok::UClass;
pub use pretok_o200k::UClassO;
pub use tokenizer::{Scanner, Tokenizer};
pub use vocab::{Vocab, VocabError, RANK_MAX};

/// Encodings this build can load from a data directory.
pub const BUILTIN_ENCODINGS: &[&str] = &["cl100k_base", "o200k_base", "o200k_harmony"];
