//! The bundled encodings, embedded in the binary.
//!
//! `Tokenizer::builtin("cl100k_base")` needs no data directory, no download and
//! no network — the vocab, special-token and Unicode-class tables are compiled
//! in (about 4 MB of `.rodata`, shared across every process that loads them).
//!
//! Turn the `embedded-data` feature off to drop them and load from a directory
//! with [`Tokenizer::load_dir`] instead.

/// Names of the encodings this build can construct with [`Tokenizer::builtin`].
///
/// [`Tokenizer::builtin`]: crate::Tokenizer::builtin
pub const BUILTIN_ENCODINGS: &[&str] = &["cl100k_base", "o200k_base", "o200k_harmony"];

#[cfg(feature = "embedded-data")]
pub(crate) mod data {
    pub const CL100K_VOCAB: &[u8] = include_bytes!("../data/cl100k.vocab");
    pub const CL100K_SPECIAL: &[u8] = include_bytes!("../data/cl100k.special");
    pub const O200K_VOCAB: &[u8] = include_bytes!("../data/o200k.vocab");
    pub const O200K_SPECIAL: &[u8] = include_bytes!("../data/o200k.special");
    pub const O200K_HARMONY_SPECIAL: &[u8] = include_bytes!("../data/o200k_harmony.special");
    pub const UNICLASS: &[u8] = include_bytes!("../data/uniclass.bin");
    pub const UNICLASS_O200K: &[u8] = include_bytes!("../data/uniclass_o200k.bin");
}
