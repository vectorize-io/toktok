//! Which texts NFC normalization might change.
//!
//! Qwen's tokenizer NFC-normalizes its input first, and this crate has no
//! normalizer (std doesn't ship one). Nearly all text is NFC already, so the
//! cheap thing is to find the few texts that might not be and normalize only
//! those: [`Tokenizer::unnormalized`](crate::Tokenizer::unnormalized) does
//! that over a batch, in parallel.
//!
//! A text can only change if it holds a combining mark, a character NFC
//! replaces, or one that composes with what comes before it. All of those are
//! at U+0300 or above, so a byte below 0xCC is never one of them. The table
//! (`data/nfc_unsafe.bin`, from `tools/gen_qwen35.py`) is Python's
//! `unicodedata`, the normalizer the Python package uses.

use std::sync::OnceLock;

struct Table {
    bmp: Vec<u64>, // one bit per codepoint below U+10000
    lo: Vec<u32>,
    hi: Vec<u32>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| {
        let raw = include_bytes!("../data/nfc_unsafe.bin");
        let (lo, hi) =
            crate::pretok::read_ranges(raw, &mut 0, "nfc_unsafe.bin").expect("bundled table");
        let mut bmp = vec![0u64; 1024];
        for (&a, &b) in lo.iter().zip(&hi) {
            for cp in a..=b.min(0xFFFF) {
                bmp[cp as usize >> 6] |= 1 << (cp & 63);
            }
        }
        Table { bmp, lo, hi }
    })
}

/// True if NFC might change `text`; false means it is NFC already.
pub fn maybe_unnormalized(text: &[u8]) -> bool {
    let t = table();
    let mut p = 0;
    while let Some(k) = text[p..].iter().position(|&b| b >= 0xCC) {
        let (cp, n) = crate::pretok::u8dec(text, p + k, text.len());
        let hit = if cp < 0x10000 {
            t.bmp[cp as usize >> 6] >> (cp & 63) & 1 != 0
        } else {
            let i = t.hi.partition_point(|&h| h < cp);
            i < t.lo.len() && t.lo[i] <= cp
        };
        if hit {
            return true;
        }
        p += k + n.max(1);
    }
    false
}
