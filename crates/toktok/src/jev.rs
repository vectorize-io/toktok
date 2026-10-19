//! Token-count **upper bound** for TypeSafe's Jev, which publishes no tokenizer.
//!
//! Probing the API (`usage.input_tokens`) shows a byte-level BPE with Qwen2's
//! pretokenizer — cl100k's grammar, but one token per digit — over NFC text, and
//! a vocab of its own. So: split the text the way Jev does, put each piece in a
//! class, and charge each class a cost fitted as an upper bound (a linear
//! program over ~1,260 probed chunks: English, code, 22 Wikipedia languages,
//! JSON/Markdown/logs/CSV/HTML, emoji and odd Unicode). Word classes are charged
//! per cl100k token of the piece, plus extra per rare (high-rank) token; scripts
//! cl100k barely knows, per character.
//!
//! ponytail: a fitted bound, ~1.2x over on average; exact needs Jev's real vocab.
//! Text is not NFC-normalized here: decomposed input only counts higher.
//!
//! ```
//! assert!(toktok::jev::count(b"hello world") >= 2); // Jev says 2
//! ```

use crate::pretok::{pretok_next_impl, u8dec};
use crate::{Tokenizer, Truncation};
use std::sync::OnceLock;

/// The piece classes, in [`features`] order.
pub const CLASSES: [&str; 31] = [
    "const",
    "lower",
    "cap",
    "upper",
    "camel",
    "latinx",
    "contr",
    "punct",
    "digit",
    "space",
    "newline",
    "run",
    "astral",
    "fullwidth",
    "cjk",
    "hangul",
    "thai",
    "indic",
    "arab",
    "heb",
    "cyr",
    "greek",
    "arm",
    "geo",
    "eth",
    "rare_lower",
    "rare_cap",
    "rare_upper",
    "rare_camel",
    "rare_latinx",
    "rare_punct",
];
const N: usize = CLASSES.len();

#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(usize)]
enum C {
    Const, Lower, Cap, Upper, Camel, Latinx, Contr, Punct, Digit, Space,
    Newline, Run, Astral, Fullwidth, Cjk, Hangul, Thai, Indic, Arab, Heb,
    Cyr, Greek, Arm, Geo, Eth,
    RareLower, RareCap, RareUpper, RareCamel, RareLatinx, RarePunct,
}

/// Cost per unit of each class, `CLASSES` order: fitted, times a 1.05 margin,
/// rounded up.
/// Cross-validated on ~1,260 real chunks: ~1 in 500 still came out under, by <8%.
#[rustfmt::skip]
const COST: [f64; N] = [
    4.313, 1.068, 1.125, 1.057, 1.329, 1.028, 0.0, 1.19, 1.044, 1.246,
    3.559, 0.259, 1.059, 1.771, 1.255, 0.889, 1.04, 0.911, 0.967, 0.854,
    0.934, 0.897, 0.919, 0.923, 1.867, 0.101, 0.0, 0.446, 4.423, 0.0,
    0.0,
];

/// (first code point, last, class) for the scripts charged per character.
#[rustfmt::skip]
const SCRIPTS: [(u32, u32, C); 17] = [
    (0x370, 0x3FF, C::Greek), (0x400, 0x52F, C::Cyr), (0x530, 0x58F, C::Arm),
    (0x590, 0x5FF, C::Heb), (0x600, 0x8FF, C::Arab), (0x900, 0xDFF, C::Indic),
    (0xE00, 0xEFF, C::Thai), (0x10A0, 0x10FF, C::Geo), (0x1100, 0x11FF, C::Hangul),
    (0x1200, 0x139F, C::Eth), (0x1F00, 0x1FFF, C::Greek), (0x2E80, 0x9FFF, C::Cjk),
    (0xAC00, 0xD7AF, C::Hangul), (0xF900, 0xFAFF, C::Cjk), (0xFB50, 0xFDFF, C::Arab),
    (0xFE70, 0xFEFF, C::Arab), (0xFF00, 0xFFEF, C::Fullwidth),
];

fn cl100k() -> &'static Tokenizer {
    static TOK: OnceLock<Tokenizer> = OnceLock::new();
    TOK.get_or_init(|| Tokenizer::builtin("cl100k_base").expect("bundled cl100k_base"))
}

/// Upper bound on Jev's token count for `text`.
pub fn count(text: &[u8]) -> u32 {
    estimate(&features(text))
}

/// [`count`] over many texts, in parallel. `threads == 0` uses every core.
pub fn count_batch(texts: &[&[u8]], threads: usize) -> Vec<u32> {
    let tok = cl100k();
    tok.par_map(texts, threads, |t, ids, workers| {
        estimate(&features_with(tok, t, ids, workers))
    })
}

fn estimate(f: &[u32; N]) -> u32 {
    let s: f64 = f.iter().zip(COST).map(|(&n, c)| n as f64 * c).sum();
    s.ceil() as u32
}

/// Per-class sizes for `text` (characters, or cl100k tokens): what the costs
/// were fitted on. Public so the fit can be redone against new probes.
pub fn features(text: &[u8]) -> [u32; N] {
    features_with(cl100k(), text, &mut Vec::new(), 1)
}

fn features_with(tok: &Tokenizer, t: &[u8], ids: &mut Vec<u32>, workers: usize) -> [u32; N] {
    let mut f = [0u32; N];
    if !t.is_empty() {
        f[C::Const as usize] = 1;
    }
    walk(tok, t, ids, workers, |_, c, n, r| {
        f[c as usize] += n;
        if let Some((rc, rn)) = r {
            f[rc as usize] += rn;
        }
        true
    });
    f
}

/// Call `on(end, class, size, rare)` for each Jev piece of `t`, in order, until
/// it returns false. `rare` is the rare-token class and count of a word piece.
fn walk(
    tok: &Tokenizer,
    t: &[u8],
    ids: &mut Vec<u32>,
    workers: usize,
    mut on: impl FnMut(usize, C, u32, Option<(C, u32)>) -> bool,
) {
    let len = t.len();
    crate::pcache::with_cache(tok.id, workers, |pc| {
        let mut p = 0;
        while p < len {
            let l = pretok_next_impl::<true, true>(&tok.u, t, p, len);
            let piece = &t[p..p + l];
            p += l;
            let (c, chars) = classify(tok, piece);
            let go = if per_char(c) {
                on(p, c, chars, None)
            } else {
                ids.clear();
                pc.emit(&tok.v, piece, ids);
                let r =
                    rare(c).map(|r| (r, ids.iter().filter(|&&id| id >= RARE_RANK).count() as u32));
                on(p, c, ids.len() as u32, r)
            };
            if !go {
                break;
            }
        }
    });
}

/// Where to cut `text` so its [`count`] fits in `max_tokens`: on a piece
/// boundary, so always on a character boundary. Same shape as
/// [`Tokenizer::truncate`]; the token figures are estimates.
pub fn truncate(text: &[u8], max_tokens: usize) -> Truncation {
    truncate_with(cl100k(), text, max_tokens, &mut Vec::new(), 1)
}

/// [`truncate`] over many texts, in parallel. `threads == 0` uses every core.
pub fn truncate_batch(texts: &[&[u8]], max_tokens: usize, threads: usize) -> Vec<Truncation> {
    let tok = cl100k();
    tok.par_map(texts, threads, |t, ids, workers| {
        truncate_with(tok, t, max_tokens, ids, workers)
    })
}

fn truncate_with(
    tok: &Tokenizer,
    t: &[u8],
    max: usize,
    ids: &mut Vec<u32>,
    workers: usize,
) -> Truncation {
    let count =
        |s: &[u8], ids: &mut Vec<u32>| estimate(&features_with(tok, s, ids, workers)) as usize;
    let total_tokens = count(t, ids);
    if total_tokens <= max {
        return Truncation {
            bytes: t.len(),
            total_tokens,
            kept_tokens: total_tokens,
        };
    }
    // Ends of the pieces whose running cost still fits, as cut candidates.
    let mut ends = vec![0usize];
    let mut cost = COST[C::Const as usize];
    walk(tok, t, ids, workers, |end, c, n, r| {
        cost +=
            COST[c as usize] * n as f64 + r.map_or(0.0, |(rc, rn)| COST[rc as usize] * rn as f64);
        let fits = cost.ceil() as usize <= max;
        if fits {
            ends.push(end);
        }
        fits
    });
    // A prefix can split its last piece differently than the whole text did
    // (`\s+(?!\S)` looks ahead), so check the real count and back off if needed.
    loop {
        let bytes = *ends.last().unwrap();
        let kept_tokens = count(&t[..bytes], ids);
        if kept_tokens <= max || bytes == 0 {
            return Truncation {
                bytes,
                total_tokens,
                kept_tokens,
            };
        }
        ends.pop();
    }
}

/// cl100k tokens ranked past this are rare words Jev's smaller vocab tends to
/// split: counting them per word class tightens the bound on rare-word text.
const RARE_RANK: u32 = 32000;

fn rare(c: C) -> Option<C> {
    Some(match c {
        C::Lower => C::RareLower,
        C::Cap => C::RareCap,
        C::Upper => C::RareUpper,
        C::Camel => C::RareCamel,
        C::Latinx => C::RareLatinx,
        C::Punct => C::RarePunct,
        _ => return None,
    })
}

fn per_char(c: C) -> bool {
    !matches!(
        c,
        C::Lower
            | C::Cap
            | C::Upper
            | C::Camel
            | C::Latinx
            | C::Contr
            | C::Punct
            | C::Space
            | C::Newline
            | C::Astral
    )
}

/// The piece's class and its length in code points.
fn classify(tok: &Tokenizer, piece: &[u8]) -> (C, u32) {
    if piece.is_ascii() {
        return (classify_ascii(piece), piece.len() as u32);
    }
    classify_unicode(tok, piece)
}

fn classify_unicode(tok: &Tokenizer, piece: &[u8]) -> (C, u32) {
    let u = &tok.u;
    let len = piece.len();
    let (mut chars, mut q) = (0u32, 0);
    let (mut all_s, mut nl, mut astral, mut any_l, mut ascii) = (true, false, false, false, true);
    let mut script = None;
    // letter-run and case state, over letters only
    let (mut prev, mut run, mut long_run) = (u32::MAX, 0u32, false);
    let (mut letters, mut uppers, mut lowers, mut first_upper, mut later_upper) =
        (0u32, 0u32, 0u32, false, false);
    while q < len {
        let (cp, nb) = u8dec(piece, q, len);
        q += nb;
        chars += 1;
        let is_s = u.is_s(cp);
        all_s &= is_s;
        nl |= cp == 10 || cp == 13;
        astral |= cp > 0xFFFF;
        ascii &= cp < 0x80;
        if script.is_none() && cp > 0x36F {
            script = SCRIPTS.iter().find(|s| s.0 <= cp && cp <= s.1).map(|s| s.2);
        }
        if u.is_l(cp) {
            any_l = true;
            run = if cp == prev { run + 1 } else { 1 };
            long_run |= run >= 9;
            prev = cp;
            let up = (b'A' as u32..=b'Z' as u32).contains(&cp);
            let lo = (b'a' as u32..=b'z' as u32).contains(&cp);
            if letters == 0 {
                first_upper = up;
            } else {
                later_upper |= up;
            }
            uppers += up as u32;
            lowers += lo as u32;
            letters += 1;
        } else {
            prev = u32::MAX;
        }
    }
    let c = if all_s {
        if nl {
            C::Newline
        } else {
            C::Space
        }
    } else if chars == 1 && u.is_n(u8dec(piece, 0, len).0) {
        C::Digit
    } else if astral {
        C::Astral // emoji, rare CJK, math letters
    } else if !any_l {
        C::Punct
    } else if piece[0] == b'\'' && chars <= 3 {
        C::Contr
    } else if let Some(s) = script {
        s
    } else if long_run {
        C::Run
    } else if !ascii {
        C::Latinx
    } else if uppers == 0 {
        C::Lower
    } else if lowers == 0 {
        if letters > 1 {
            C::Upper
        } else {
            C::Cap
        }
    } else if first_upper && !later_upper {
        C::Cap
    } else {
        C::Camel
    };
    (c, chars)
}

/// `classify` for an all-ASCII piece, on bytes: the common case.
fn classify_ascii(p: &[u8]) -> C {
    let ws = |b: u8| matches!(b, b' ' | b'\t'..=b'\r');
    if p.iter().all(|&b| ws(b)) {
        return if p.iter().any(|&b| b == b'\n' || b == b'\r') {
            C::Newline
        } else {
            C::Space
        };
    }
    if p.len() == 1 && p[0].is_ascii_digit() {
        return C::Digit;
    }
    let letters = p.iter().filter(|b| b.is_ascii_alphabetic());
    let Some(&first) = letters.clone().next() else {
        return C::Punct;
    };
    if p[0] == b'\'' && p.len() <= 3 {
        return C::Contr;
    }
    if p.windows(9)
        .any(|w| w[0].is_ascii_alphabetic() && w.iter().all(|&b| b == w[0]))
    {
        return C::Run;
    }
    let uppers = letters.clone().filter(|b| b.is_ascii_uppercase()).count();
    if uppers == 0 {
        C::Lower
    } else if uppers == letters.clone().count() {
        if uppers > 1 {
            C::Upper
        } else {
            C::Cap
        }
    } else if first.is_ascii_uppercase() && uppers == 1 {
        C::Cap
    } else {
        C::Camel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_fast_path_matches_unicode_path() {
        let tok = cl100k();
        let text: &[u8] =
            b"Hello world, getValue HTTPServer I'm it'S 42 x=1;\n\t\tfoo_bar __init__ \
            aaaaaaaaaaaa zzzzzzzzz A B ... --- \r\n   (x) 'll 'LL #include $x @user \x0b\x0c";
        let mut p = 0;
        while p < text.len() {
            let l = pretok_next_impl::<true, true>(&tok.u, text, p, text.len());
            let piece = &text[p..p + l];
            assert_eq!(
                classify_ascii(piece),
                classify_unicode(tok, piece).0,
                "{piece:?}"
            );
            p += l;
        }
    }
}
