//! Hand-coded pretokenizer for the FIXED o200k_base regex (GPT-4o). Reproduces the
//! 7 alternatives (ordered, first-match, greedy):
//!   1: [^\r\n\p{L}\p{N}]? [UPPER]* [LOWER]+ (?i:'s|'t|'re|'ve|'m|'ll|'d)?
//!   2: [^\r\n\p{L}\p{N}]? [UPPER]+ [LOWER]* (?i:'s|'t|'re|'ve|'m|'ll|'d)?
//!   3: \p{N}{1,3}
//!   4:  ?[^\s\p{L}\p{N}]+ [\r\n/]*
//!   5: \s*[\r\n]+      6: \s+(?!\S)      7: \s+
//! where UPPER=[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}], LOWER=[\p{Ll}\p{Lm}\p{Lo}\p{M}].
//!
//! Port of quicktok's `src/pretok_o200k.hpp`.

use crate::pretok::{
    ascii_lower_run, ascii_punct_run, ascii_upper_run, ascii_ws_run, mark, read_ranges, u8dec,
};
use crate::vocab::VocabError;
use std::io::Read;

/// 5 classes: 0=L 1=N 2=S 3=UPPER 4=LOWER
pub struct UClassO {
    lo: [Vec<u32>; 5],
    hi: [Vec<u32>; 5],
    bmp: Vec<u8>, // 65536, bit c = class c
}

#[inline(always)]
fn rin(lo: &[u32], hi: &[u32], cp: u32) -> bool {
    let (mut a, mut b) = (0isize, lo.len() as isize - 1);
    while a <= b {
        let m = ((a + b) >> 1) as usize;
        if cp < lo[m] {
            b = m as isize - 1;
        } else if cp > hi[m] {
            a = m as isize + 1;
        } else {
            return true;
        }
    }
    false
}

impl UClassO {
    #[inline(always)]
    fn cls(&self, c: usize, cp: u32) -> bool {
        if cp < 65536 {
            unsafe { (*self.bmp.get_unchecked(cp as usize) >> c) & 1 != 0 }
        } else {
            rin(&self.lo[c], &self.hi[c], cp)
        }
    }
    #[inline(always)]
    pub fn is_l(&self, cp: u32) -> bool {
        self.cls(0, cp)
    }
    #[inline(always)]
    pub fn is_n(&self, cp: u32) -> bool {
        self.cls(1, cp)
    }
    #[inline(always)]
    pub fn is_s(&self, cp: u32) -> bool {
        self.cls(2, cp)
    }
    #[inline(always)]
    pub fn is_u(&self, cp: u32) -> bool {
        self.cls(3, cp)
    }
    #[inline(always)]
    pub fn is_lo(&self, cp: u32) -> bool {
        self.cls(4, cp)
    }

    pub fn empty() -> UClassO {
        UClassO {
            lo: Default::default(),
            hi: Default::default(),
            bmp: Vec::new(),
        }
    }

    pub fn load(path: &std::path::Path) -> Result<UClassO, VocabError> {
        let mut raw = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut raw))
            .map_err(|e| {
                VocabError(format!(
                    "toktok: cannot open uniclass file {}: {e}",
                    path.display()
                ))
            })?;
        Self::from_bytes(&raw, &path.display().to_string())
    }

    pub fn from_bytes(raw: &[u8], what: &str) -> Result<UClassO, VocabError> {
        let mut u = UClassO {
            lo: Default::default(),
            hi: Default::default(),
            bmp: vec![0u8; 65536],
        };
        let mut p = 0usize;
        for c in 0..5 {
            let (lo, hi) = read_ranges(raw, &mut p, what)?;
            mark(&mut u.bmp, &lo, &hi, 1u8 << c);
            u.lo[c] = lo;
            u.hi[c] = hi;
        }
        Ok(u)
    }
}

/// alt1: [UPPER]*[LOWER]+ at `start` -> end, or 0. Greedy U* then L+.
/// For cp < 0x80, UPPER == [A-Z] and LOWER == [a-z] exactly (ASCII has no
/// Lt/Lm/Lo/M), so the SIMD case runs are the same predicate as the scalar loop.
#[inline]
pub fn o_match_ul(u: &UClassO, t: &[u8], start: usize, len: usize) -> usize {
    let mut i = start;
    let mut sawmb = false;
    loop {
        // UPPER* -> [start, i)
        i = ascii_upper_run(t, i, len);
        if i >= len || t[i] < 0x80 {
            break;
        }
        let (cp, n) = u8dec(t, i, len);
        if u.is_u(cp) {
            i += n;
            sawmb = true;
        } else {
            break;
        }
    }
    let mut j = i;
    let mut any = false;
    loop {
        // LOWER+
        let j2 = ascii_lower_run(t, j, len);
        if j2 > j {
            j = j2;
            any = true;
        }
        if j >= len || t[j] < 0x80 {
            break;
        }
        let (cp, n) = u8dec(t, j, len);
        if u.is_lo(cp) {
            j += n;
            any = true;
        } else {
            break;
        }
    }
    if any {
        return j;
    }
    // LOWER+ empty: backtrack the greedy UPPER* to the LAST LOWER-eligible (BOTH)
    // char in [start, i) — it becomes the trailing LOWER+. (e.g. "亚洲AV": UPPER*
    // grabbed 亚洲AV, but 亚洲 are Lo/BOTH and AV are Lu/UPPER-only, so alt1 matches
    // "亚洲".) A pure-ASCII UPPER* run has no LOWER-eligible char: skip the scan.
    if !sawmb {
        return 0;
    }
    let mut e = 0usize;
    let mut p = start;
    while p < i {
        let (cp, n) = u8dec(t, p, len);
        if u.is_lo(cp) {
            e = p + n;
        }
        p += n;
    }
    e
}

/// alt2: [UPPER]+[LOWER]* at `start` -> end, or 0.
#[inline]
pub fn o_match_upl(u: &UClassO, t: &[u8], start: usize, len: usize) -> usize {
    let mut i = start;
    loop {
        i = ascii_upper_run(t, i, len);
        if i >= len || t[i] < 0x80 {
            break;
        }
        let (cp, n) = u8dec(t, i, len);
        if u.is_u(cp) {
            i += n;
        } else {
            break;
        }
    }
    if i == start {
        return 0;
    }
    let mut j = i;
    loop {
        j = ascii_lower_run(t, j, len);
        if j >= len || t[j] < 0x80 {
            break;
        }
        let (cp, n) = u8dec(t, j, len);
        if u.is_lo(cp) {
            j += n;
        } else {
            break;
        }
    }
    j
}

/// (?i:'s|'t|'re|'ve|'m|'ll|'d)? suffix after a letter match ending at `e`.
#[inline(always)]
pub fn o_contraction(t: &[u8], e: usize, len: usize) -> usize {
    if e >= len || t[e] != b'\'' {
        return e;
    }
    let lc = |x: u8| if x.is_ascii_uppercase() { x + 32 } else { x };
    if e + 1 >= len {
        return e;
    }
    let c1 = lc(t[e + 1]);
    if c1 == b's' || c1 == b't' || c1 == b'm' || c1 == b'd' {
        return e + 2;
    }
    if e + 2 < len {
        let c2 = lc(t[e + 2]);
        if (c1 == b'r' && c2 == b'e') || (c1 == b'v' && c2 == b'e') || (c1 == b'l' && c2 == b'l') {
            return e + 3;
        }
    }
    e
}

/// Find ONE o200k-family pretoken starting at p; returns its byte length.
///   `CONTR`       : alts 1 & 2 carry the contraction suffix (o200k: yes; Tekken: no)
///   `SINGLE_DIGIT`: alt 3 is \p{N} (Tekken) vs \p{N}{1,3} (o200k)
#[inline]
pub fn pretok_next_o200k_impl<const CONTR: bool, const SINGLE_DIGIT: bool>(
    u: &UClassO,
    t: &[u8],
    p: usize,
    len: usize,
) -> usize {
    let b = t[p];
    let (cp, nb) = u8dec(t, p, len);
    // --- alts 1 & 2 (letters), each tried prefix-consumed-first then prefix-empty ---
    let prefelig = cp != 13 && cp != 10 && !u.is_l(cp) && !u.is_n(cp);
    for alt in 0..2 {
        let mut e = 0usize;
        if prefelig {
            e = if alt == 0 {
                o_match_ul(u, t, p + nb, len)
            } else {
                o_match_upl(u, t, p + nb, len)
            };
        }
        if e == 0 {
            e = if alt == 0 {
                o_match_ul(u, t, p, len)
            } else {
                o_match_upl(u, t, p, len)
            };
        }
        if e > 0 {
            let e = if CONTR { o_contraction(t, e, len) } else { e };
            return e - p;
        }
    }
    // --- alt 3: \p{N}{1,3} ---
    if u.is_n(cp) {
        if SINGLE_DIGIT {
            return nb;
        }
        let (mut q, mut cnt) = (p, 0);
        while q < len && cnt < 3 {
            let (c2, n2) = u8dec(t, q, len);
            if u.is_n(c2) {
                q += n2;
                cnt += 1;
            } else {
                break;
            }
        }
        return q - p;
    }
    // --- alt 4:  ?[^\s\p{L}\p{N}]+[\r\n/]* ---
    {
        let mut q = p;
        if b == b' ' {
            q += 1;
        }
        let s4 = q;
        q = ascii_punct_run(t, q, len);
        while q < len {
            let (c2, n2) = u8dec(t, q, len);
            if !u.is_s(c2) && !u.is_l(c2) && !u.is_n(c2) {
                q += n2;
            } else {
                break;
            }
        }
        if q > s4 {
            while q < len && (t[q] == b'\r' || t[q] == b'\n' || t[q] == b'/') {
                q += 1;
            }
            return q - p;
        }
    }
    // --- whitespace alts (5, 6, 7) on the maximal \s run [p, e) ---
    {
        let mut e = p;
        let mut lastnl = usize::MAX;
        let mut lastlen = 1usize;
        let e0 = ascii_ws_run(t, e, len, &mut lastnl);
        if e0 > e {
            lastlen = 1;
            e = e0;
        }
        while e < len {
            let (c2, n2) = u8dec(t, e, len);
            if !u.is_s(c2) {
                break;
            }
            if c2 == 13 || c2 == 10 {
                lastnl = e;
            }
            lastlen = n2;
            e += n2;
        }
        if e == p {
            return 1;
        }
        if lastnl != usize::MAX {
            return lastnl + 1 - p; // alt5 \s*[\r\n]+
        }
        if e == len {
            return e - p; // alt6 \s+(?!\S), run to EOF
        }
        if e - p > lastlen {
            return (e - lastlen) - p; // alt6, all but the last ws char
        }
        e - p // alt7 \s+
    }
}

#[inline]
pub fn pretok_next_o200k(u: &UClassO, t: &[u8], p: usize, len: usize) -> usize {
    pretok_next_o200k_impl::<true, false>(u, t, p, len)
}
