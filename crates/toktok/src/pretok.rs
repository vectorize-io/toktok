//! Hand-coded pretokenizer for the FIXED cl100k_base regex — the "specialize the
//! known pattern, skip the general regex engine" win. Reproduces the 8 alternatives:
//!   '(?i:[sdmt]|ll|ve|re) | [^\r\n\p{L}\p{N}]?+\p{L}++ | \p{N}{1,3}+
//!   | ?[^\s\p{L}\p{N}]++[\r\n]*+ | \s++$ | \s*[\r\n] | \s+(?!\S) | \s
//! Unicode \p{L}/\p{N}/\s come from data/uniclass.bin (exact vs the reference engine).
//!
//! Port of quicktok's `src/pretok.hpp`.

use std::io::Read;

/// length-advance over a run of ASCII letters [A-Za-z] starting at q, SIMD 16 B
/// per step. Stops at the first non-ASCII-letter byte (incl. any byte >= 0x80,
/// which the caller handles via `u8dec`). Every ISA path computes the IDENTICAL
/// predicate as the scalar tail — `(b|0x20).wrapping_sub(b'a') <= 25`.
#[inline(always)]
pub fn ascii_letter_run(t: &[u8], q: usize, len: usize) -> usize {
    let mut q = q;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        // 128-bit (SSE2): natural-text words are short (~5 B), so a 32-B stride
        // overshoots and loses to 16-B on real corpora.
        let v20 = _mm_set1_epi8(0x20u8 as i8);
        let va = _mm_set1_epi8(b'a' as i8);
        let v25 = _mm_set1_epi8(25);
        let vz = _mm_setzero_si128();
        while q + 16 <= len {
            let v = _mm_loadu_si128(t.as_ptr().add(q) as *const __m128i);
            let nl = _mm_subs_epu8(_mm_sub_epi8(_mm_or_si128(v, v20), va), v25); // 0 iff a-z
            let m = (_mm_movemask_epi8(_mm_cmpeq_epi8(nl, vz)) as u32) & 0xFFFF;
            if m != 0xFFFF {
                return q + ((!m) & 0xFFFF).trailing_zeros() as usize;
            }
            q += 16;
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while q + 16 <= len {
            let v = vld1q_u8(t.as_ptr().add(q));
            let lo = vorrq_u8(v, vdupq_n_u8(0x20)); // ASCII-lowercase fold
                                                    // (lo-'a') > 25 -> not a-z (also flags >= 0x80)
            let notlet = vcgtq_u8(vsubq_u8(lo, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            if vmaxvq_u8(notlet) != 0 {
                let mut m = [0u8; 16];
                vst1q_u8(m.as_mut_ptr(), notlet);
                for (j, &x) in m.iter().enumerate() {
                    if x != 0 {
                        return q + j;
                    }
                }
            }
            q += 16;
        }
    }
    while q < len {
        let b = unsafe { *t.get_unchecked(q) };
        if (b | 0x20).wrapping_sub(b'a') <= 25 {
            q += 1;
        } else {
            break;
        }
    }
    q
}

/// length-advance over a run of ASCII bytes in [base, base+25] from q — i.e.
/// [A-Z] (base=b'A') or [a-z] (base=b'a'). Any byte >= 0x80 stops the run.
#[inline(always)]
fn ascii_case_run(t: &[u8], q: usize, len: usize, base: u8) -> usize {
    // empty-run pre-check: the o200k cascade probes UPPER* on lowercase words (and
    // vice versa) constantly — bail on the first byte before paying for a SIMD block.
    if q >= len || unsafe { *t.get_unchecked(q) }.wrapping_sub(base) > 25 {
        return q;
    }
    let mut q = q;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let vb = _mm_set1_epi8(base as i8);
        let v25 = _mm_set1_epi8(25);
        let vz = _mm_setzero_si128();
        while q + 16 <= len {
            let v = _mm_loadu_si128(t.as_ptr().add(q) as *const __m128i);
            let nc = _mm_subs_epu8(_mm_sub_epi8(v, vb), v25); // 0 iff in [base, base+25]
            let m = (_mm_movemask_epi8(_mm_cmpeq_epi8(nc, vz)) as u32) & 0xFFFF;
            if m != 0xFFFF {
                return q + ((!m) & 0xFFFF).trailing_zeros() as usize;
            }
            q += 16;
        }
    }
    while q < len && unsafe { *t.get_unchecked(q) }.wrapping_sub(base) <= 25 {
        q += 1;
    }
    q
}

#[inline(always)]
pub fn ascii_upper_run(t: &[u8], q: usize, len: usize) -> usize {
    ascii_case_run(t, q, len, b'A')
}
#[inline(always)]
pub fn ascii_lower_run(t: &[u8], q: usize, len: usize) -> usize {
    ascii_case_run(t, q, len, b'a')
}

/// Advance over a run of ASCII whitespace (\t\n\v\f\r and space) from q, updating
/// `lastnl` to the last \r/\n position seen. Matches uniclass \s for ASCII exactly.
#[inline(always)]
pub fn ascii_ws_run(t: &[u8], q: usize, len: usize, lastnl: &mut usize) -> usize {
    let mut q = q;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let sp = _mm_set1_epi8(0x20);
        let n9 = _mm_set1_epi8(9);
        let four = _mm_set1_epi8(4);
        let z = _mm_setzero_si128();
        let lf = _mm_set1_epi8(0x0A);
        let cr = _mm_set1_epi8(0x0D);
        while q + 16 <= len {
            let v = _mm_loadu_si128(t.as_ptr().add(q) as *const __m128i);
            let ctl = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_sub_epi8(v, n9), four), z); // v in 9..13
            let isws = _mm_or_si128(_mm_cmpeq_epi8(v, sp), ctl);
            let mws = (_mm_movemask_epi8(isws) as u32) & 0xFFFF;
            let mnl = (_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(v, lf), _mm_cmpeq_epi8(v, cr)))
                as u32)
                & 0xFFFF;
            if mws != 0xFFFF {
                let stop = ((!mws) & 0xFFFF).trailing_zeros();
                let before = mnl & ((1u32 << stop) - 1);
                if before != 0 {
                    *lastnl = q + 31 - before.leading_zeros() as usize;
                }
                return q + stop as usize;
            }
            if mnl != 0 {
                *lastnl = q + 31 - mnl.leading_zeros() as usize;
            }
            q += 16;
        }
    }
    while q < len {
        let b = unsafe { *t.get_unchecked(q) };
        if b == 0x20 || b.wrapping_sub(9) <= 4 {
            if b == 0x0A || b == 0x0D {
                *lastnl = q;
            }
            q += 1;
        } else {
            break;
        }
    }
    q
}

/// Advance over a run of ASCII bytes that are NOT \s, \p{L}, or \p{N} (alt4's class).
#[inline(always)]
pub fn ascii_punct_run(t: &[u8], q: usize, len: usize) -> usize {
    let mut q = q;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let n9 = _mm_set1_epi8(9);
        let four = _mm_set1_epi8(4);
        let z = _mm_setzero_si128();
        let sp = _mm_set1_epi8(0x20);
        let v20 = _mm_set1_epi8(0x20);
        let va = _mm_set1_epi8(b'a' as i8);
        let v25 = _mm_set1_epi8(25);
        let v0 = _mm_set1_epi8(b'0' as i8);
        let v9 = _mm_set1_epi8(9);
        while q + 16 <= len {
            let v = _mm_loadu_si128(t.as_ptr().add(q) as *const __m128i);
            let ws = _mm_or_si128(
                _mm_cmpeq_epi8(v, sp),
                _mm_cmpeq_epi8(_mm_subs_epu8(_mm_sub_epi8(v, n9), four), z),
            );
            let let_ = _mm_cmpeq_epi8(
                _mm_subs_epu8(_mm_sub_epi8(_mm_or_si128(v, v20), va), v25),
                z,
            );
            let dig = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_sub_epi8(v, v0), v9), z);
            let hi = _mm_cmpgt_epi8(z, v); // v >= 0x80 (signed < 0)
            let mstop =
                (_mm_movemask_epi8(_mm_or_si128(_mm_or_si128(ws, let_), _mm_or_si128(dig, hi)))
                    as u32)
                    & 0xFFFF;
            if mstop != 0 {
                return q + mstop.trailing_zeros() as usize;
            }
            q += 16;
        }
    }
    while q < len {
        let b = unsafe { *t.get_unchecked(q) };
        if b >= 0x80 {
            break;
        }
        let l = b | 0x20;
        if b == 0x20
            || b.wrapping_sub(9) <= 4
            || l.wrapping_sub(b'a') <= 25
            || b.wrapping_sub(b'0') <= 9
        {
            break;
        }
        q += 1;
    }
    q
}

/// decode one UTF-8 codepoint at p; returns (codepoint, byte length 1..4).
#[inline(always)]
pub fn u8dec(t: &[u8], p: usize, len: usize) -> (u32, usize) {
    let c = unsafe { *t.get_unchecked(p) };
    if c < 0x80 {
        return (c as u32, 1);
    }
    unsafe {
        if c >> 5 == 0x6 && p + 1 < len {
            return (
                (((c & 0x1F) as u32) << 6) | (*t.get_unchecked(p + 1) & 0x3F) as u32,
                2,
            );
        }
        if c >> 4 == 0xE && p + 2 < len {
            return (
                (((c & 0x0F) as u32) << 12)
                    | (((*t.get_unchecked(p + 1) & 0x3F) as u32) << 6)
                    | (*t.get_unchecked(p + 2) & 0x3F) as u32,
                3,
            );
        }
        if c >> 3 == 0x1E && p + 3 < len {
            return (
                (((c & 0x07) as u32) << 18)
                    | (((*t.get_unchecked(p + 1) & 0x3F) as u32) << 12)
                    | (((*t.get_unchecked(p + 2) & 0x3F) as u32) << 6)
                    | (*t.get_unchecked(p + 3) & 0x3F) as u32,
                4,
            );
        }
    }
    (c as u32, 1) // malformed: treat as 1 byte
}

/// Unicode class table for the cl100k pattern: \p{L}, \p{N}, \s.
pub struct UClass {
    llo: Vec<u32>,
    lhi: Vec<u32>,
    nlo: Vec<u32>,
    nhi: Vec<u32>,
    slo: Vec<u32>,
    shi: Vec<u32>,
    bmp: Vec<u8>, // 65536: bits L|N|S for cp < 2^16 (O(1); covers CJK & most scripts)
}

#[inline(always)]
fn range_in(lo: &[u32], hi: &[u32], cp: u32) -> bool {
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

impl UClass {
    #[inline(always)]
    pub fn is_l(&self, cp: u32) -> bool {
        if cp < 65536 {
            unsafe { *self.bmp.get_unchecked(cp as usize) & 1 != 0 }
        } else {
            range_in(&self.llo, &self.lhi, cp)
        }
    }
    #[inline(always)]
    pub fn is_n(&self, cp: u32) -> bool {
        if cp < 65536 {
            unsafe { *self.bmp.get_unchecked(cp as usize) & 2 != 0 }
        } else {
            range_in(&self.nlo, &self.nhi, cp)
        }
    }
    #[inline(always)]
    pub fn is_s(&self, cp: u32) -> bool {
        if cp < 65536 {
            unsafe { *self.bmp.get_unchecked(cp as usize) & 4 != 0 }
        } else {
            range_in(&self.slo, &self.shi, cp)
        }
    }

    /// Exact heap footprint of the class tables, in bytes.
    pub fn memory_bytes(&self) -> usize {
        [
            &self.llo, &self.lhi, &self.nlo, &self.nhi, &self.slo, &self.shi,
        ]
        .iter()
        .map(|v| std::mem::size_of_val(&v[..]))
        .sum::<usize>()
            + self.bmp.len()
    }

    pub fn empty() -> UClass {
        UClass {
            llo: vec![],
            lhi: vec![],
            nlo: vec![],
            nhi: vec![],
            slo: vec![],
            shi: vec![],
            bmp: vec![],
        }
    }

    pub fn load(path: &std::path::Path) -> Result<UClass, crate::vocab::VocabError> {
        let mut raw = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut raw))
            .map_err(|e| {
                crate::vocab::VocabError(format!(
                    "toktok: cannot open uniclass file {}: {e}",
                    path.display()
                ))
            })?;
        Self::from_bytes(&raw, &path.display().to_string())
    }

    pub fn from_bytes(raw: &[u8], what: &str) -> Result<UClass, crate::vocab::VocabError> {
        let mut p = 0usize;
        let (llo, lhi) = read_ranges(raw, &mut p, what)?;
        let (nlo, nhi) = read_ranges(raw, &mut p, what)?;
        let (slo, shi) = read_ranges(raw, &mut p, what)?;
        let mut u = UClass {
            llo,
            lhi,
            nlo,
            nhi,
            slo,
            shi,
            bmp: vec![0u8; 65536],
        };
        mark(&mut u.bmp, &u.llo, &u.lhi, 1);
        mark(&mut u.bmp, &u.nlo, &u.nhi, 2);
        mark(&mut u.bmp, &u.slo, &u.shi, 4);
        Ok(u)
    }
}

pub(crate) fn read_ranges(
    raw: &[u8],
    p: &mut usize,
    what: &str,
) -> Result<(Vec<u32>, Vec<u32>), crate::vocab::VocabError> {
    let bad = || crate::vocab::VocabError(format!("toktok: bad uniclass file: {what}"));
    if *p + 4 > raw.len() {
        return Err(bad());
    }
    let n = u32::from_le_bytes(raw[*p..*p + 4].try_into().unwrap()) as usize;
    *p += 4;
    if n > 100_000 || *p + n * 8 > raw.len() {
        return Err(bad());
    }
    let (mut lo, mut hi) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let a = u32::from_le_bytes(raw[*p..*p + 4].try_into().unwrap());
        let b = u32::from_le_bytes(raw[*p + 4..*p + 8].try_into().unwrap());
        *p += 8;
        if a > b || b > 0x10FFFF {
            return Err(bad());
        }
        lo.push(a);
        hi.push(b);
    }
    Ok((lo, hi))
}

pub(crate) fn mark(bmp: &mut [u8], lo: &[u32], hi: &[u32], bit: u8) {
    for i in 0..lo.len() {
        if lo[i] < 65536 {
            let b = hi[i].min(65535);
            for c in lo[i]..=b {
                bmp[c as usize] |= bit;
            }
        }
    }
}

/// Find ONE pretoken starting at p; returns its byte length (no emit/advance).
///
/// Two axes parameterize the three cl100k-family grammars:
///   `O200K_WS`    : whitespace cascade is o200k-style `\s*[\r\n]+ | \s+(?!\S) | \s+`
///                   (Llama-3, Qwen) vs cl100k's `\s++$ | \s*[\r\n] | \s+(?!\S) | \s`.
///   `SINGLE_DIGIT`: number alt is `\p{N}` (Qwen) vs `\p{N}{1,3}`.
#[inline]
pub fn pretok_next_impl<const O200K_WS: bool, const SINGLE_DIGIT: bool>(
    u: &UClass,
    t: &[u8],
    p: usize,
    len: usize,
) -> usize {
    let b = t[p];
    let (cp, nb) = u8dec(t, p, len);
    // --- alt 1: '(?i:[sdmt]|ll|ve|re) ---
    if b == b'\'' {
        let lc = |x: u8| if x.is_ascii_uppercase() { x + 32 } else { x };
        if p + 1 < len {
            let c1 = lc(t[p + 1]);
            if c1 == b's' || c1 == b'd' || c1 == b'm' || c1 == b't' {
                return 2;
            }
            if p + 2 < len {
                let c2 = lc(t[p + 2]);
                if (c1 == b'l' && c2 == b'l')
                    || (c1 == b'v' && c2 == b'e')
                    || (c1 == b'r' && c2 == b'e')
                {
                    return 3;
                }
            }
        }
    }
    // --- alt 2: [^\r\n\p{L}\p{N}]?+ \p{L}++ ---
    {
        let mut q = p;
        let mut pfx = false;
        let cp_l = u.is_l(cp);
        if !cp_l && cp != 13 && cp != 10 && !u.is_n(cp) && p + nb < len {
            let (c2, _) = u8dec(t, p + nb, len);
            if u.is_l(c2) {
                pfx = true;
                q = p + nb;
            }
        }
        if pfx || cp_l {
            if !pfx {
                q = p;
            }
            loop {
                q = ascii_letter_run(t, q, len);
                if q >= len || t[q] < 0x80 {
                    break;
                }
                let (c2, n2) = u8dec(t, q, len);
                if u.is_l(c2) {
                    q += n2;
                } else {
                    break;
                }
            }
            return q - p;
        }
    }
    // --- alt 3: \p{N}{1,3}+  (Qwen: exactly one \p{N} codepoint) ---
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
    // --- alt 4: ' ?[^\s\p{L}\p{N}]++[\r\n]*+ ---
    {
        let mut q = p;
        if b == b' ' {
            q += 1;
        }
        let s4 = q;
        q = ascii_punct_run(t, q, len); // ASCII punct/symbol run (SIMD)
        while q < len {
            let (c2, n2) = u8dec(t, q, len);
            if !u.is_s(c2) && !u.is_l(c2) && !u.is_n(c2) {
                q += n2;
            } else {
                break;
            }
        }
        if q > s4 {
            while q < len && (t[q] == b'\r' || t[q] == b'\n') {
                q += 1;
            }
            return q - p;
        }
    }
    // --- whitespace alts on the maximal \s run [p, e) ---
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
        if O200K_WS {
            // Llama-3 / Qwen: \s*[\r\n]+ | \s+(?!\S) | \s+
            if lastnl != usize::MAX {
                return lastnl + 1 - p;
            }
            if e == len {
                return e - p;
            }
            if e - p > lastlen {
                return (e - lastlen) - p;
            }
            e - p
        } else {
            // cl100k: \s++$ | \s*[\r\n] | \s+(?!\S) | \s
            if e == len {
                return e - p;
            }
            if lastnl != usize::MAX {
                return lastnl + 1 - p;
            }
            if e - p > lastlen {
                return (e - lastlen) - p;
            }
            nb
        }
    }
}

#[inline]
pub fn pretok_next(u: &UClass, t: &[u8], p: usize, len: usize) -> usize {
    pretok_next_impl::<false, false>(u, t, p, len)
}

/// Iterate the cl100k pretokens of `t`, calling `cb(offset, length)` in order.
/// (Used by the pretok-only tests/benches; encode uses the fused product machine.)
pub fn pretok(u: &UClass, t: &[u8], mut cb: impl FnMut(usize, usize)) {
    let len = t.len();
    let mut p = 0usize;
    while p < len {
        // word fast path
        let b = t[p];
        let ls = if b == b' ' { p + 1 } else { p };
        if ls < len && (t[ls] | 0x20).wrapping_sub(b'a') <= 25 {
            let q = ascii_letter_run(t, ls, len);
            if q == len || t[q] < 0x80 {
                cb(p, q - p);
                p = q;
                continue;
            }
        }
        let l = pretok_next(u, t, p, len);
        cb(p, l);
        p += l;
    }
}
