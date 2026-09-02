//! The public tokenizer: encoding registry, the fused pretok+merge product
//! machines, specials, decode and batch encode.
//!
//! Port of quicktok's `src/quicktok.cpp`.

use crate::mb::encode_mb;
use crate::pretok::{self, UClass};
use crate::pretok_o200k::{self as pk8, UClassO};
use crate::vocab::{Vocab, VocabError, RANK_MAX};
use std::path::{Path, PathBuf};

/// What [`Tokenizer::truncate`] found: where to cut, and what it cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Truncation {
    /// Byte offset to cut the input at — always on a character boundary.
    /// Equal to the input length when nothing needed truncating.
    pub bytes: usize,
    /// Tokens in the whole input, whether or not it was truncated. This is the
    /// number to report when you want to say how much was dropped.
    pub total_tokens: usize,
    /// Tokens in the kept prefix. Equals `total_tokens` when nothing was cut,
    /// and can be one less than `max_tokens` when the cut landed inside a
    /// character.
    pub kept_tokens: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scanner {
    /// cl100k_base grammar (GPT-3.5 / GPT-4)
    Cl100k,
    /// o200k_base grammar (GPT-4o, GPT-OSS harmony)
    O200k,
}

pub struct Tokenizer {
    pub(crate) v: Vocab,
    name: String,
    scanner: Scanner,
    u: UClass,                    // cl100k-pattern classes (L/N/S)
    uo: UClassO,                  // o200k-pattern classes (L/N/S/UPPER/LOWER)
    specials: Vec<(String, u32)>, // sorted by id
}

fn load_specials(path: &Path) -> Result<Vec<(String, u32)>, VocabError> {
    let raw = match std::fs::read(path) {
        Ok(r) => r,
        Err(_) => return Ok(Vec::new()), // specials file is optional
    };
    specials_from_bytes(&raw, &path.display().to_string())
}

fn specials_from_bytes(raw: &[u8], what: &str) -> Result<Vec<(String, u32)>, VocabError> {
    let bad = || VocabError(format!("toktok: bad special-tokens file: {what}"));
    if raw.len() < 4 {
        return Err(bad());
    }
    let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    if n > 4096 {
        return Err(bad()); // o200k_harmony ships 1091 specials
    }
    let mut p = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if p + 6 > raw.len() {
            return Err(bad());
        }
        let id = u32::from_le_bytes(raw[p..p + 4].try_into().unwrap());
        let len = u16::from_le_bytes(raw[p + 4..p + 6].try_into().unwrap()) as usize;
        p += 6;
        if len == 0 || len > 64 || p + len > raw.len() {
            return Err(bad());
        }
        let s = String::from_utf8(raw[p..p + len].to_vec()).map_err(|_| bad())?;
        p += len;
        out.push((s, id));
    }
    Ok(out)
}

impl Tokenizer {
    /// Construct a bundled encoding from the tables embedded in the binary — no
    /// data directory, no download, no network.
    ///
    /// Names: `cl100k_base` (GPT-3.5/GPT-4), `o200k_base` (GPT-4o), and
    /// `o200k_harmony` (GPT-OSS). Requires the default `embedded-data` feature.
    ///
    /// ```
    /// let tok = toktok::Tokenizer::builtin("cl100k_base").unwrap();
    /// assert_eq!(tok.encode(b"hello world"), [15339, 1917]);
    /// ```
    #[cfg(feature = "embedded-data")]
    pub fn builtin(encoding: &str) -> Result<Tokenizer, VocabError> {
        use crate::builtin::data;
        let (vocab, specials, scanner) = match encoding {
            "cl100k_base" => (data::CL100K_VOCAB, data::CL100K_SPECIAL, Scanner::Cl100k),
            "o200k_base" => (data::O200K_VOCAB, data::O200K_SPECIAL, Scanner::O200k),
            // GPT-OSS harmony: identical pattern AND merge ranks to o200k_base, so
            // it reuses o200k's vocab and scanner — only the specials differ.
            "o200k_harmony" => (
                data::O200K_VOCAB,
                data::O200K_HARMONY_SPECIAL,
                Scanner::O200k,
            ),
            _ => return Err(unknown_encoding(encoding)),
        };
        let (u, uo) = match scanner {
            Scanner::Cl100k => (
                UClass::from_bytes(data::UNICLASS, "uniclass.bin")?,
                UClassO::empty(),
            ),
            Scanner::O200k => (
                UClass::empty(),
                UClassO::from_bytes(data::UNICLASS_O200K, "uniclass_o200k.bin")?,
            ),
        };
        Ok(Tokenizer {
            v: Vocab::from_bytes(vocab, encoding)?,
            name: encoding.to_string(),
            scanner,
            u,
            uo,
            specials: specials_from_bytes(specials, encoding)?,
        })
    }

    /// Load a bundled encoding from a data directory.
    ///
    /// Built-in: `cl100k_base`, `o200k_base`, `o200k_harmony`.
    pub fn load_dir(dir: impl AsRef<Path>, encoding: &str) -> Result<Tokenizer, VocabError> {
        let dir: PathBuf = dir.as_ref().to_path_buf();
        let (vocab_file, uni_o200k, specials_file, scanner) = match encoding {
            "cl100k_base" => ("cl100k.vocab", false, "cl100k.special", Scanner::Cl100k),
            "o200k_base" => ("o200k.vocab", true, "o200k.special", Scanner::O200k),
            // GPT-OSS harmony: identical pattern AND merge ranks to o200k_base (so it
            // reuses o200k.vocab and the o200k scanner) — only the specials differ.
            "o200k_harmony" => ("o200k.vocab", true, "o200k_harmony.special", Scanner::O200k),
            _ => return Err(unknown_encoding(encoding)),
        };
        let v = Vocab::load(&dir.join(vocab_file))?;
        let (u, uo) = if uni_o200k {
            (
                UClass::empty(),
                UClassO::load(&dir.join("uniclass_o200k.bin"))?,
            )
        } else {
            (UClass::load(&dir.join("uniclass.bin"))?, UClassO::empty())
        };
        Ok(Tokenizer {
            v,
            name: encoding.to_string(),
            scanner,
            u,
            uo,
            specials: load_specials(&dir.join(specials_file))?,
        })
    }

    /// The underlying vocab tables (for `memory_breakdown` and diagnostics).
    pub fn vocab(&self) -> &Vocab {
        &self.v
    }

    pub fn encoding(&self) -> &str {
        &self.name
    }
    pub fn vocab_size(&self) -> usize {
        self.v.size()
    }
    pub fn special_tokens(&self) -> &[(String, u32)] {
        &self.specials
    }

    /// tiktoken semantics: max token id + 1, specials included (cl100k -> 100277).
    pub fn n_vocab(&self) -> usize {
        let mut hi = self.v.size();
        for (_, sid) in &self.specials {
            if *sid as usize + 1 > hi {
                hi = *sid as usize + 1;
            }
        }
        hi
    }

    /// Exact heap footprint of this loaded encoding, in bytes (vocab tables plus
    /// the Unicode class tables and specials).
    pub fn memory_bytes(&self) -> usize {
        self.v.memory_bytes()
            + self.u.memory_bytes()
            + self.uo.memory_bytes()
            + self
                .specials
                .iter()
                .map(|(s, _)| s.len() + std::mem::size_of::<(String, u32)>())
                .sum::<usize>()
    }

    pub fn known_id(&self, id: u32) -> bool {
        id < self.v.n || self.specials.iter().any(|(_, sid)| *sid == id)
    }

    /// Exact token bytes -> id (base vocab first, then specials); -1 if neither.
    pub fn token_id(&self, b: &[u8]) -> i64 {
        let r = self.v.find_id(b);
        if r != RANK_MAX {
            return r as i64;
        }
        for (s, sid) in &self.specials {
            if s.as_bytes() == b {
                return *sid as i64;
            }
        }
        -1
    }

    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        if id < self.v.n {
            Some(self.v.token_bytes(id))
        } else {
            None
        }
    }

    /// Ordinary encode (no special-token handling) — tiktoken's `encode_ordinary`.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 8);
        self.encode_into(text, &mut out);
        out
    }

    pub fn encode_into(&self, text: &[u8], out: &mut Vec<u32>) {
        match self.scanner {
            Scanner::Cl100k => self.encode_core_cl100k(text, out),
            Scanner::O200k => self.encode_core_o200k(text, out),
        }
    }

    pub fn count(&self, text: &[u8]) -> usize {
        let mut out = Vec::with_capacity(text.len() / 3 + 8);
        self.encode_into(text, &mut out);
        out.len()
    }

    /// Where to cut `text` so it fits in `max_tokens`, and what it cost.
    ///
    /// See [`Tokenizer::truncate`].
    pub fn truncate(&self, text: &[u8], max_tokens: usize) -> Truncation {
        crate::scratch::with_ids(|ids| {
            ids.clear();
            self.encode_into(text, ids);
            let total_tokens = ids.len();
            if total_tokens <= max_tokens {
                return Truncation {
                    bytes: text.len(),
                    total_tokens,
                    kept_tokens: total_tokens,
                };
            }
            // Token bounds tile the input exactly, so the byte offset after
            // `max_tokens` tokens is the same string `decode(ids[..max_tokens])`
            // would produce — without building either the ids or the string.
            let mut bytes = 0usize;
            for &id in &ids[..max_tokens] {
                bytes += self.v.token_len(id) as usize;
            }
            // A token can end mid-character: byte-level BPE splits rare
            // characters across tokens, so `🚀` (F0 9F 9A 80) can arrive as
            // ` \xf0\x9f` + `\x9a` + `\x80`. Cutting there is what leaves the
            // familiar U+FFFD at the tail of `decode(encode(x)[:n])`. Back off to
            // the character boundary instead — and only that far, since the
            // straddling token can carry text before the partial character (that
            // ` ` in the example) which is worth keeping.
            while bytes > 0 && (text[bytes] & 0xC0) == 0x80 {
                bytes -= 1;
            }
            // tokens wholly inside the kept bytes, for reporting
            let (mut off, mut kept_tokens) = (0usize, 0usize);
            for &id in &ids[..max_tokens] {
                let end = off + self.v.token_len(id) as usize;
                if end > bytes {
                    break;
                }
                off = end;
                kept_tokens += 1;
            }
            Truncation {
                bytes,
                total_tokens,
                kept_tokens,
            }
        })
    }

    /// Ordinary ids + the exclusive byte bound of each token (tiles the input).
    pub fn encode_with_offsets(&self, text: &[u8]) -> (Vec<u32>, Vec<u32>) {
        let ids = self.encode(text);
        let mut bounds = Vec::with_capacity(ids.len() + 1);
        bounds.push(0u32);
        let mut off = 0u32;
        for &id in &ids {
            off += self.v.token_len(id);
            bounds.push(off);
        }
        (ids, bounds)
    }

    /// Encode, turning every occurrence of any special-token string into its id.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.encode_allowed(text, |_| true)
    }

    /// Encode with only the specials selected by `allow` treated as specials.
    pub fn encode_allowed(&self, text: &str, allow: impl Fn(&str) -> bool) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 8);
        let bytes = text.as_bytes();
        let allowed: Vec<&(String, u32)> = self.specials.iter().filter(|(s, _)| allow(s)).collect();
        let mut p = 0usize;
        while p < bytes.len() {
            // leftmost occurrence of any allowed special from p
            let mut best_pos = usize::MAX;
            let mut best_i = 0usize;
            for (i, (s, _)) in allowed.iter().enumerate() {
                if let Some(q) = find_from(bytes, s.as_bytes(), p) {
                    if q < best_pos {
                        best_pos = q;
                        best_i = i;
                    }
                }
            }
            if best_pos == usize::MAX {
                self.encode_into(&bytes[p..], &mut out);
                break;
            }
            if best_pos > p {
                self.encode_into(&bytes[p..best_pos], &mut out);
            }
            out.push(allowed[best_i].1);
            p = best_pos + allowed[best_i].0.len();
        }
        out
    }

    pub fn decode_into(&self, ids: &[u32], out: &mut Vec<u8>) {
        for &id in ids {
            if id < self.v.n {
                out.extend_from_slice(self.v.token_bytes(id));
                continue;
            }
            for (s, sid) in &self.specials {
                if *sid == id {
                    out.extend_from_slice(s.as_bytes());
                    break;
                }
            }
            // other out-of-range ids are skipped
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        self.decode_into(ids, &mut out);
        out
    }

    // ---- the fused pretok + merge product machines ----
    //
    // One loop owns BOTH the pretok boundary rules and token emission for the full
    // ASCII grammar of each family — no per-piece scanner dispatch on the hot path.
    // ANY Unicode contact at a run boundary falls back to the exact scalar scanner
    // for one piece, so output is byte-exact by construction.

    #[inline(always)]
    fn emit(&self, t: &[u8], off: usize, plen: usize, out: &mut Vec<u32>) {
        let piece = unsafe { t.get_unchecked(off..off + plen) };
        let first = self.v.next_match(piece);
        self.v.encode_with_first(piece, first, out);
    }

    /// pieces starting with a *valid* 3-byte UTF-8 char take the multibyte-optimized
    /// encoder. An ill-formed 3-byte-lead sequence must fall through to the
    /// byte-accurate path — otherwise encode would be lossy on invalid UTF-8.
    #[inline(always)]
    fn merge_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if piece.len() >= 3
            && piece[0] >= 0xE0
            && piece[0] < 0xF0
            && (piece[1] & 0xC0) == 0x80
            && (piece[2] & 0xC0) == 0x80
        {
            encode_mb(&self.v, piece, out);
        } else {
            self.v.encode(piece, out);
        }
    }

    fn encode_core_cl100k(&self, t: &[u8], out: &mut Vec<u32>) {
        let l = t.len();
        let mut p = 0usize;
        while p < l {
            let b0 = unsafe { *t.get_unchecked(p) };
            let mut fb = b0 >= 0x80;
            let mut adv = 0usize;
            if !fb {
                'blk: {
                    if b0 == b'\'' && p + 1 < l {
                        // '(?i:[sdmt]|ll|ve|re)
                        let c1 = at(t, p + 1) | 0x20;
                        if c1 == b's' || c1 == b'd' || c1 == b'm' || c1 == b't' {
                            self.emit(t, p, 2, out);
                            adv = 2;
                            break 'blk;
                        }
                        if p + 2 < l {
                            let c2 = at(t, p + 2) | 0x20;
                            if (c1 == b'l' && c2 == b'l')
                                || (c1 == b'v' && c2 == b'e')
                                || (c1 == b'r' && c2 == b'e')
                            {
                                self.emit(t, p, 3, out);
                                adv = 3;
                                break 'blk;
                            }
                        }
                    }
                    // [^\r\n\p{L}\p{N}]?+\p{L}++
                    let ls = if a_let(b0) {
                        Some(p)
                    } else if !a_dig(b0)
                        && b0 != b'\r'
                        && b0 != b'\n'
                        && p + 1 < l
                        && a_let(at(t, p + 1))
                    {
                        Some(p + 1)
                    } else {
                        None
                    };
                    if let Some(ls) = ls {
                        let we = pretok::ascii_letter_run(t, ls, l);
                        if we < l && at(t, we) >= 0x80 {
                            fb = true;
                            break 'blk;
                        }
                        self.emit(t, p, we - p, out);
                        adv = we - p;
                        break 'blk;
                    }
                    if a_dig(b0) {
                        // \p{N}{1,3}
                        let (mut q, mut c) = (p + 1, 1);
                        while q < l && c < 3 && a_dig(at(t, q)) {
                            q += 1;
                            c += 1;
                        }
                        if c < 3 && q < l && at(t, q) >= 0x80 {
                            fb = true;
                            break 'blk;
                        }
                        self.emit(t, p, q - p, out);
                        adv = q - p;
                        break 'blk;
                    }
                    {
                        //  ?punct+[\r\n]*
                        let mut q = p + if b0 == b' ' { 1 } else { 0 };
                        let s4 = q;
                        while q < l && a_pun(at(t, q)) {
                            q += 1;
                        }
                        if q > s4 {
                            if q < l && at(t, q) >= 0x80 {
                                fb = true;
                                break 'blk;
                            }
                            while q < l && (at(t, q) == b'\r' || at(t, q) == b'\n') {
                                q += 1;
                            }
                            self.emit(t, p, q - p, out);
                            adv = q - p;
                            break 'blk;
                        }
                    }
                    if a_ws(b0) {
                        // \s++$ | \s*[\r\n] | \s+(?!\S) | \s
                        let mut e2 = p;
                        let mut lastnl = usize::MAX;
                        while e2 < l && a_ws(at(t, e2)) {
                            if at(t, e2) == b'\r' || at(t, e2) == b'\n' {
                                lastnl = e2;
                            }
                            e2 += 1;
                        }
                        if e2 < l && at(t, e2) >= 0x80 {
                            fb = true;
                            break 'blk;
                        }
                        let plen = if e2 == l {
                            e2 - p
                        } else if lastnl != usize::MAX {
                            lastnl + 1 - p
                        } else if e2 - p > 1 {
                            e2 - p - 1
                        } else {
                            1
                        };
                        self.emit(t, p, plen, out);
                        adv = plen;
                        break 'blk;
                    }
                    fb = true; // unreachable for ASCII
                }
            }
            if fb {
                let mut len = pretok::pretok_next(&self.u, t, p, l);
                if len == 0 {
                    len = 1;
                }
                self.merge_piece(&t[p..p + len], out);
                p += len;
            } else {
                p += adv;
            }
        }
    }

    fn encode_core_o200k(&self, t: &[u8], out: &mut Vec<u32>) {
        let l = t.len();
        let u = &self.uo;
        let mut p = 0usize;
        // Grammar mirrored from pretok_next_o200k: prefix-consumed-FIRST on both word
        // alts; UPPER*LOWER+ then UPPER+LOWER* (for ASCII a pure [A-Z]* run has no
        // LOWER-eligible char, so the CJK backtrack scan never applies natively);
        // ATTACHED contractions; '/' in the punct tail; lastnl-first ws order.
        while p < l {
            let b0 = unsafe { *t.get_unchecked(p) };
            let mut fb = b0 >= 0x80;
            let mut adv = 0usize;
            if !fb {
                let mut hitmb = false;
                'blk: {
                    // UPPER* LOWER+
                    let match_ul = |st: usize, hitmb: &mut bool| -> usize {
                        let i = pretok::ascii_upper_run(t, st, l);
                        if i < l && at(t, i) >= 0x80 {
                            *hitmb = true;
                            return 0;
                        }
                        let j = pretok::ascii_lower_run(t, i, l);
                        if j < l && at(t, j) >= 0x80 {
                            *hitmb = true;
                            return 0;
                        }
                        if j > i {
                            j
                        } else {
                            0
                        }
                    };
                    // UPPER+ LOWER*
                    let match_upl = |st: usize, hitmb: &mut bool| -> usize {
                        let i = pretok::ascii_upper_run(t, st, l);
                        if i == st {
                            return 0;
                        }
                        if i < l && at(t, i) >= 0x80 {
                            *hitmb = true;
                            return 0;
                        }
                        let j = pretok::ascii_lower_run(t, i, l);
                        if j < l && at(t, j) >= 0x80 {
                            *hitmb = true;
                            return 0;
                        }
                        j
                    };
                    let prefelig = !a_let(b0) && !a_dig(b0) && b0 != b'\r' && b0 != b'\n';
                    if prefelig && p + 1 < l && at(t, p + 1) >= 0x80 {
                        fb = true;
                        break 'blk;
                    }
                    let mut e = 0usize;
                    if prefelig && p + 1 < l {
                        e = match_ul(p + 1, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'blk;
                    }
                    if e == 0 {
                        e = match_ul(p, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'blk;
                    }
                    if e == 0 && prefelig && p + 1 < l {
                        e = match_upl(p + 1, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'blk;
                    }
                    if e == 0 {
                        e = match_upl(p, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'blk;
                    }
                    if e != 0 {
                        let e = pk8::o_contraction(t, e, l);
                        self.emit(t, p, e - p, out);
                        adv = e - p;
                        break 'blk;
                    }
                    if a_dig(b0) {
                        // \p{N}{1,3}
                        let (mut q, mut c) = (p + 1, 1);
                        while q < l && c < 3 && a_dig(at(t, q)) {
                            q += 1;
                            c += 1;
                        }
                        if c < 3 && q < l && at(t, q) >= 0x80 {
                            fb = true;
                            break 'blk;
                        }
                        self.emit(t, p, q - p, out);
                        adv = q - p;
                        break 'blk;
                    }
                    {
                        //  ?punct+[\r\n/]*
                        let mut q = p + if b0 == b' ' { 1 } else { 0 };
                        let s4 = q;
                        while q < l && a_pun(at(t, q)) {
                            q += 1;
                        }
                        if q > s4 {
                            if q < l && at(t, q) >= 0x80 {
                                fb = true;
                                break 'blk;
                            }
                            while q < l
                                && (at(t, q) == b'\r' || at(t, q) == b'\n' || at(t, q) == b'/')
                            {
                                q += 1;
                            }
                            self.emit(t, p, q - p, out);
                            adv = q - p;
                            break 'blk;
                        }
                    }
                    if a_ws(b0) {
                        // \s*[\r\n]+ | \s+(?!\S) | \s+
                        let mut e2 = p;
                        let mut lastnl = usize::MAX;
                        while e2 < l && a_ws(at(t, e2)) {
                            if at(t, e2) == b'\r' || at(t, e2) == b'\n' {
                                lastnl = e2;
                            }
                            e2 += 1;
                        }
                        if e2 < l && at(t, e2) >= 0x80 {
                            fb = true;
                            break 'blk;
                        }
                        let plen = if lastnl != usize::MAX {
                            lastnl + 1 - p
                        } else if e2 == l {
                            e2 - p
                        } else if e2 - p > 1 {
                            e2 - p - 1
                        } else {
                            1
                        };
                        self.emit(t, p, plen, out);
                        adv = plen;
                        break 'blk;
                    }
                    fb = true; // unreachable for ASCII
                }
            }
            if fb {
                let mut len = pk8::pretok_next_o200k(u, t, p, l);
                if len == 0 {
                    len = 1;
                }
                self.merge_piece(&t[p..p + len], out);
                p += len;
            } else {
                p += adv;
            }
        }
    }
}

/// Unchecked byte read — the product machines below index only positions they
/// have already bounds-tested against `l`, and the checks cost ~10% on ASCII.
#[inline(always)]
fn at(t: &[u8], i: usize) -> u8 {
    debug_assert!(i < t.len());
    unsafe { *t.get_unchecked(i) }
}

fn unknown_encoding(encoding: &str) -> VocabError {
    VocabError(format!(
        "toktok: unknown encoding: {encoding} (built-in: {})",
        crate::BUILTIN_ENCODINGS.join(", ")
    ))
}

// ASCII class predicates for the product machines. a_pun deliberately includes
// control bytes and DEL: the regex class is [^\s\p{L}\p{N}].
#[inline(always)]
fn a_let(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') <= 25
}
#[inline(always)]
fn a_dig(b: u8) -> bool {
    b.wrapping_sub(b'0') <= 9
}
#[inline(always)]
fn a_ws(b: u8) -> bool {
    b.wrapping_sub(9) <= 4 || b == b' '
}
#[inline(always)]
fn a_pun(b: u8) -> bool {
    b < 0x80 && !a_ws(b) && !a_let(b) && !a_dig(b)
}

#[inline]
fn find_from(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from + needle.len() > hay.len() {
        return None;
    }
    let first = needle[0];
    let mut i = from;
    while i + needle.len() <= hay.len() {
        if hay[i] == first && &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

impl Tokenizer {
    /// Token counts for many texts, in parallel — the counting counterpart of
    /// `encode_batch`. Nothing is materialized: one scratch buffer per thread is
    /// reused across documents, so a counting workload allocates O(threads)
    /// instead of O(total tokens).
    pub fn count_batch(&self, texts: &[&[u8]], threads: usize, with_special: bool) -> Vec<u32> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut out = vec![0u32; texts.len()];
        if texts.is_empty() {
            return out;
        }
        let count1 = |s: &[u8], buf: &mut Vec<u32>| -> u32 {
            buf.clear();
            if with_special {
                // encode_allowed builds its own vector; counting still beats
                // encode_batch because nothing is kept after the count
                if let Ok(st) = std::str::from_utf8(s) {
                    return self.encode_with_special(st).len() as u32;
                }
            }
            self.encode_into(s, buf);
            buf.len() as u32
        };
        let n = self.threads_for(threads, texts.len());
        if n <= 1 {
            let mut buf = Vec::with_capacity(4096);
            for (i, t) in texts.iter().enumerate() {
                out[i] = count1(t, &mut buf);
            }
            return out;
        }
        struct Slots(*mut u32);
        unsafe impl Sync for Slots {}
        let slots = Slots(out.as_mut_ptr());
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let (slots, next) = (&slots, &next);
            for _ in 0..n {
                scope.spawn(move || {
                    let mut buf = Vec::with_capacity(4096);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= texts.len() {
                            break;
                        }
                        let c = count1(texts[i], &mut buf);
                        unsafe { *slots.0.add(i) = c };
                    }
                });
            }
        });
        out
    }

    /// `truncate` over many texts, in parallel.
    pub fn truncate_batch(
        &self,
        texts: &[&[u8]],
        max_tokens: usize,
        threads: usize,
    ) -> Vec<Truncation> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let empty = Truncation {
            bytes: 0,
            total_tokens: 0,
            kept_tokens: 0,
        };
        let mut out = vec![empty; texts.len()];
        if texts.is_empty() {
            return out;
        }
        let n = self.threads_for(threads, texts.len());
        if n <= 1 {
            for (i, t) in texts.iter().enumerate() {
                out[i] = self.truncate(t, max_tokens);
            }
            return out;
        }
        struct Slots(*mut Truncation);
        unsafe impl Sync for Slots {}
        let slots = Slots(out.as_mut_ptr());
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let (slots, next) = (&slots, &next);
            for _ in 0..n {
                scope.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= texts.len() {
                        break;
                    }
                    let t = self.truncate(texts[i], max_tokens);
                    unsafe { *slots.0.add(i) = t };
                });
            }
        });
        out
    }

    /// Thread count for a batch call: `threads == 0` means hardware concurrency,
    /// and never more threads than there are documents.
    fn threads_for(&self, threads: usize, ndocs: usize) -> usize {
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n = if threads != 0 { threads } else { hw };
        n.min(ndocs)
    }

    /// Encode many texts in parallel (work-stealing over an atomic cursor, like
    /// the C++ `encode_batch`). `threads == 0` picks the hardware concurrency.
    pub fn encode_batch(
        &self,
        texts: &[&[u8]],
        threads: usize,
        with_special: bool,
    ) -> Vec<Vec<u32>> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); texts.len()];
        if texts.is_empty() {
            return out;
        }
        let enc1 = |s: &[u8]| -> Vec<u32> {
            if with_special {
                // specials are matched on bytes; invalid UTF-8 can't contain one
                match std::str::from_utf8(s) {
                    Ok(st) => self.encode_with_special(st),
                    Err(_) => self.encode(s),
                }
            } else {
                self.encode(s)
            }
        };
        let n = self.threads_for(threads, texts.len());
        if n <= 1 {
            for (i, t) in texts.iter().enumerate() {
                out[i] = enc1(t);
            }
            return out;
        }
        // work-stealing over an atomic cursor; each index is written by exactly
        // one thread, so the raw-pointer handoff below is race-free.
        struct Slots(*mut Vec<u32>);
        unsafe impl Sync for Slots {}
        let slots = Slots(out.as_mut_ptr());
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let slots = &slots;
            let next = &next;
            for _ in 0..n {
                scope.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= texts.len() {
                        break;
                    }
                    let ids = enc1(texts[i]);
                    unsafe { *slots.0.add(i) = ids };
                });
            }
        });
        out
    }
}
