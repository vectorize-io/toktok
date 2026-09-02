//! Multibyte-piece encoder — the TRIE2 walk specialized for pieces that start
//! with a 3-byte UTF-8 lead (CJK and friends). Port of quicktok's
//! `src/trie2_mb.cpp`.
//!
//!  - 3-byte UTF-8 start: r3 resolves the whole first char with 0 probes
//!  - in the loop, bytes >= 0x80 probe the odd table FIRST (CJK tokens end at odd
//!    depths 3, 9, ...); the hasdeeper flag lets the walk stop after 1 probe
//!  - ASCII bytes inside a mixed piece fall back to the e2-first order

use crate::vocab::{
    mix36, Vocab, E2BEST_NONE, E2_CMP, E2_USED, OTAB_DEEPBIT, OTAB_TOKMASK, RANK_MAX,
};

/// next_match with the multibyte-optimized walk.
#[inline]
pub fn nm_mb(v: &Vocab, text: &[u8]) -> u32 {
    let len = text.len();
    if len == 0 {
        return RANK_MAX;
    }
    unsafe {
        if len == 1 {
            let n = *v.root_child.get_unchecked(*text.get_unchecked(0) as usize);
            return if n != 0 {
                *v.tnode_tok.get_unchecked(n as usize)
            } else {
                RANK_MAX
            };
        }
        let (mut node, mut best);
        let mut i = 2usize;
        let mut odd_covered = false;
        // r3 indexes by the codepoint of a 3-byte UTF-8 char (continuation bits
        // masked), so it is valid ONLY for well-formed sequences; ill-formed bytes
        // take the byte-accurate r2 path so encode stays lossless.
        let b0 = *text.get_unchecked(0);
        if (0xE0..0xF0).contains(&b0)
            && len >= 3
            && (*text.get_unchecked(1) & 0xC0) == 0x80
            && (*text.get_unchecked(2) & 0xC0) == 0x80
        {
            let i3 = (((b0 as usize) & 0xF) << 12)
                | (((*text.get_unchecked(1) as usize) & 0x3F) << 6)
                | ((*text.get_unchecked(2) as usize) & 0x3F);
            node = *v.r3node.get_unchecked(i3);
            best = *v.r3best.get_unchecked(i3);
            odd_covered = true; // depth<=3 known, 0 probes
        } else {
            let idx = ((b0 as usize) << 8) | *text.get_unchecked(1) as usize;
            node = *v.r2node.get_unchecked(idx);
            best = *v.r2best.get_unchecked(idx);
        }
        while node != 0 && i + 1 < len {
            let mut odd_done = odd_covered;
            odd_covered = false;
            if !odd_done && *text.get_unchecked(i) >= 0x80 {
                // odd-first: token likely ends at an odd depth
                odd_done = true;
                let o = v.odd_lookup(node, *text.get_unchecked(i));
                if o != RANK_MAX {
                    best = o & OTAB_TOKMASK;
                    if (o as u64) & OTAB_DEEPBIT == 0 {
                        return best; // no deeper token exists: done in 1 probe
                    }
                }
            }
            let k = ((node as u64) << 16)
                | ((*text.get_unchecked(i) as u64) << 8)
                | *text.get_unchecked(i + 1) as u64;
            let m = mix36(k);
            let mut h = (m >> v.e2tb) as u32;
            let want = ((m & ((1 << 25) - 1)) << 38) | E2_USED;
            let mut val = 0u64;
            loop {
                let s = *v.e2.get_unchecked(h as usize);
                if s == 0 {
                    break;
                }
                if s & E2_CMP == want {
                    val = s;
                    break;
                }
                h = (h + 1) & v.e2mask;
            }
            if val == 0 {
                if !odd_done {
                    let o = v.odd_lookup(node, *text.get_unchecked(i));
                    if o != RANK_MAX {
                        best = o & OTAB_TOKMASK;
                    }
                }
                return best;
            }
            let b18 = (val as u32) & E2BEST_NONE;
            if b18 != E2BEST_NONE {
                best = b18;
            }
            node = ((val >> 18) as u32) & 0xFFFFF;
            i += 2;
        }
        if node != 0 && i < len && !odd_covered {
            let o = v.odd_lookup(node, *text.get_unchecked(i));
            if o != RANK_MAX {
                best = o & OTAB_TOKMASK;
            }
        }
        best
    }
}

/// Mirror of `Vocab::encode_with_first` (greedy fast path + backtracking
/// fallback) with every next_match replaced by `nm_mb`.
pub fn encode_mb(v: &Vocab, text: &[u8], out: &mut Vec<u32>) {
    let len = text.len() as u32;
    if len == 0 {
        return;
    }
    let first = nm_mb(v, text);
    let out_start = out.len();
    {
        let (mut pos, mut last, mut nt) = (0u32, RANK_MAX, first);
        let mut ok = true;
        while pos < len {
            let token = nt;
            if last != RANK_MAX && !v.is_valid_token_pair(last, token) {
                ok = false;
                break;
            }
            out.push(token);
            last = token;
            pos += v.token_len(token);
            nt = if pos < len {
                nm_mb(v, unsafe { text.get_unchecked(pos as usize..) })
            } else {
                RANK_MAX
            };
        }
        if ok {
            return;
        }
        out.truncate(out_start);
    }
    crate::scratch::with_scratch(|toks, bf| {
        v.backtrack(text, first, toks, bf, |s| nm_mb(v, s));
        out.extend_from_slice(toks);
    });
}
