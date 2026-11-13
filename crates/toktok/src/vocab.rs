//! Vocab: trie tables, merge-validity memo, and the backtracking BPE encoder.
//!
//! Faithful Rust port of quicktok's `src/bpe.hpp` (itself a port of the `bpe`
//! crate's `encode_via_backtracking`). Everything the encoder needs:
//!   next_match       longest vocab token that is a prefix of text  (2-byte trie)
//!   next_prefix[id]  longest proper-prefix token of token id
//!   split[id]        the two tokens id was merged from (or (id,id) if original)
//!   pair_lookup      (a,b) -> merged token, if it exists
//!   is_valid_token_pair  merge-reversal compatibility check
//!
//! The data-structure engineering (packed 2-byte trie slots, dense bijectively
//! mixed validity memos, odd-depth side table, r2/r3 direct tables) is what makes
//! this fast; the layouts and bit packings below are bit-identical to the C++.

// The table-construction loops below walk token ids and index several parallel
// arrays with them (`tb`, `tnode_tok`, `npm`, `split`). Rewriting them as zipped
// iterators would obscure that they are all keyed by the same id.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};

pub const RANK_MAX: u32 = u32::MAX;
// pair_lookup value width: a token id, or a merge rank (embeddinggemma-2 has
// 514,906 merges plus its starting symbols, past 19 bits)
const PL_VALBITS: u32 = 20;
const PL_VALMASK: u64 = (1 << PL_VALBITS) - 1;

// memo capacity. 2^20 peak on M1; x86 wants 2^21 for the wide-id memo.
const IVBITS: u32 = 20;
const IVTAGBITS: u32 = 34 - IVBITS;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
const IVBITS_W: u32 = 21;
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
const IVBITS_W: u32 = 20;

/// bijective 36-bit mixer (self-inverse xorshift / odd multiply mod 2^36 / xorshift)
/// — shared by the wide-id validity memo and the packed e2 trie table. Because every
/// stage is invertible, index bits + tag bits reconstruct the input exactly: a tagged
/// slot can never alias a different key.
#[inline(always)]
pub fn mix36(k: u64) -> u64 {
    let m = k ^ (k >> 18);
    let m = m.wrapping_mul((0x9E37_79B9_7F4A_7C15u64 & 0xF_FFFF_FFFF) | 1) & 0xF_FFFF_FFFF;
    m ^ (m >> 18)
}

// odd-token side-table layout: 18-bit token field (fits cl100k's 100,256 and
// o200k's 199,998 ids). slot = (key+1)<<19 | hasdeeper<<18 | token.
pub const OTAB_TOKMASK: u32 = (1 << 18) - 1;
pub const OTAB_DEEPBIT: u64 = 1 << 18;
const OTAB_KEYSH: u32 = 19;
const OTAB_LOWMASK: u32 = (1 << 19) - 1;

// target load factors (per-arch tunables in the C++ build; the M1 defaults)
const EDGE_LOAD: f64 = 0.45;
const E2_LOAD: f64 = 0.45;

const HMUL: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Debug)]
pub struct VocabError(pub String);

impl std::fmt::Display for VocabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for VocabError {}

pub struct Vocab {
    pub all: Vec<u8>,     // token bytes concatenated, by id
    pub tstart: Vec<u32>, // tstart[id]..tstart[id+1]
    pub tlen: Vec<u8>,    // token byte-length — L1-hot, 1 load vs 2 in tstart
    pub n: u32,
    b2id: HashMap<Box<[u8]>, u32>,
    // trie edges: open-addressing, slot = (key+1)<<32 | child; key = node<<8|byte
    etab: Vec<u64>,
    emask: u32,
    enodes: u32, // occupied etab slots, for the growth check during construction
    pub root_child: [u32; 256], // direct edges from root (hottest: next_match restarts here)
    pub r2node: Vec<u32>,
    pub r2best: Vec<u32>, // 65536: node after 2 bytes / deepest token in first <=2 bytes
    pub tnode_tok: Vec<u32>, // per trie node: token id ending here, or RANK_MAX
    npm: Vec<u32>,        // next_prefix_match
    split: Vec<(u32, u32)>,
    /// Merge priority per id, when it isn't the id itself (empty otherwise):
    /// Qwen3.5 has merges whose parts form after the token they make, and those
    /// fire as soon as the later part exists.
    order: Vec<u32>,
    /// Pair table holds each merge's own firing rank, not the merged token
    /// (embeddinggemma-2: one token can be built by several merges, each with
    /// its own priority). `order` then holds each token's building merge rank.
    pair_rank: bool,
    // with pair_rank, plk maps a pair to its merge's firing rank (what
    // is_valid_token_pair compares); per firing rank - nbase, the merge's list
    // index (the exact encoder's priority) and the token it makes
    fraw: Vec<u32>,
    fout: Vec<u32>,
    // starting symbols, for the exact encoder: a byte's fallback token, and how
    // many ids rank as starting symbols (order[id] < nbase)
    byte_tok: Vec<u32>,
    nbase: u32,
    // pair_lookup, flat open-addressing, ONE u64/slot: ((t1<<18|t2)+1)<<20 | rank
    plk: Vec<u64>,
    plmask: u32,
    // dense direct-mapped memo for is_valid_token_pair: u16/slot,
    // slot = 0x8000 | tag<<1 | result; 0 = empty. Relaxed atomics: each slot value
    // is self-consistent (tag+result in one store), so concurrent encode() on one
    // Tokenizer is safe — a racing thread sees old or new, both correct.
    ivm: Vec<AtomicU16>,
    ivmask: u32,
    // wide-id memo (vocabs with ids >= 2^17, e.g. o200k): same scheme over a
    // 36-bit pair key. u32 slots = 0x8000_0000 | tag<<1 | result.
    ivmw: Vec<AtomicU32>,
    pub wide_ids: bool,
    // 2-byte-radix trie: the walk consumes 2 bytes per ONE 8-byte slot load.
    //   slot = [63]=used | [62:38]=tag25 | [37:18]=child | [17:0]=best (0x3FFFF=none)
    pub e2: Vec<u64>,
    pub e2mask: u32,
    pub e2tb: u32, // index shift = 36 - log2(slots)
    pub otab: Vec<u64>,
    pub omask: u32,
    pub r3node: Vec<u32>,
    pub r3best: Vec<u32>,
}

pub const E2_USED: u64 = 1 << 63;
pub const E2BEST_NONE: u32 = 0x3FFFF;
const E2_TAGMASK: u64 = (1 << 25) - 1;
pub const E2_CMP: u64 = E2_USED | (E2_TAGMASK << 38);

impl Vocab {
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.n as usize
    }
    #[inline(always)]
    pub fn token_len(&self, id: u32) -> u32 {
        unsafe { *self.tlen.get_unchecked(id as usize) as u32 }
    }
    #[inline(always)]
    pub fn next_prefix(&self, id: u32) -> u32 {
        unsafe { *self.npm.get_unchecked(id as usize) }
    }

    #[inline(always)]
    fn pl_get(&self, key: u64) -> u32 {
        let mut i = (key.wrapping_mul(HMUL) >> 40) as u32 & self.plmask;
        let want = (key + 1) << PL_VALBITS;
        loop {
            let s = unsafe { *self.plk.get_unchecked(i as usize) };
            if s == 0 {
                return RANK_MAX;
            }
            if s & !PL_VALMASK == want {
                return (s & PL_VALMASK) as u32;
            }
            i = (i + 1) & self.plmask;
        }
    }
    fn pl_put(&mut self, key: u64, val: u32) {
        let mut i = (key.wrapping_mul(HMUL) >> 40) as u32 & self.plmask;
        let want = (key + 1) << PL_VALBITS;
        loop {
            let s = self.plk[i as usize];
            if s == 0 {
                break;
            }
            if s & !PL_VALMASK == want {
                self.plk[i as usize] = want | val as u64;
                return;
            }
            i = (i + 1) & self.plmask;
        }
        self.plk[i as usize] = want | val as u64;
    }

    // ---- byte trie: construction only. The runtime walk uses e2/otab/r2/r3, so
    // etab is dropped at the end of `load` and this must not be called after it.
    #[inline]
    fn edge(&self, node: u32, b: u8) -> u32 {
        if node == 0 {
            return self.root_child[b as usize];
        }
        let k = ((node as u64) << 8) | b as u64;
        let mut i = (k.wrapping_mul(HMUL) >> 40) as u32 & self.emask;
        let key = (k + 1) as u32;
        loop {
            let s = self.etab[i as usize];
            if s == 0 {
                return 0; // node 0 = root = "no edge" (root is never a child)
            }
            if (s >> 32) as u32 == key {
                return s as u32;
            }
            i = (i + 1) & self.emask;
        }
    }

    /// (hasdeeper|tok) or RANK_MAX if absent
    #[inline(always)]
    pub fn odd_lookup(&self, node: u32, b: u8) -> u32 {
        let k = ((node as u64) << 8) | b as u64;
        let mut i = (k.wrapping_mul(HMUL) >> 40) as u32 & self.omask;
        loop {
            let s = unsafe { *self.otab.get_unchecked(i as usize) };
            if s == 0 {
                return RANK_MAX;
            }
            if (s >> OTAB_KEYSH) == k + 1 {
                return (s as u32) & OTAB_LOWMASK;
            }
            i = (i + 1) & self.omask;
        }
    }

    /// longest token that is a prefix of text[0..len) — 2-byte steps, 1 slot load each.
    ///
    /// NOTE: deliberately NO multibyte dispatch in here; multibyte pieces are routed
    /// per-piece by the tokenizer to `encode_mb` (see mb.rs) — an extra branch here
    /// costs the ASCII hot path measurably.
    #[inline(always)]
    pub fn next_match(&self, text: &[u8]) -> u32 {
        let len = text.len();
        if len == 0 {
            return RANK_MAX;
        }
        unsafe {
            if len == 1 {
                let n = *self
                    .root_child
                    .get_unchecked(*text.get_unchecked(0) as usize);
                return if n != 0 {
                    *self.tnode_tok.get_unchecked(n as usize)
                } else {
                    RANK_MAX
                };
            }
            let idx = ((*text.get_unchecked(0) as usize) << 8) | *text.get_unchecked(1) as usize;
            let mut node = *self.r2node.get_unchecked(idx);
            let mut best = *self.r2best.get_unchecked(idx);
            let mut i = 2usize;
            while node != 0 && i + 1 < len {
                let k = ((node as u64) << 16)
                    | ((*text.get_unchecked(i) as u64) << 8)
                    | *text.get_unchecked(i + 1) as u64;
                let m = mix36(k);
                let mut h = (m >> self.e2tb) as u32;
                let want = ((m & E2_TAGMASK) << 38) | E2_USED;
                let mut val = 0u64;
                loop {
                    let s = *self.e2.get_unchecked(h as usize);
                    if s == 0 {
                        break;
                    }
                    if s & E2_CMP == want {
                        val = s;
                        break;
                    }
                    h = (h + 1) & self.e2mask;
                }
                if val == 0 {
                    // no 2-byte step: an odd-depth token may still extend one byte
                    let o = self.odd_lookup(node, *text.get_unchecked(i));
                    if o != RANK_MAX {
                        best = o & OTAB_TOKMASK;
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
            if node != 0 && i < len {
                let o = self.odd_lookup(node, *text.get_unchecked(i));
                if o != RANK_MAX {
                    best = o & OTAB_TOKMASK;
                }
            }
            best
        }
    }

    #[inline(always)]
    pub fn is_valid_token_pair(&self, t1: u32, t2: u32) -> bool {
        if self.wide_ids {
            return self.ivtp_wide(t1, t2);
        }
        let mk = ((t1 as u64) << 17) | t2 as u64; // 34-bit pair key
        let m = mk ^ (mk >> 17);
        let m = m.wrapping_mul((HMUL & 0x3_FFFF_FFFF) | 1) & 0x3_FFFF_FFFF;
        let m = m ^ (m >> 17); // bijection done
        let h = (m >> IVTAGBITS) as u32;
        let want = (0x8000u32 | (((m as u32) & ((1 << IVTAGBITS) - 1)) << 1)) as u16;
        let s = unsafe { self.ivm.get_unchecked(h as usize).load(Ordering::Relaxed) };
        if s & 0xFFFE == want {
            return s & 1 != 0;
        }
        let res = self.ivtp_slow(t1, t2);
        unsafe {
            self.ivm
                .get_unchecked(h as usize)
                .store(want | res as u16, Ordering::Relaxed)
        };
        res
    }

    /// wide-id (o200k-class) validity memo: dense u32 slots over a 36-bit pair key.
    #[inline]
    pub fn ivtp_wide(&self, t1: u32, t2: u32) -> bool {
        let m = mix36(((t1 as u64) << 18) | t2 as u64);
        let h = (m >> (36 - IVBITS_W)) as u32;
        let want = 0x8000_0000u32 | (((m as u32) & ((1 << (36 - IVBITS_W)) - 1)) << 1);
        let s = unsafe { self.ivmw.get_unchecked(h as usize).load(Ordering::Relaxed) };
        if s & 0xFFFF_FFFE == want {
            return s & 1 != 0;
        }
        let res = self.ivtp_slow(t1, t2);
        unsafe {
            self.ivmw
                .get_unchecked(h as usize)
                .store(want | res as u32, Ordering::Relaxed)
        };
        res
    }

    #[inline(always)]
    fn order(&self, id: u32) -> u32 {
        if self.order.is_empty() {
            id
        } else {
            self.order[id as usize]
        }
    }

    fn ivtp_slow(&self, t1: u32, t2: u32) -> bool {
        let (mut t1, mut t2) = (t1, t2);
        let mut limit = RANK_MAX;
        loop {
            let c = self.pl_get(((t1 as u64) << 18) | t2 as u64);
            if c != RANK_MAX && (if self.pair_rank { c } else { self.order(c) }) < limit {
                return false;
            }
            let (o1, o2) = (t1, t2);
            if self.order(o1) > self.order(o2) {
                limit = self.order(o1);
                t1 = unsafe { self.split.get_unchecked(o1 as usize).1 };
                if t1 == o1 {
                    limit = self.order(o2) + 1;
                    t2 = unsafe { self.split.get_unchecked(o2 as usize).0 };
                    if t2 == o2 {
                        return true;
                    }
                }
            } else {
                limit = self.order(o2) + 1;
                t2 = unsafe { self.split.get_unchecked(o2 as usize).0 };
                if t2 == o2 {
                    limit = self.order(o1);
                    t1 = unsafe { self.split.get_unchecked(o1 as usize).1 };
                    if t1 == o1 {
                        return true;
                    }
                }
            }
        }
    }

    /// Byte pairs no producible token contains, as a 256x256 bit table
    /// (`prev << 8 | next`): BPE can never merge across one, so text can be cut
    /// there and each side encoded alone. Only for vocabs with a merge list.
    pub fn cut_table(&self) -> Vec<u64> {
        let mut joined = vec![0u64; 1024];
        for id in 0..self.n as usize {
            if self.order[id] == RANK_MAX {
                continue; // never produced
            }
            for w in self.token_bytes(id as u32).windows(2) {
                let k = (w[0] as usize) << 8 | w[1] as usize;
                joined[k >> 6] |= 1 << (k & 63);
            }
        }
        joined.iter().map(|w| !w).collect()
    }

    /// Hugging Face's BPE, merge by merge: lowest list rank first, leftmost on
    /// ties, starting from characters with byte fallback. Only for vocabs with
    /// a merge list (`MRG1`); embeddinggemma-2 sends long whitespace runs here,
    /// where many equal merges race and the backtracking check can't tell.
    pub fn encode_exact(&self, text: &[u8], out: &mut Vec<u32>) {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        debug_assert!(self.pair_rank);
        // (id, prev, next); a merged-away symbol gets id RANK_MAX
        let mut sym: Vec<(u32, u32, u32)> = Vec::with_capacity(text.len());
        let mut i = 0;
        while i < text.len() {
            if text[i] < 0x80 {
                // the character's token, or its byte's when the vocab lacks it
                sym.push((self.byte_tok[text[i] as usize], 0, 0));
                i += 1;
                continue;
            }
            let l = utf8_len(&text[i..]);
            let id = self.find_id(&text[i..i + l]);
            if id != RANK_MAX && self.order[id as usize] < self.nbase {
                sym.push((id, 0, 0));
            } else {
                for &b in &text[i..i + l] {
                    sym.push((self.byte_tok[b as usize], 0, 0));
                }
            }
            i += l;
        }
        let n = sym.len() as u32;
        for (k, s) in sym.iter_mut().enumerate() {
            s.1 = (k as u32).wrapping_sub(1);
            s.2 = k as u32 + 1;
        }
        // list index of the merge (a, b): its priority here
        let rank = |a: u32, b: u32| match self.pl_get(((a as u64) << 18) | b as u64) {
            RANK_MAX => RANK_MAX,
            f => self.fraw[(f - self.nbase) as usize],
        };
        let mut heap = BinaryHeap::new();
        for k in 0..n.saturating_sub(1) {
            let r = rank(sym[k as usize].0, sym[k as usize + 1].0);
            if r != RANK_MAX {
                heap.push(Reverse((r, k)));
            }
        }
        while let Some(Reverse((r, k))) = heap.pop() {
            let (a, _, nx) = sym[k as usize];
            if a == RANK_MAX || nx >= n || rank(a, sym[nx as usize].0) != r {
                continue; // stale: a side was merged since this was queued
            }
            let nn = sym[nx as usize].2;
            let f = self.pl_get(((a as u64) << 18) | sym[nx as usize].0 as u64);
            sym[k as usize].0 = self.fout[(f - self.nbase) as usize];
            sym[k as usize].2 = nn;
            sym[nx as usize].0 = RANK_MAX;
            if nn < n {
                sym[nn as usize].1 = k;
                let r2 = rank(sym[k as usize].0, sym[nn as usize].0);
                if r2 != RANK_MAX {
                    heap.push(Reverse((r2, k)));
                }
            }
            let pv = sym[k as usize].1;
            if pv < n {
                let r2 = rank(sym[pv as usize].0, sym[k as usize].0);
                if r2 != RANK_MAX {
                    heap.push(Reverse((r2, pv)));
                }
            }
        }
        let mut k = 0u32;
        while k < n {
            out.push(sym[k as usize].0);
            k = sym[k as usize].2;
        }
    }

    // ---- backtracking encode (faithful port of BacktrackEncoder) ----
    #[inline(always)]
    pub fn encode(&self, text: &[u8], out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let first = self.next_match(text);
        self.encode_with_first(text, first, out);
    }

    /// encode using a precomputed `first = next_match(text)` (lets the fused
    /// pretok+merge loop reuse the walk it already did).
    pub fn encode_with_first(&self, text: &[u8], first: u32, out: &mut Vec<u32>) {
        let len = text.len() as u32;
        if len == 0 {
            return;
        }
        let out_start = out.len();
        // GREEDY fast path: backtracking only ever backtracks when is_valid fails
        // (rare), so for any piece that tokenizes greedily the bitfield machinery
        // is pure overhead.
        {
            let (mut pos, mut last, mut nt) = (0u32, RANK_MAX, first);
            let mut ok = true;
            while pos < len {
                let token = nt;
                if last != RANK_MAX && !self.is_valid_token_pair(last, token) {
                    ok = false;
                    break;
                }
                out.push(token);
                last = token;
                pos += self.token_len(token);
                nt = if pos < len {
                    self.next_match(unsafe { text.get_unchecked(pos as usize..) })
                } else {
                    RANK_MAX
                };
            }
            if ok {
                return;
            }
            out.truncate(out_start); // greedy hit an invalid pair -> full backtracking
        }
        crate::scratch::with_scratch(|toks, bf| {
            self.backtrack(text, first, toks, bf, |v| self.next_match(v));
            out.extend_from_slice(toks);
        });
    }

    /// Shared backtracking driver. `nm` supplies next_match (the mb encoder passes
    /// its multibyte-optimized walk); logic is identical either way.
    #[inline(always)]
    pub(crate) fn backtrack<F: Fn(&[u8]) -> u32>(
        &self,
        text: &[u8],
        first: u32,
        toks: &mut Vec<u32>,
        bf: &mut Vec<u64>,
        nm: F,
    ) {
        let len = text.len() as u32;
        toks.clear();
        let words = ((len + 1 + 63) >> 6) as usize;
        bf.clear();
        bf.resize(words, !0u64);
        let mut pos = 0u32;
        let mut next_token = first;
        while next_token != RANK_MAX {
            let mut token = next_token;
            let last = toks.last().copied().unwrap_or(RANK_MAX);
            loop {
                let end = pos + self.token_len(token);
                let is_set = (bf[(end >> 6) as usize] >> (end & 63)) & 1 != 0;
                if is_set && (last == RANK_MAX || self.is_valid_token_pair(last, token)) {
                    toks.push(token);
                    pos = end;
                    next_token = if pos < len {
                        nm(&text[pos as usize..])
                    } else {
                        RANK_MAX
                    };
                    break;
                }
                let shorter = self.next_prefix(token);
                if shorter != RANK_MAX {
                    token = shorter;
                    continue;
                }
                bf[(pos >> 6) as usize] &= !(1u64 << (pos & 63));
                if !toks.is_empty() {
                    toks.pop();
                    pos -= self.token_len(last);
                }
                next_token = last;
                break;
            }
        }
    }

    /// Per-table breakdown of `memory_bytes`, largest first — answers "why is
    /// this encoding N MiB" without a profiler.
    pub fn memory_breakdown(&self) -> Vec<(&'static str, usize)> {
        fn v<T>(x: &[T]) -> usize {
            std::mem::size_of_val(x)
        }
        let b2id: usize = self
            .b2id
            .keys()
            .map(|k| k.len() + std::mem::size_of::<(Box<[u8]>, u32)>())
            .sum();
        let mut out = vec![
            ("e2 (2-byte trie)", v(&self.e2)),
            ("b2id", b2id),
            ("ivm/ivmw (memo)", v(&self.ivm) + v(&self.ivmw)),
            ("plk (pair lookup)", v(&self.plk)),
            ("r2node/r2best", v(&self.r2node) + v(&self.r2best)),
            ("r3node/r3best", v(&self.r3node) + v(&self.r3best)),
            ("tnode_tok", v(&self.tnode_tok)),
            ("otab (odd tokens)", v(&self.otab)),
            ("split", v(&self.split)),
            ("order", v(&self.order)),
            ("merge list", v(&self.fraw) + v(&self.fout)),
            ("npm", v(&self.npm)),
            (
                "token bytes",
                v(&self.all) + v(&self.tstart) + v(&self.tlen),
            ),
        ];
        out.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        out
    }

    /// Exact heap footprint of the built tables, in bytes — deterministic, unlike
    /// an RSS delta (which depends on what the allocator hands back to the OS).
    pub fn memory_bytes(&self) -> usize {
        fn v<T>(x: &[T]) -> usize {
            std::mem::size_of_val(x)
        }
        let b2id: usize = self
            .b2id
            .keys()
            .map(|k| k.len() + std::mem::size_of::<(Box<[u8]>, u32)>())
            .sum();
        v(&self.all)
            + v(&self.tstart)
            + v(&self.tlen)
            + b2id
            + v(&self.r2node)
            + v(&self.r2best)
            + v(&self.tnode_tok)
            + v(&self.npm)
            + v(&self.split)
            + v(&self.order)
            + v(&self.fraw)
            + v(&self.fout)
            + v(&self.plk)
            + v(&self.ivm)
            + v(&self.ivmw)
            + v(&self.e2)
            + v(&self.otab)
            + v(&self.r3node)
            + v(&self.r3best)
    }

    pub fn find_id(&self, b: &[u8]) -> u32 {
        self.b2id.get(b).copied().unwrap_or(RANK_MAX)
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let (a, b) = (self.tstart[id as usize], self.tstart[id as usize + 1]);
        &self.all[a as usize..b as usize]
    }

    // ---- construction ----
    fn edge_build(&mut self, node: u32, b: u8) -> u32 {
        if node == 0 {
            if self.root_child[b as usize] != 0 {
                return self.root_child[b as usize];
            }
            let c = self.tnode_tok.len() as u32;
            self.tnode_tok.push(RANK_MAX);
            self.root_child[b as usize] = c;
            return c;
        }
        let k = ((node as u64) << 8) | b as u64;
        let mut i = (k.wrapping_mul(HMUL) >> 40) as u32 & self.emask;
        let key = (k + 1) as u32;
        loop {
            let s = self.etab[i as usize];
            if s == 0 {
                break;
            }
            if (s >> 32) as u32 == key {
                return s as u32;
            }
            i = (i + 1) & self.emask;
        }
        let child = self.tnode_tok.len() as u32;
        self.tnode_tok.push(RANK_MAX);
        self.etab[i as usize] = ((key as u64) << 32) | child as u64;
        // Grow at the target load factor instead of allocating a worst-case table
        // up front: sizing it `all.len() * 2` costs 32 MiB of transient RSS for
        // cl100k (the allocator keeps it), against ~4 MiB for the table we end up
        // needing. Doubling peaks at 1.5x the final size instead.
        self.enodes += 1;
        if self.enodes as f64 > (self.emask as f64 + 1.0) * EDGE_LOAD {
            self.grow_etab();
        }
        child
    }

    fn grow_etab(&mut self) {
        let want = (self.emask as usize + 1) * 2;
        let nm = (want - 1) as u32;
        let mut ne = vec![0u64; want];
        for &s in self.etab.iter() {
            if s != 0 {
                let k = (s >> 32) - 1;
                let mut j = (k.wrapping_mul(HMUL) >> 40) as u32 & nm;
                while ne[j as usize] != 0 {
                    j = (j + 1) & nm;
                }
                ne[j as usize] = s;
            }
        }
        self.etab = ne;
        self.emask = nm;
    }

    pub fn load(path: &std::path::Path) -> Result<Vocab, VocabError> {
        let mut f = std::fs::File::open(path).map_err(|e| {
            VocabError(format!(
                "toktok: cannot open vocab file {}: {e}",
                path.display()
            ))
        })?;
        let mut raw = Vec::new();
        f.read_to_end(&mut raw)
            .map_err(|e| VocabError(format!("toktok: cannot read {}: {e}", path.display())))?;
        Self::from_bytes(&raw, &path.display().to_string())
    }

    pub fn from_bytes(raw: &[u8], what: &str) -> Result<Vocab, VocabError> {
        let fail = |why: &str| VocabError(format!("toktok: bad vocab file ({why}): {what}"));
        if raw.len() < 4 {
            return Err(fail("truncated header"));
        }
        let n = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        // token ids must fit the 18-bit packing used by the odd-token table and the
        // 36-bit memo pair key (cl100k = 100,256; o200k = 199,998)
        if n == 0 || n > (1 << 18) {
            return Err(fail("token count out of range (max 262144 ids)"));
        }
        let n_us = n as usize;
        let mut tb: Vec<&[u8]> = vec![&[]; n_us];
        let mut seen = vec![false; n_us];
        let mut p = 4usize;
        for _ in 0..n {
            if p + 2 > raw.len() {
                return Err(fail("truncated record"));
            }
            let bl = u16::from_le_bytes(raw[p..p + 2].try_into().unwrap()) as usize;
            p += 2;
            if bl == 0 || bl > 255 {
                return Err(fail("token length out of range"));
            }
            if p + bl + 4 > raw.len() {
                return Err(fail("truncated token bytes"));
            }
            let bytes = &raw[p..p + bl];
            p += bl;
            let r = u32::from_le_bytes(raw[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            if r >= n_us {
                return Err(fail("rank out of range"));
            }
            if seen[r] {
                return Err(fail("duplicate rank"));
            }
            seen[r] = true;
            tb[r] = bytes;
        }
        // Optional merge section: `SPL1` then one byte per id, the length of the
        // left part of that token's merge (0 = no merge). Vocabs whose merge list
        // isn't the one rank order implies (Qwen3.5) ship it, and a token with
        // no merge is one their BPE can never produce, so it is never matched.
        // Then optionally `MRG1`, u32 count, and per merge in list order one
        // u32: merged id | left part's byte length << 24. Merges then rank by
        // that list, not by id (embeddinggemma-2), and SPL1's 255 marks a
        // starting symbol (a character, which can be several bytes).
        let mut hsplit: Option<&[u8]> = None;
        let mut prk: Option<&[u8]> = None;
        if p != raw.len() {
            if raw.len() - p < 4 + n_us || &raw[p..p + 4] != b"SPL1" {
                return Err(fail("trailing bytes"));
            }
            hsplit = Some(&raw[p + 4..p + 4 + n_us]);
            let q = p + 4 + n_us;
            if q != raw.len() {
                if raw.len() - q < 8 || &raw[q..q + 4] != b"MRG1" {
                    return Err(fail("trailing bytes"));
                }
                let cnt = u32::from_le_bytes(raw[q + 4..q + 8].try_into().unwrap()) as usize;
                if raw.len() - q - 8 != cnt * 4 || cnt + n_us > PL_VALMASK as usize {
                    return Err(fail("bad merge list"));
                }
                prk = Some(&raw[q + 8..]);
            }
        }
        let base = |id: usize| prk.is_some() && hsplit.is_some_and(|h| h[id] == 255);
        let reachable =
            |id: usize| hsplit.map_or(true, |h| h[id] != 0 || (prk.is_none() && tb[id].len() == 1));

        let mut v = Vocab {
            all: Vec::new(),
            tstart: Vec::with_capacity(n_us + 1),
            tlen: Vec::new(),
            n,
            b2id: HashMap::with_capacity(n_us * 2),
            etab: Vec::new(),
            emask: 0,
            enodes: 0,
            root_child: [0; 256],
            r2node: Vec::new(),
            r2best: Vec::new(),
            tnode_tok: Vec::new(),
            npm: Vec::new(),
            split: Vec::new(),
            order: Vec::new(),
            pair_rank: prk.is_some(),
            fraw: Vec::new(),
            fout: Vec::new(),
            byte_tok: Vec::new(),
            nbase: 0,
            plk: Vec::new(),
            plmask: 0,
            ivm: Vec::new(),
            ivmask: 0,
            ivmw: Vec::new(),
            wide_ids: false,
            e2: Vec::new(),
            e2mask: 0,
            e2tb: 0,
            otab: Vec::new(),
            omask: 0,
            r3node: Vec::new(),
            r3best: Vec::new(),
        };

        // all/tstart + b2id, by id order
        v.tstart.push(0);
        for id in 0..n_us {
            v.all.extend_from_slice(tb[id]);
            v.tstart.push(v.all.len() as u32);
            // two ids can share bytes (a byte token and the character it is);
            // the one BPE produces wins
            let key = tb[id].to_vec().into_boxed_slice();
            if reachable(id) || !v.b2id.contains_key(&key) {
                v.b2id.insert(key, id as u32);
            }
        }
        v.tlen = (0..n_us)
            .map(|id| (v.tstart[id + 1] - v.tstart[id]) as u8)
            .collect();

        // byte trie — starts small and doubles (see `grow_etab`)
        let ecap = 1024usize;
        v.etab = vec![0u64; ecap];
        v.emask = (ecap - 1) as u32;
        v.tnode_tok = vec![RANK_MAX]; // root = node 0
        for id in 0..n_us {
            let mut node = 0u32;
            for &b in tb[id] {
                node = v.edge_build(node, b);
            }
            if reachable(id) {
                v.tnode_tok[node as usize] = id as u32;
            }
        }
        // 2-level direct table: node + deepest-token after the first <=2 bytes
        v.r2node = vec![0u32; 65536];
        v.r2best = vec![RANK_MAX; 65536];
        for b0 in 0..256usize {
            let n1 = v.root_child[b0];
            if n1 == 0 {
                continue;
            }
            let tok1 = v.tnode_tok[n1 as usize];
            for b1 in 0..256usize {
                let idx = (b0 << 8) | b1;
                let n2 = v.edge(n1, b1 as u8);
                if n2 != 0 {
                    v.r2node[idx] = n2;
                    let t2 = v.tnode_tok[n2 as usize];
                    v.r2best[idx] = if t2 != RANK_MAX { t2 } else { tok1 };
                } else {
                    v.r2node[idx] = 0;
                    v.r2best[idx] = tok1;
                }
            }
        }
        // 2-byte trie + odd-token side table, derived from the (complete) byte trie
        {
            if v.tnode_tok.len() >= (1 << 20) {
                return Err(VocabError(format!(
                    "toktok: trie too large for packed e2 (max 2^20 nodes): {what}"
                )));
            }
            let (mut ne2, mut nodd) = (0usize, 0usize);
            for id in 0..n_us {
                let l = v.tlen[id] as usize;
                if l >= 4 {
                    ne2 += (l - 2) / 2;
                }
                if l >= 3 && (l & 1) == 1 {
                    nodd += 1;
                }
            }
            let mut capo = 1024usize;
            while nodd as f64 / capo as f64 > 0.45 {
                capo <<= 1;
            }
            v.otab = vec![0u64; capo];
            v.omask = (capo - 1) as u32;
            // construction-only FULL-KEY table (packing needs the original keys,
            // which displaced open-addressing slots don't preserve)
            let mut capf = 1024usize;
            while ne2 as f64 / capf as f64 > E2_LOAD {
                capf <<= 1;
            }
            let mut ef = vec![(0u64, 0u64); capf]; // (key+1, child<<32|best)
            let fmask = (capf - 1) as u32;
            let mut path: Vec<u32> = Vec::new();
            for id in 0..n_us {
                let t = tb[id];
                let l = t.len();
                path.clear();
                path.resize(l + 1, 0);
                let mut node = 0u32;
                for j in 0..l {
                    node = if node == 0 {
                        v.root_child[t[j] as usize]
                    } else {
                        v.edge(node, t[j])
                    };
                    path[j + 1] = node;
                }
                let mut d = 2usize;
                while d + 2 <= l {
                    let k = ((path[d] as u64) << 16) | ((t[d] as u64) << 8) | t[d + 1] as u64;
                    let mut h = (k.wrapping_mul(HMUL) >> 40) as u32 & fmask;
                    while ef[h as usize].0 != 0 {
                        if ef[h as usize].0 == k + 1 {
                            break;
                        }
                        h = (h + 1) & fmask;
                    }
                    if ef[h as usize].0 == 0 {
                        let bt2 = v.tnode_tok[path[d + 2] as usize];
                        let bt1 = v.tnode_tok[path[d + 1] as usize];
                        let best = if bt2 != RANK_MAX { bt2 } else { bt1 };
                        ef[h as usize] = (k + 1, ((path[d + 2] as u64) << 32) | best as u64);
                    }
                    d += 2;
                }
                if l >= 3 && (l & 1) == 1 && reachable(id) {
                    // token ends at odd depth: side-table entry from its even parent
                    let k = ((path[l - 1] as u64) << 8) | t[l - 1] as u64;
                    let mut h = (k.wrapping_mul(HMUL) >> 40) as u32 & v.omask;
                    while v.otab[h as usize] != 0 {
                        if v.otab[h as usize] >> OTAB_KEYSH == k + 1 {
                            break;
                        }
                        h = (h + 1) & v.omask;
                    }
                    if v.otab[h as usize] == 0 {
                        v.otab[h as usize] = ((k + 1) << OTAB_KEYSH) | id as u64;
                        // hasdeeper OR'd in later
                    }
                }
            }
            // pack into the tagged final table; floor 2^17 slots so the same-tag
            // home-slot separation 2^(e2bits-11) is >= 64. After packing verify no
            // circular occupied run reaches that bound (the exactness invariant).
            let used2 = ef.iter().filter(|s| s.0 != 0).count();
            let mut want = 1usize << 17;
            while used2 as f64 / want as f64 > E2_LOAD {
                want <<= 1;
            }
            loop {
                let e2bits = want.trailing_zeros();
                v.e2 = vec![0u64; want];
                v.e2mask = (want - 1) as u32;
                v.e2tb = 36 - e2bits;
                for &(kp1, val) in ef.iter() {
                    if kp1 == 0 {
                        continue;
                    }
                    let k = kp1 - 1;
                    let child = (val >> 32) as u32;
                    let best32 = val as u32;
                    let best18 = if best32 == RANK_MAX {
                        E2BEST_NONE
                    } else {
                        best32
                    };
                    let m = mix36(k);
                    let mut j = (m >> v.e2tb) as u32;
                    while v.e2[j as usize] != 0 {
                        j = (j + 1) & v.e2mask;
                    }
                    v.e2[j as usize] =
                        E2_USED | ((m & E2_TAGMASK) << 38) | ((child as u64) << 18) | best18 as u64;
                }
                let bound = 1u64 << (e2bits - 11);
                let (mut run, mut maxrun, mut lead) = (0u64, 0u64, 0u64);
                let mut open = true;
                for i in 0..want {
                    if v.e2[i] != 0 {
                        run += 1;
                        if run > maxrun {
                            maxrun = run;
                        }
                    } else {
                        if open {
                            lead = run;
                            open = false;
                        }
                        run = 0;
                    }
                }
                if open {
                    maxrun = want as u64;
                } else if run + lead > maxrun {
                    maxrun = run + lead;
                }
                if maxrun < bound {
                    break;
                }
                want <<= 1; // never fires at sane loads; exactness insurance
            }
        }
        // next_prefix_match[id] = longest proper-prefix token (direct byte-trie walk)
        v.npm = vec![RANK_MAX; n_us];
        for id in 0..n_us {
            let t = tb[id];
            let (mut node, mut best) = (0u32, RANK_MAX);
            for j in 0..t.len().saturating_sub(1) {
                node = if node == 0 && j == 0 {
                    v.root_child[t[0] as usize]
                } else {
                    v.edge(node, t[j])
                };
                if node == 0 {
                    break;
                }
                let tk = v.tnode_tok[node as usize];
                if tk != RANK_MAX {
                    best = tk;
                }
            }
            // a starting symbol is never cut: a character in the vocab never
            // comes out as its bytes
            v.npm[id] = if base(id) { RANK_MAX } else { best };
        }
        // split_table + pair_lookup (ids in rank order)
        let mut pcap = 1usize;
        let npairs = prk.map_or(0, |x| x.len() / 4);
        while pcap < n_us.max(npairs) * 2 {
            pcap <<= 1;
        }
        v.plk = vec![0u64; pcap];
        v.plmask = (pcap - 1) as u32;
        let icap = 1usize << IVBITS;
        v.ivmask = (icap - 1) as u32;
        v.wide_ids = n > (1 << 17); // ids past 17 bits: the 34-bit mixer can't pack the pair key
        if v.wide_ids {
            v.ivmw = (0..(1usize << IVBITS_W))
                .map(|_| AtomicU32::new(0))
                .collect();
        } else {
            v.ivm = (0..icap).map(|_| AtomicU16::new(0)).collect();
        }
        v.split.reserve(n_us);
        for id in 0..n_us {
            let t = tb[id];
            if let Some(h) = hsplit {
                let l1 = h[id] as usize;
                let (a, b) = if l1 == 0 || l1 >= t.len() || base(id) {
                    (id as u32, id as u32)
                } else {
                    (v.find_id(&t[..l1]), v.find_id(&t[l1..]))
                };
                if a == RANK_MAX || b == RANK_MAX {
                    return Err(fail("bad merge"));
                }
                if a != id as u32 && prk.is_none() {
                    v.pl_put(((a as u64) << 18) | b as u64, id as u32);
                }
                v.split.push((a, b));
                continue;
            }
            let mut token1 = v.npm[id];
            let mut done = false;
            while token1 != RANK_MAX {
                let l1 = v.token_len(token1) as usize;
                let token2 = v.find_id(&t[l1..]);
                if token2 != RANK_MAX
                    && token1 < id as u32
                    && token2 < id as u32
                    && v.is_valid_token_pair(token1, token2)
                {
                    v.pl_put(((token1 as u64) << 18) | token2 as u64, id as u32);
                    v.split.push((token1, token2));
                    done = true;
                    break;
                }
                token1 = v.npm[token1 as usize];
            }
            if !done {
                v.split.push((id as u32, id as u32));
            }
        }
        if let Some(prk) = prk {
            let m = prk.len() / 4;
            let mut pairs = Vec::with_capacity(m);
            let mut mout = Vec::with_capacity(m);
            for (r, c) in prk.chunks_exact(4).enumerate() {
                let w = u32::from_le_bytes(c.try_into().unwrap());
                let (out, l) = ((w & 0xFF_FFFF) as usize, (w >> 24) as usize);
                if out >= n_us || l == 0 || l >= tb[out].len() {
                    return Err(fail("bad merge"));
                }
                let (a, b) = (v.find_id(&tb[out][..l]), v.find_id(&tb[out][l..]));
                if a == RANK_MAX || b == RANK_MAX {
                    return Err(fail("bad merge"));
                }
                let key = ((a as u64) << 18) | b as u64;
                if v.pl_get(key) != RANK_MAX {
                    return Err(fail("repeated merge"));
                }
                v.pl_put(key, r as u32);
                pairs.push((a, b));
                mout.push(out as u32);
            }
            // starting symbols rank first, then the merges as they fire
            v.order = vec![RANK_MAX; n_us];
            v.byte_tok = vec![RANK_MAX; 256];
            let mut next = 0u32;
            for id in 0..n_us {
                if base(id) {
                    v.order[id] = next;
                    next += 1;
                    if tb[id].len() == 1 {
                        v.byte_tok[tb[id][0] as usize] = id as u32;
                    }
                }
            }
            // a byte its character token covers falls back to that token
            for b in 0..128usize {
                if v.byte_tok[b] == RANK_MAX {
                    v.byte_tok[b] = v.find_id(&[b as u8]);
                }
            }
            v.nbase = next;
            let mut built = vec![RANK_MAX; m]; // merge -> the token it builds, if canonical
            for id in 0..n_us {
                let (a, b) = v.split[id];
                if a != id as u32 {
                    let r = v.pl_get(((a as u64) << 18) | b as u64);
                    if r == RANK_MAX {
                        return Err(fail("token built by a merge not in the list"));
                    }
                    built[r as usize] = id as u32;
                }
            }
            let fire = fire_order(&pairs, &built, &v.order, next);
            (v.fraw, v.fout) = (vec![0; m], vec![0; m]);
            for (r, &(a, b)) in pairs.iter().enumerate() {
                let f = fire[r];
                v.pl_put(((a as u64) << 18) | b as u64, f);
                v.fraw[(f - next) as usize] = r as u32;
                v.fout[(f - next) as usize] = mout[r];
            }
            for id in 0..n_us {
                let (a, b) = v.split[id];
                if a != id as u32 {
                    v.order[id] = v.pl_get(((a as u64) << 18) | b as u64);
                }
            }
        } else if hsplit.is_some() {
            v.order = merge_order(&v.split);
        }
        // is_valid_token_pair is only PURE once split/pair_lookup are complete;
        // construction-time calls populated the memo with stale results. Clear it.
        for s in &v.ivm {
            s.store(0, Ordering::Relaxed);
        }
        for s in &v.ivmw {
            s.store(0, Ordering::Relaxed);
        }
        // hasdeeper flags + r3 direct table
        {
            let mut haschild = vec![0u8; v.tnode_tok.len()];
            for i in 0..=v.emask as usize {
                let s = v.etab[i];
                if s != 0 {
                    haschild[(((s >> 32) - 1) >> 8) as usize] = 1;
                }
            }
            for h in 0..=v.omask as usize {
                if v.otab[h] != 0 {
                    let k = (v.otab[h] >> OTAB_KEYSH) - 1;
                    let m = v.edge((k >> 8) as u32, (k & 255) as u8);
                    if m != 0 && haschild[m as usize] == 1 {
                        v.otab[h] |= OTAB_DEEPBIT;
                    }
                }
            }
            v.r3node = vec![0u32; 1 << 16];
            v.r3best = vec![RANK_MAX; 1 << 16];
            for b0 in 0xE0usize..=0xEF {
                for b1 in 0x80usize..=0xBF {
                    let idx2 = (b0 << 8) | b1;
                    let n2 = v.r2node[idx2];
                    let bs2 = v.r2best[idx2];
                    for b2 in 0x80usize..=0xBF {
                        let i3 = ((b0 & 0xF) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
                        let mut best = bs2;
                        if n2 != 0 {
                            let m = v.edge(n2, b2 as u8);
                            if m != 0 {
                                let tk = v.tnode_tok[m as usize];
                                if tk != RANK_MAX {
                                    best = tk;
                                }
                            }
                        }
                        v.r3node[i3] = n2;
                        v.r3best[i3] = best;
                    }
                }
            }
        }
        // The byte trie has done its job (it seeded r2/r3/e2/otab/npm); nothing
        // reads it during encode, and it is the single largest table — 4 MiB for
        // cl100k, 8 MiB for o200k. Drop it.
        v.etab = Vec::new();
        v.emask = 0;
        Ok(v)
    }
}

/// When each merge fires, as a rank after the `first` starting symbols
/// (`order[id] != RANK_MAX` marks those). A merge listed before one of its
/// parts exists fires right after that part forms, ahead of anything listed
/// later. `built[r]` is the token merge `r` builds as that token's own last
/// merge. A merge whose parts never form never fires; those rank last.
fn fire_order(pairs: &[(u32, u32)], built: &[u32], order: &[u32], first: u32) -> Vec<u32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let m = pairs.len();
    let mut fire = vec![RANK_MAX; m];
    let mut made: Vec<bool> = order.iter().map(|&o| o != RANK_MAX).collect();
    let mut done = vec![false; m];
    let mut waiting: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut ready = BinaryHeap::new();
    let mut next = first;
    for r in 0..m as u32 {
        ready.push(Reverse(r));
        while let Some(Reverse(q)) = ready.pop() {
            if done[q as usize] {
                continue;
            }
            let (a, b) = pairs[q as usize];
            if let Some(p) = [a, b].into_iter().find(|&p| !made[p as usize]) {
                waiting.entry(p).or_default().push(q);
                continue;
            }
            done[q as usize] = true;
            fire[q as usize] = next;
            next += 1;
            let t = built[q as usize];
            if t != RANK_MAX && !made[t as usize] {
                made[t as usize] = true;
                if let Some(w) = waiting.remove(&t) {
                    ready.extend(w.into_iter().map(Reverse));
                }
            }
        }
    }
    for f in fire.iter_mut().filter(|f| **f == RANK_MAX) {
        *f = next;
        next += 1;
    }
    fire
}

/// Each token's merge priority, as a BPE that applies the lowest-ranked merge
/// first sees it. A merge ranked below one of its parts can only fire once that
/// part exists, and then fires straight away, ahead of anything ranked higher —
/// so it is ordered right after its later part. Identity when no merge is like
/// that.
fn merge_order(split: &[(u32, u32)]) -> Vec<u32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = split.len();
    let mut order = vec![RANK_MAX; n];
    let mut waiting: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut ready = BinaryHeap::new();
    let mut next = 0u32;
    for id in 0..n as u32 {
        if order[id as usize] != RANK_MAX {
            continue;
        }
        ready.push(Reverse(id));
        while let Some(Reverse(t)) = ready.pop() {
            if order[t as usize] != RANK_MAX {
                continue;
            }
            let (a, b) = split[t as usize];
            let pending = [a, b]
                .into_iter()
                .find(|&p| p != t && order[p as usize] == RANK_MAX);
            if let Some(p) = pending {
                waiting.entry(p).or_default().push(t);
                continue;
            }
            order[t as usize] = next;
            next += 1;
            if let Some(w) = waiting.remove(&t) {
                ready.extend(w.into_iter().map(Reverse));
            }
        }
    }
    order
}

/// Byte length of the UTF-8 character `t` starts with; 1 for a byte that
/// doesn't start a well-formed one (it then falls back to its byte token).
fn utf8_len(t: &[u8]) -> usize {
    let l = match t[0] {
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 1,
    };
    if l > t.len()
        || t[1..l].iter().any(|&b| b & 0xC0 != 0x80)
        || std::str::from_utf8(&t[..l]).is_err()
    {
        1
    } else {
        l
    }
}
