//! Bounded per-thread memoization of short pretokens.
//!
//! The backtracking encoder in [`crate::vocab`] is exact but recomputes every
//! pretoken from scratch, and natural text repeats pretokens hard: on this
//! repo's corpora 94-99% of pretokens are <= 15 bytes and only 0.12-0.38 M of
//! them are distinct per 25 MB. Memoizing that Zipf head turns almost every
//! pretoken into one hash probe.
//!
//! The idea (and the value packing below) is ported from
//! [gigatoken](https://github.com/marcelroed/gigatoken) (MIT), which measured
//! the same distribution at web scale. Two deliberate departures:
//!
//! * **Bounded, never grown.** gigatoken sizes per worker from a Heaps'-law
//!   estimate and clamps at 2^22 slots (134 MB *per worker*). Measured on this
//!   repo's corpora, distinct short pretokens saturate around 0.4 M, so
//!   everything above ~2^18 buys nothing; capping instead of growing also means
//!   no rehash storms and a footprint that cannot surprise a caller. Eviction is
//!   safe here because this is a cache, not a memo of record — a lost entry
//!   costs one re-encode, never a wrong answer.
//! * **No spill arena.** Values that do not fit inline (> 4 tokens, ~1-2% of
//!   pretokens) are simply not cached, rather than parked in a side arena with
//!   its own lifetime and wipe rules. They take the ordinary encoder path.
//!
//! The table is allocated zeroed, and an all-zero entry *is* the empty state, so
//! a fresh table costs virtual address space and nothing resident until entries
//! are actually inserted. A thread that encodes one short string never
//! materializes 4 MB.

use crate::vocab::Vocab;

/// Longest pretoken that fits a packed key (16 bytes minus the length tag).
pub const KEY_MAX: usize = 15;

/// Largest table one cache may take: 2^18 slots x 32 B = 8 MiB. Measured on
/// this repo's corpora, throughput rises steeply to 2^18 and is flat at 2^19,
/// and top-N coverage saturates in the same place — so this is the knee, not a
/// guess.
const MAX_BITS: u32 = 19;
/// Smallest table worth keeping. Below this the hit rate falls off faster than
/// the memory saved is worth; a thread that cannot get at least this much runs
/// uncached instead.
const MIN_BITS: u32 = 15;

/// Total pretoken-cache bytes this process will allocate, across every thread
/// and every encoding.
///
/// A *process* budget, deliberately: gigatoken hands each worker the full
/// budget independently (`fork_sized` clamps at 2^22 slots = 134 MB **per
/// worker**), so its footprint scales with core count. Here 14 workers share
/// one ceiling, and a thread that arrives after it is exhausted encodes
/// uncached rather than pushing the process further.
const BUDGET_BYTES: usize = 32 << 20;

/// Pairs of slots probed before giving up and evicting. Two pairs = four slots =
/// two cache lines; the third probe is worth less than the eviction it avoids.
const PAIRS: usize = 3;

/// `val` bit 7: this entry's tokens did not fit inline. Never stored — the
/// insert path drops such pretokens instead — but the encode loop still tests it
/// so a future spilling variant cannot silently emit garbage.
const VAL_SPILL: u64 = 0x80;
const COUNT_MASK: u64 = 0x7F;

/// One slot: packed pretoken key plus its packed encoding. Exactly 32 bytes, so
/// two slots share a cache line and never straddle one.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Entry {
    key: u128,
    val: u64,
    ext: u64,
}

const EMPTY: Entry = Entry {
    key: 0,
    val: 0,
    ext: 0,
};

/// Pack a pretoken's bytes into a `u128` key, length tagged into the top byte so
/// that `"a"` and `"a\0"` cannot collide and no real key is ever zero (zero is
/// the empty-slot sentinel).
///
/// Returns `None` for anything longer than [`KEY_MAX`], which the caller routes
/// to the uncached path.
#[inline(always)]
pub fn pack_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n == 0 || n > KEY_MAX {
        return None;
    }
    let p = bytes.as_ptr();
    // A 16-byte read starting <= 4096-16 into a page cannot leave that page, and
    // the page is mapped because `p` points at >= 1 valid byte. Near a page tail
    // fall back to a copy — correctness over speed on a path taken once every
    // few hundred pretokens.
    let raw = if (p as usize) & 4095 <= 4096 - 16 {
        // SAFETY: bounded above; see comment.
        let v = unsafe { (p as *const u128).read_unaligned() };
        // Table lookup, not `(1 << n*8) - 1`: a variable 128-bit shift is a
        // multi-instruction sequence on both targets, and this is the single
        // hottest line in the encoder.
        v & unsafe { *PACK_MASK.get_unchecked(n) }
    } else {
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(bytes);
        u128::from_le_bytes(lanes)
    };
    Some(raw | ((n as u128) << 120))
}

/// `PACK_MASK[n]` keeps the low `n` bytes. Index 16 is unused (`n <= 15`) but
/// keeps the table a power of two.
static PACK_MASK: [u128; 17] = {
    let mut m = [0u128; 17];
    let mut i = 1;
    while i <= 16 {
        m[i] = if i == 16 {
            u128::MAX
        } else {
            (1u128 << (i * 8)) - 1
        };
        i += 1;
    }
    m
};

/// Index hash. Deliberately shallow: the whole chain from the pretoken's bytes
/// to the probe load — read, mask, hash, index, load — is serial and on the
/// critical path of every pretoken, so latency beats avalanche quality. Two
/// multiplies that issue in parallel and one xor cost ~5 cycles; a
/// splitmix-style finisher costs three *dependent* multiplies and measured
/// slower despite a better distribution.
///
/// The result's high bits carry the mixing (multiply spreads entropy upward),
/// so callers index with `>> shift`, never `& mask`.
#[inline(always)]
fn hash(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    lo.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ hi.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

/// Pack up to four ids into `(val, ext)`.
///
/// `val` low byte is the count; token 1 occupies bits 8-31 (24 bits, which every
/// id in cl100k's 100,256 and o200k's 199,998 fits) and token 2 bits 32-63;
/// tokens 3 and 4 are `ext`'s two `u32` lanes. `None` when the ids do not fit,
/// which the caller treats as "do not cache this one".
#[inline(always)]
fn pack_val(ids: &[u32]) -> Option<(u64, u64)> {
    match *ids {
        [a] if a < (1 << 24) => Some((1 | ((a as u64) << 8), 0)),
        [a, b] if a < (1 << 24) => Some((2 | ((a as u64) << 8) | ((b as u64) << 32), 0)),
        [a, b, c] if a < (1 << 24) => Some((3 | ((a as u64) << 8) | ((b as u64) << 32), c as u64)),
        [a, b, c, d] if a < (1 << 24) => Some((
            4 | ((a as u64) << 8) | ((b as u64) << 32),
            c as u64 | ((d as u64) << 32),
        )),
        _ => None,
    }
}

/// One thread's cache for one tokenizer.
pub struct PretokenCache {
    slots: Box<[Entry]>,
    /// `64 - log2(slots.len())`: how far to shift a hash down to an index.
    /// Held rather than derived so the hot path does not reload the slice
    /// length. 64 marks a bypass cache (see [`PretokenCache::disabled`]) — a
    /// shift of 64 is not a valid index computation, so the hot path tests it
    /// before use.
    shift: u32,
    /// Which tokenizer built these entries — ids from one vocab are meaningless
    /// to another, so a mismatch must never be served.
    owner: u64,
}

impl PretokenCache {
    fn new(owner: u64, bits: u32) -> Self {
        // A `vec![]` of an all-zero element goes through the zeroing allocator,
        // i.e. calloc: untouched slots cost address space and no resident page.
        // A thread that encodes one short string never materializes the table.
        let n = 1usize << bits;
        Self {
            slots: vec![EMPTY; n].into_boxed_slice(),
            shift: 64 - bits,
            owner,
        }
    }

    /// A cache that always misses, for threads that arrive after the process
    /// budget is spent. Allocates nothing.
    fn disabled(owner: u64) -> Self {
        Self {
            slots: Box::new([]),
            shift: 64,
            owner,
        }
    }

    fn bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<Entry>()
    }

    /// Append `piece`'s ids to `out`, encoding and memoizing on a miss.
    ///
    /// The probe happens *before* any vocab walk. That ordering is the whole
    /// point: `next_match` is the dominant cost of encoding a pretoken (for the
    /// ~90% that are a single token it *is* the encode), so a cache consulted
    /// after it would save only the backtracking tail and pay for itself in
    /// nothing.
    #[inline(always)]
    pub fn emit(&mut self, v: &Vocab, piece: &[u8], out: &mut Vec<u32>) {
        let Some(key) = pack_key(piece) else {
            v.encode(piece, out);
            return;
        };
        if self.shift == 64 {
            v.encode(piece, out);
            return;
        }
        self.probe_emit(v, piece, out, key, hash(key));
    }

    /// Probe for `key` and append the result, encoding on a miss.
    #[inline(always)]
    fn probe_emit(&mut self, v: &Vocab, piece: &[u8], out: &mut Vec<u32>, key: u128, h: u64) {
        let base = (h >> self.shift) as usize & !1;
        // The home pair is one cache line and holds the overwhelming majority of
        // hits, so it is tested here, straight-line, and everything else is a
        // call. Keeping the rest of the walk out of this function is what lets
        // the scanner's loop stay small.
        //
        // SAFETY: `base` is an even index produced by shifting the hash down to
        // `log2(len)` bits, and the table always holds at least two slots, so
        // both `base` and `base + 1` are in range.
        let (e0, e1) = unsafe {
            (
                *self.slots.get_unchecked(base),
                *self.slots.get_unchecked(base + 1),
            )
        };
        let (val, ext) = if e0.key == key {
            (e0.val, e0.ext)
        } else if e1.key == key {
            (e1.val, e1.ext)
        } else {
            return self.probe_cold(v, piece, out, key, h);
        };
        if val & VAL_SPILL != 0 {
            return self.probe_cold(v, piece, out, key, h);
        }
        emit_packed(out, val, ext);
    }

    /// Everything the home pair did not answer: the rest of the probe walk, and
    /// the encode-and-install when that comes up empty too.
    #[cold]
    #[inline(never)]
    fn probe_cold(&mut self, v: &Vocab, piece: &[u8], out: &mut Vec<u32>, key: u128, h: u64) {
        let wrap = self.slots.len() - 1;
        let base = (h >> self.shift) as usize & !1;
        for pair in 1..PAIRS {
            let idx = (base + pair * 2) & wrap;
            for k in 0..2 {
                let e = self.slots[idx + k];
                if e.key == key && e.val & VAL_SPILL == 0 {
                    emit_packed(out, e.val, e.ext);
                    return;
                }
            }
        }
        let start = out.len();
        let first = v.next_match(piece);
        v.encode_with_first(piece, first, out);
        let Some((val, ext)) = pack_val(&out[start..]) else {
            return; // > 4 tokens: rare, left uncached rather than spilled
        };
        // Prefer an empty slot anywhere in the probe window; only evict when the
        // whole window is occupied. Evicting the first slot of the home pair
        // keeps the just-used entry on the line the next probe will load.
        let mut victim = base;
        for pair in 0..PAIRS {
            let idx = (base + pair * 2) & wrap;
            for k in 0..2 {
                if self.slots[idx + k].key == 0 {
                    victim = idx + k;
                    self.slots[victim] = Entry { key, val, ext };
                    return;
                }
            }
        }
        self.slots[victim] = Entry { key, val, ext };
    }
}

/// Append a packed value's `1..=4` ids.
///
/// All four lanes are stored unconditionally and the length advanced by the real
/// count: two `u64` stores, no branch on the count, and none of
/// `extend_from_slice`'s capacity dance. Lanes past the count are another key's
/// leftovers, written above the length and never read — `encode_into` reserves
/// the slack that makes that legal.
#[inline(always)]
fn emit_packed(out: &mut Vec<u32>, val: u64, ext: u64) {
    let n = (val & COUNT_MASK) as usize;
    let len = out.len();
    if len + 4 <= out.capacity() {
        // Lanes 1-2 are already adjacent in `val`'s high bits; only token 1
        // needs masking out of its 24-bit field.
        let ab = ((val >> 8) & 0x00FF_FFFF) | (val & 0xFFFF_FFFF_0000_0000);
        // SAFETY: capacity checked above; `out` is `u32`, so the four lanes are
        // 16 bytes and stay inside the allocation.
        unsafe {
            let q = out.as_mut_ptr().add(len);
            (q as *mut u64).write_unaligned(ab);
            (q.add(2) as *mut u64).write_unaligned(ext);
            out.set_len(len + n);
        }
    } else {
        let lanes = [
            ((val >> 8) & 0x00FF_FFFF) as u32,
            (val >> 32) as u32,
            ext as u32,
            (ext >> 32) as u32,
        ];
        out.extend_from_slice(&lanes[..n]);
    }
}

/// Live table bytes across the whole process, pooled or checked out.
static LIVE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Caches whose thread has exited, kept for the next thread to claim.
///
/// `count_batch`/`encode_batch` open a fresh `std::thread::scope` per call, so
/// without this every batch call would allocate and zero a table per worker and
/// throw away a warm one at the end. The pool makes the tables outlive the
/// threads that filled them: the second batch call reuses the first's, already
/// warm.
static POOL: std::sync::Mutex<Vec<PretokenCache>> = std::sync::Mutex::new(Vec::new());

/// Per-cache ceiling for a caller that will run `workers` threads concurrently.
///
/// The caller states its own concurrency rather than the table guessing from
/// `available_parallelism`: a single-threaded `encode` should get the whole
/// budget, while a 14-worker `count_batch` must divide it — otherwise the first
/// worker to arrive takes the ceiling and the other thirteen run uncached.
fn share_bits(workers: usize) -> u32 {
    let share = BUDGET_BYTES / workers.max(1);
    let mut b = MAX_BITS;
    while b > MIN_BITS && (1usize << b) * std::mem::size_of::<Entry>() > share {
        b -= 1;
    }
    b
}

/// Reserve budget for a new cache and return its size in bits, or `None` when
/// the process ceiling leaves no room — that thread then runs uncached.
///
/// The reservation is a CAS loop rather than a read-then-add: `thread::scope`
/// starts every batch worker at once, and a racy check let all of them observe
/// an empty budget and allocate the maximum table apiece.
fn reserve_bits(workers: usize) -> Option<u32> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut live = LIVE_BYTES.load(Relaxed);
    loop {
        let cap = share_bits(workers);
        let left = BUDGET_BYTES.saturating_sub(live);
        let mut b = cap;
        while (1usize << b) * std::mem::size_of::<Entry>() > left {
            if b == MIN_BITS {
                return None;
            }
            b -= 1;
        }
        let want = (1usize << b) * std::mem::size_of::<Entry>();
        match LIVE_BYTES.compare_exchange_weak(live, live + want, Relaxed, Relaxed) {
            Ok(_) => return Some(b),
            Err(cur) => live = cur,
        }
    }
}

/// Claim a cache for `owner`: a pooled one if a matching table is waiting,
/// otherwise a fresh one sized by the remaining budget.
fn acquire(owner: u64, workers: usize) -> PretokenCache {
    if let Ok(mut pool) = POOL.lock() {
        if let Some(i) = pool.iter().position(|c| c.owner == owner) {
            return pool.swap_remove(i);
        }
        // A cache for a different tokenizer is not reusable — its entries are
        // another vocab's ids — but its budget is. Retiring one here keeps a
        // process that switches encodings from paying the ceiling twice.
        while LIVE_BYTES.load(std::sync::atomic::Ordering::Relaxed) >= BUDGET_BYTES {
            let Some(c) = pool.pop() else { break };
            LIVE_BYTES.fetch_sub(c.bytes(), std::sync::atomic::Ordering::Relaxed);
            drop(c);
        }
    }
    match reserve_bits(workers) {
        Some(bits) => PretokenCache::new(owner, bits),
        None => PretokenCache::disabled(owner),
    }
}

/// Hand caches back when a worker thread exits, so the next one starts warm.
struct Holder(Vec<PretokenCache>);

impl Drop for Holder {
    fn drop(&mut self) {
        let Ok(mut pool) = POOL.lock() else { return };
        for c in self.0.drain(..) {
            if c.bytes() == 0 {
                continue; // a bypass cache owns nothing worth pooling
            }
            pool.push(c);
        }
    }
}

thread_local! {
    /// One cache per (thread, tokenizer). Two covers the realistic worst case —
    /// a process alternating cl100k and o200k — without the bookkeeping of a
    /// real eviction policy; a third encoding recycles the older of the two.
    static CACHES: std::cell::RefCell<Holder> = const {
        std::cell::RefCell::new(Holder(Vec::new()))
    };
}

const MAX_CACHES: usize = 2;

/// Run `f` with this thread's cache for tokenizer `owner`, claiming one on first
/// use. The cache outlives the call — and, via the pool, the thread.
#[inline]
pub fn with_cache<R>(owner: u64, workers: usize, f: impl FnOnce(&mut PretokenCache) -> R) -> R {
    CACHES.with(|c| {
        // No current caller encodes from inside an encode, but a re-entrant one
        // would find the cell already borrowed, and panicking a tokenizer over a
        // cache is the wrong trade: run that call uncached instead.
        let Ok(mut h) = c.try_borrow_mut() else {
            return f(&mut PretokenCache::disabled(owner));
        };
        let pos = match h.0.iter().position(|c| c.owner == owner) {
            Some(i) => i,
            None => {
                if h.0.len() == MAX_CACHES {
                    let old = h.0.remove(0);
                    if let Ok(mut pool) = POOL.lock() {
                        if old.bytes() != 0 {
                            pool.push(old);
                        }
                    }
                }
                let fresh = acquire(owner, workers);
                h.0.push(fresh);
                h.0.len() - 1
            }
        };
        f(&mut h.0[pos])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_length_tagged_and_never_zero() {
        assert_ne!(pack_key(b"a").unwrap(), pack_key(b"a\0").unwrap());
        assert_ne!(pack_key(b"a").unwrap(), 0);
        assert_eq!(pack_key(b""), None);
        assert_eq!(pack_key(&[0u8; 16]), None);
        assert!(pack_key(&[0u8; 15]).is_some());
    }

    #[test]
    fn key_packing_survives_a_page_boundary() {
        // Straddle the fallback: a pretoken whose last byte is the page's last.
        let page = 4096;
        let mut buf = vec![0u8; page * 2];
        for n in 1..=KEY_MAX {
            let off = page - n;
            for (i, b) in buf[off..off + n].iter_mut().enumerate() {
                *b = b'a' + i as u8;
            }
            let at_edge = pack_key(&buf[off..off + n]).unwrap();
            let copied = pack_key(&buf[off..off + n]).unwrap();
            assert_eq!(at_edge, copied, "n={n}");
        }
    }

    #[test]
    fn val_packing_round_trips() {
        for ids in [
            vec![1u32],
            vec![1, 2],
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
            vec![100_255, 199_997, 0, 7],
        ] {
            let (val, ext) = pack_val(&ids).unwrap();
            let n = (val & COUNT_MASK) as usize;
            let lanes = [
                ((val >> 8) & 0x00FF_FFFF) as u32,
                (val >> 32) as u32,
                ext as u32,
                (ext >> 32) as u32,
            ];
            assert_eq!(&lanes[..n], &ids[..]);
        }
        assert_eq!(pack_val(&[1, 2, 3, 4, 5]), None);
    }
}
