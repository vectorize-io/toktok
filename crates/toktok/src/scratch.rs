//! Per-thread scratch buffers for the backtracking driver (the C++ uses
//! `static thread_local std::vector`s for the same reason: no allocation in the
//! rare backtracking path).

use std::cell::RefCell;

thread_local! {
    static SCRATCH: RefCell<(Vec<u32>, Vec<u64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

thread_local! {
    /// Separate from SCRATCH on purpose: `encode_into` borrows that one for the
    /// backtracking driver, so a caller holding ids across an encode needs its own.
    static IDS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// A reusable id buffer for operations that need the whole encoding but do not
/// hand it to the caller — counting, truncating — so they allocate once per
/// thread instead of once per call.
#[inline]
pub fn with_ids<R>(f: impl FnOnce(&mut Vec<u32>) -> R) -> R {
    IDS.with(|s| f(&mut s.borrow_mut()))
}

#[inline]
pub fn with_scratch<R>(f: impl FnOnce(&mut Vec<u32>, &mut Vec<u64>) -> R) -> R {
    SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        let (toks, bf) = &mut *b;
        f(toks, bf)
    })
}
