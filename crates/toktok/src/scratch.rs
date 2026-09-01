//! Per-thread scratch buffers for the backtracking driver (the C++ uses
//! `static thread_local std::vector`s for the same reason: no allocation in the
//! rare backtracking path).

use std::cell::RefCell;

thread_local! {
    static SCRATCH: RefCell<(Vec<u32>, Vec<u64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

#[inline]
pub fn with_scratch<R>(f: impl FnOnce(&mut Vec<u32>, &mut Vec<u64>) -> R) -> R {
    SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        let (toks, bf) = &mut *b;
        f(toks, bf)
    })
}
