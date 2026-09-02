//! Hammer the pretoken cache's pool from many threads and batch sizes at once,
//! and check every result against a sequential encode. The cache is per-thread
//! but pooled process-wide, threads are respawned per batch call, and the budget
//! is reserved with a CAS -- this exercises all three together.
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    let text = std::fs::read("../corpus/commoncrawl.txt").unwrap();
    let docs: Vec<&[u8]> = text
        .split(|&b| b == b'\n')
        .filter(|d| !d.is_empty())
        .take(20_000)
        .collect();

    let bad = AtomicUsize::new(0);
    let mut rounds = 0usize;
    for enc in ["cl100k_base", "o200k_base"] {
        let tok = toktok::Tokenizer::builtin(enc).unwrap();
        let want: Vec<Vec<u32>> = docs.iter().map(|d| tok.encode(d)).collect();
        let counts: Vec<u32> = want.iter().map(|v| v.len() as u32).collect();
        for threads in [0usize, 1, 2, 3, 5, 8, 13, 32] {
            for take in [1usize, 7, 100, 5000, 20_000] {
                let n = take.min(docs.len());
                let got = tok.encode_batch(&docs[..n], threads, false);
                if got[..] != want[..n] {
                    bad.fetch_add(1, Ordering::Relaxed);
                    eprintln!("MISMATCH encode_batch {enc} threads={threads} take={n}");
                }
                let c = tok.count_batch(&docs[..n], threads, false);
                if c[..] != counts[..n] {
                    bad.fetch_add(1, Ordering::Relaxed);
                    eprintln!("MISMATCH count_batch {enc} threads={threads} take={n}");
                }
                rounds += 1;
            }
        }
    }

    // Two encodings hammered concurrently from separate threads: the caches are
    // keyed by tokenizer, and a leak between them would show up as wrong ids.
    let a = toktok::Tokenizer::builtin("cl100k_base").unwrap();
    let b = toktok::Tokenizer::builtin("o200k_base").unwrap();
    let (wa, wb) = (a.encode(&text[..200_000]), b.encode(&text[..200_000]));
    std::thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(|| {
                for _ in 0..20 {
                    if a.encode(&text[..200_000]) != wa || b.encode(&text[..200_000]) != wb {
                        bad.fetch_add(1, Ordering::Relaxed);
                        eprintln!("MISMATCH concurrent cross-encoding");
                    }
                }
            });
        }
    });

    let n = bad.load(Ordering::Relaxed);
    println!("{rounds} batch rounds + 320 concurrent cross-encoding passes: {n} mismatches");
    std::process::exit(if n == 0 { 0 } else { 1 });
}
