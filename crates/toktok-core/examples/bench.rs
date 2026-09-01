//! Native throughput bench — the apples-to-apples counterpart of quicktok's
//! `bench/bench_file.cpp` (no Python object marshalling in the measurement).
//!
//!   cargo run --release --example bench -- bench/corpus/pile.txt cl100k_base [reps]

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "bench/corpus/pile.txt".into());
    let enc = args.next().unwrap_or_else(|| "cl100k_base".into());
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let t0 = Instant::now();
    let tok = toktok_core::Tokenizer::load_dir("python/toktok/data", &enc)
        .unwrap_or_else(|e| panic!("{e}"));
    let load = t0.elapsed();

    let text = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mb = text.len() as f64 / 1e6;

    let mut out = Vec::with_capacity(text.len() / 3 + 8);
    let mut best = f64::INFINITY;
    let mut ntok = 0;
    for _ in 0..reps {
        out.clear();
        let t = Instant::now();
        tok.encode_into(&text, &mut out);
        best = best.min(t.elapsed().as_secs_f64());
        ntok = out.len();
    }
    println!(
        "{path} · {enc} · {mb:.1} MB · {ntok} tokens · load {:.0} ms\n  {:.1} MB/s (best of {reps})",
        load.as_secs_f64() * 1e3,
        mb / best,
    );
}
