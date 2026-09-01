//! toktok vs the popular exact Rust BPE tokenizers, on the same corpora as
//! `bench/compare.py` — no Python anywhere in the measurement.
//!
//! Encoders:
//!   toktok       this crate
//!   bpe-openai   github/rust-gems — the same backtracking algorithm, the
//!                fastest exact tokenizer we know of before quicktok
//!   tiktoken-rs  the widely used Rust port of tiktoken
//!
//! Every encoder's ids are checked against toktok's before anything is timed,
//! so a row only appears if it is byte-exact.
//!
//!   cargo run --release -- ../corpus/pile.txt cl100k_base

use std::time::Instant;

const REPS: usize = 5;

fn rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
        let pages: u64 = s.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        return pages * 4096;
    }
    #[cfg(not(target_os = "linux"))]
    {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output();
        match out {
            Ok(o) => String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .unwrap_or(0)
                * 1024,
            Err(_) => 0,
        }
    }
}

struct Row {
    label: &'static str,
    load_ms: f64,
    rss_load: i64,
    mbs: f64,
    p50: f64,
    p99: f64,
    p999: f64,
    exact: bool,
}

fn pct(sorted: &[u128], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let k = (((q * sorted.len() as f64) + 0.5).round() as usize).clamp(1, sorted.len()) - 1;
    sorted[k] as f64 / 1e3 // ns -> µs
}

/// Time one encoder: whole-corpus throughput + per-document latency percentiles.
fn measure(
    label: &'static str,
    text: &[u8],
    docs: &[&[u8]],
    load_ms: f64,
    rss_load: i64,
    encode: impl Fn(&[u8]) -> Vec<u32>,
    reference: Option<&[u32]>,
) -> Option<Row> {
    let ids = encode(text);
    let exact = reference.map(|r| r == ids.as_slice()).unwrap_or(true);
    if !exact {
        let n = reference.map(|r| r.iter().zip(&ids).take_while(|(a, b)| a == b).count());
        eprintln!("  !! {label} differs from toktok at token {n:?} — excluded");
        return None;
    }
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        let out = encode(text);
        best = best.min(t.elapsed().as_secs_f64());
        std::hint::black_box(out);
    }
    for d in docs.iter().take(50) {
        std::hint::black_box(encode(d));
    }
    let mut lat: Vec<u128> = Vec::with_capacity(docs.len());
    for d in docs {
        let t = Instant::now();
        std::hint::black_box(encode(d));
        lat.push(t.elapsed().as_nanos());
    }
    lat.sort_unstable();
    Some(Row {
        label,
        load_ms,
        rss_load,
        mbs: text.len() as f64 / 1e6 / best,
        p50: pct(&lat, 0.50),
        p99: pct(&lat, 0.99),
        p999: pct(&lat, 0.999),
        exact,
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "../corpus/pile.txt".into());
    let enc = args.next().unwrap_or_else(|| "cl100k_base".into());

    let text = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    // documents for the latency distribution, split the way the corpora are joined
    let docs: Vec<&[u8]> = text
        .split(|&b| b == b'\n')
        .filter(|d| !d.is_empty())
        .take(4000)
        .collect();
    println!(
        "\n=== {} · {enc} · {:.1} MB · {} docs · best of {REPS} ===",
        path,
        text.len() as f64 / 1e6,
        docs.len()
    );

    let mut rows = Vec::new();

    // --- toktok (the reference for exactness) ---
    let r0 = rss_bytes();
    let t0 = Instant::now();
    let tok = toktok::Tokenizer::builtin(&enc).unwrap_or_else(|e| panic!("{e}"));
    let (tok_load, tok_rss) = (t0.elapsed().as_secs_f64() * 1e3, rss_bytes() as i64 - r0 as i64);
    let reference = tok.encode(&text);
    rows.extend(measure(
        "toktok",
        &text,
        &docs,
        tok_load,
        tok_rss,
        |t| tok.encode(t),
        Some(&reference),
    ));
    println!(
        "    {} tokens · toktok tables {:.1} MiB exact",
        reference.len(),
        tok.memory_bytes() as f64 / 1048576.0
    );

    // --- bpe-openai ---
    {
        let r0 = rss_bytes();
        let t0 = Instant::now();
        let b = match enc.as_str() {
            "cl100k_base" => Some(bpe_openai::cl100k_base()),
            "o200k_base" => Some(bpe_openai::o200k_base()),
            _ => None,
        };
        let (load, rss) = (t0.elapsed().as_secs_f64() * 1e3, rss_bytes() as i64 - r0 as i64);
        if let Some(b) = b {
            rows.extend(measure(
                "bpe-openai",
                &text,
                &docs,
                load,
                rss,
                |t| b.encode(std::str::from_utf8(t).unwrap_or("")),
                Some(&reference),
            ));
        }
    }

    // --- tiktoken-rs ---
    {
        let r0 = rss_bytes();
        let t0 = Instant::now();
        let bpe = match enc.as_str() {
            "cl100k_base" => tiktoken_rs::cl100k_base().ok(),
            "o200k_base" => tiktoken_rs::o200k_base().ok(),
            _ => None,
        };
        let (load, rss) = (t0.elapsed().as_secs_f64() * 1e3, rss_bytes() as i64 - r0 as i64);
        if let Some(bpe) = bpe {
            rows.extend(measure(
                "tiktoken-rs",
                &text,
                &docs,
                load,
                rss,
                |t| bpe.encode_ordinary(std::str::from_utf8(t).unwrap_or("")),
                Some(&reference),
            ));
        }
    }

    rows.sort_by(|a, b| b.mbs.partial_cmp(&a.mbs).unwrap());
    let base = rows.last().map(|r| r.mbs).unwrap_or(1.0);
    println!(
        "\n  {:<12}  {:>8}  {:>7}  {:>9}  {:>8}  {:>9}  {:>9}  {:>9}",
        "encoder", "MB/s", "vs slow", "RSS@load", "load", "p50 µs", "p99 µs", "p99.9 µs"
    );
    for r in &rows {
        println!(
            "  {:<12}  {:8.1}  {:6.2}x  {:8.1}M  {:7.0}ms  {:9.1}  {:9.1}  {:9.1}{}",
            r.label,
            r.mbs,
            r.mbs / base,
            r.rss_load as f64 / 1048576.0,
            r.load_ms,
            r.p50,
            r.p99,
            r.p999,
            if r.exact { "" } else { "  (INEXACT)" }
        );
    }
    println!("      RSS@load is cumulative in one process — each encoder loads after the previous");
}
