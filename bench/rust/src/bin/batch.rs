//! Parallel batch throughput + peak RSS, the axis `toktok-bench` does not cover.
use std::time::Instant;

fn peak_rss_mb() -> f64 {
    #[cfg(unix)]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        // macOS reports bytes, Linux kilobytes.
        let v = ru.ru_maxrss as f64;
        if cfg!(target_os = "macos") { v / 1048576.0 } else { v / 1024.0 }
    }
}

/// CPU seconds burned by every thread of this process so far. Wall-clock
/// throughput on a loaded machine is mostly a measure of the load; total CPU per
/// MB is what the work actually costs.
#[cfg(unix)]
fn cpu_seconds() -> f64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return f64::NAN;
        }
        let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
        t(ru.ru_utime) + t(ru.ru_stime)
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or("../corpus/pile.txt".into());
    let enc = std::env::args().nth(2).unwrap_or("cl100k_base".into());
    let threads: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let reps: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let text = std::fs::read(&path).unwrap();
    // Paragraphs, not lines: with 190k line-sized docs the work-stealing cursor
    // and thread wake-ups dominate and the tokenizer barely shows up.
    let docs: Vec<&[u8]> = text
        .split(|&b| b == b'\n')
        .filter(|d| !d.is_empty())
        .collect::<Vec<_>>()
        .chunks(16)
        .map(|c| {
            let start = c[0].as_ptr() as usize - text.as_ptr() as usize;
            let last = c[c.len() - 1];
            let end = last.as_ptr() as usize - text.as_ptr() as usize + last.len();
            &text[start..end]
        })
        .collect();
    let mb = text.len() as f64 / 1e6;
    let tok = toktok::Tokenizer::builtin(&enc).unwrap();

    let rss_load = peak_rss_mb();
    let mut best = f64::INFINITY;
    let mut best_cpu = f64::NAN;
    let mut n = 0usize;
    for _ in 0..reps {
        let (c0, t0) = (cpu_seconds(), Instant::now());
        let counts = tok.count_batch(&docs, threads, false);
        let (e, cpu) = (t0.elapsed().as_secs_f64(), cpu_seconds() - c0);
        if e < best {
            best = e;
            best_cpu = cpu;
        }
        n = counts.iter().map(|&c| c as usize).sum();
    }
    println!(
        "{:<16}{:<12} threads={:<3} {:>7.1} MB/s  CPU-s/MB {:.4}  {} docs  {} tokens  peak RSS {:>6.1} MB (load {:.1})",
        path.rsplit('/').next().unwrap(), enc, threads, mb / best, best_cpu / mb, docs.len(), n, peak_rss_mb(), rss_load
    );
}
