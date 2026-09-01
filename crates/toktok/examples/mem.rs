fn main() {
    for enc in ["cl100k_base", "o200k_base"] {
        let t = toktok::Tokenizer::builtin(enc).unwrap();
        println!("{enc}: total {:.1} MiB", t.memory_bytes() as f64 / 1048576.0);
        for (k, n) in t.vocab().memory_breakdown() {
            if n > 0 { println!("   {k:<20} {:6.2} MiB", n as f64 / 1048576.0); }
        }
    }
}
