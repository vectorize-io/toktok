//! Byte-exactness against tiktoken, via the fixtures in tests/vectors_*.bin
//! (regenerate with `python tools/gen_vectors.py`).

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn load(encoding: &str) -> toktok::Tokenizer {
    toktok::Tokenizer::builtin(encoding).unwrap()
}

fn rd_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

fn check(encoding: &str) {
    let tok = load(encoding);
    let blob = std::fs::read(root().join(format!("tests/vectors_{encoding}.bin"))).unwrap();
    let mut p = 0usize;
    let n = rd_u32(&blob, &mut p);
    for case in 0..n {
        let tl = rd_u32(&blob, &mut p) as usize;
        let text = &blob[p..p + tl];
        p += tl;
        let ni = rd_u32(&blob, &mut p) as usize;
        let want: Vec<u32> = (0..ni).map(|_| rd_u32(&blob, &mut p)).collect();
        let got = tok.encode(text);
        assert_eq!(
            got,
            want,
            "{encoding} case {case}: {:?}",
            String::from_utf8_lossy(text)
        );
        assert_eq!(tok.decode(&got), text, "{encoding} roundtrip case {case}");
    }
}

#[test]
fn cl100k_matches_tiktoken() {
    check("cl100k_base");
}

#[test]
fn o200k_matches_tiktoken() {
    check("o200k_base");
}

#[test]
fn harmony_shares_o200k_ranks_and_adds_specials() {
    let o = load("o200k_base");
    let h = load("o200k_harmony");
    assert_eq!(o.encode(b"hello world"), h.encode(b"hello world"));
    assert!(h.special_tokens().len() > o.special_tokens().len());
    assert!(h.special_tokens().iter().any(|(s, _)| s == "<|message|>"));
}

#[test]
fn specials_and_invalid_utf8() {
    let tok = load("cl100k_base");
    let ids = tok.encode_with_special("a<|endoftext|>b");
    assert!(ids.contains(&100257));
    assert_eq!(tok.decode(&ids), b"a<|endoftext|>b");
    // encode_ordinary must NOT turn a special string into its id
    assert!(!tok.encode(b"a<|endoftext|>b").contains(&100257));
    // lone continuation / truncated sequences must still round-trip byte-exactly
    for bad in [
        &b"\xE4\xB8"[..],
        &b"\x80\x80"[..],
        &b"abc\xE4\xB8\x2D"[..],
        &b"\xF0\x9F\x9A"[..],
    ] {
        assert_eq!(tok.decode(&tok.encode(bad)), bad);
    }
}

#[test]
fn offsets_tile_the_input() {
    let tok = load("cl100k_base");
    let text = "Hello, 日本語 world! 123\n\n  indented".as_bytes();
    let (ids, bounds) = tok.encode_with_offsets(text);
    assert_eq!(bounds.len(), ids.len() + 1);
    assert_eq!(*bounds.last().unwrap() as usize, text.len());
    for (i, w) in bounds.windows(2).enumerate() {
        assert_eq!(
            tok.token_bytes(ids[i]).unwrap(),
            &text[w[0] as usize..w[1] as usize]
        );
    }
}

#[test]
fn batch_matches_sequential() {
    let tok = load("cl100k_base");
    let texts: Vec<String> = (0..500)
        .map(|i| format!("doc {i}: hello 日本語 world {i}\n"))
        .collect();
    let refs: Vec<&[u8]> = texts.iter().map(|s| s.as_bytes()).collect();
    let got = tok.encode_batch(&refs, 8, false);
    for (i, r) in refs.iter().enumerate() {
        assert_eq!(got[i], tok.encode(r));
    }
}

#[test]
fn count_batch_matches_encode() {
    let tok = load("cl100k_base");
    let texts: Vec<String> = (0..300)
        .map(|i| format!("doc {i}: hello 日本語 world {i}\n"))
        .collect();
    let refs: Vec<&[u8]> = texts.iter().map(|s| s.as_bytes()).collect();
    for threads in [1, 8] {
        let counts = tok.count_batch(&refs, threads, false);
        for (i, r) in refs.iter().enumerate() {
            assert_eq!(counts[i] as usize, tok.encode(r).len());
        }
    }
    // with_special: the special string collapses to one id
    let with_sp: Vec<&[u8]> = vec![b"a<|endoftext|>b"];
    assert_eq!(
        tok.count_batch(&with_sp, 1, true)[0] as usize,
        tok.encode_with_special("a<|endoftext|>b").len()
    );
    assert!(tok.count_batch(&[], 4, false).is_empty());
}

#[test]
fn memory_accounting_is_consistent() {
    let tok = load("cl100k_base");
    let total = tok.memory_bytes();
    let parts: usize = tok.vocab().memory_breakdown().iter().map(|&(_, n)| n).sum();
    // the breakdown covers the vocab tables; the tokenizer adds class tables
    assert!(parts <= total && parts > total / 2, "{parts} vs {total}");
    assert!(total > 8 << 20 && total < 32 << 20, "{total}");
    assert!(load("o200k_base").memory_bytes() > total);
}

#[test]
fn truncate_matches_encode_slice_decode() {
    let tok = load("cl100k_base");
    let cases: Vec<&[u8]> = vec![
        b"hello world",
        b"",
        "The quick brown fox jumps over the lazy dog, repeatedly and at length.".as_bytes(),
        "日本語のテキストです。これはテストです。".as_bytes(),
        "mixed ASCII and 日本語 and emoji 🚀🎉 and more text".as_bytes(),
        "\u{1F600}\u{1F601}\u{1F602}\u{1F603}".as_bytes(),
    ];
    for text in cases {
        let all = tok.encode(text);
        for max in 0..=all.len() + 2 {
            let t = tok.truncate(text, max);
            assert_eq!(t.total_tokens, all.len(), "total for max={max}");

            // the kept prefix is exactly what encode/slice/decode would give,
            // except we never cut a character in half
            let reference = tok.decode(&all[..all.len().min(max)]);
            let got = &text[..t.bytes];
            if max >= all.len() {
                assert_eq!(got, &reference[..], "no truncation expected at max={max}");
            } else {
                assert!(
                    reference.starts_with(got),
                    "truncated prefix diverges at max={max}"
                );
                // the only difference is the partial character we declined to
                // emit — at most 3 bytes, never any complete character
                assert!(
                    reference.len() - got.len() < 4,
                    "dropped {} bytes at max={max}, expected at most a partial character",
                    reference.len() - got.len()
                );
            }
            assert!(
                std::str::from_utf8(got).is_ok(),
                "truncation left invalid UTF-8 at max={max}"
            );
            // kept_tokens counts whole tokens of the ORIGINAL encoding inside the
            // kept bytes. Re-encoding the prefix can differ — a shortened tail
            // tokenizes on its own terms — which is equally true of the
            // decode(encode(x)[:n]) pattern this replaces.
            assert!(t.kept_tokens <= max.min(all.len()));
            let kept_bytes: usize = all[..t.kept_tokens]
                .iter()
                .map(|&id| tok.token_bytes(id).unwrap().len())
                .sum();
            assert!(
                kept_bytes <= t.bytes,
                "kept_tokens overruns the cut at max={max}"
            );
        }
    }
}

#[test]
fn truncate_cuts_on_a_character_boundary() {
    let tok = load("cl100k_base");
    // a character whose bytes span more than one token: cutting between them
    // must not produce invalid UTF-8
    let text = "🚀🚀🚀🚀".as_bytes();
    let all = tok.encode(text);
    assert!(all.len() > 4, "expected the emoji to span several tokens");
    for max in 1..all.len() {
        let t = tok.truncate(text, max);
        let kept = &text[..t.bytes];
        assert!(
            std::str::from_utf8(kept).is_ok(),
            "invalid UTF-8 at max={max}"
        );
        assert_eq!(t.bytes % 4, 0, "cut inside a 4-byte character at max={max}");
    }
}

#[test]
fn unknown_encoding_errors() {
    assert!(toktok::Tokenizer::builtin("nope").is_err());
    // the directory loader agrees, and still works for supplying your own data
    assert!(toktok::Tokenizer::load_dir(root().join("crates/toktok/data"), "nope").is_err());
    assert!(toktok::Tokenizer::load_dir(root().join("crates/toktok/data"), "cl100k_base").is_ok());
}
