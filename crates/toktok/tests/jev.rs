//! The Jev estimate against a few counts TypeSafe's API reported (the full set,
//! ~440 texts, is checked from Python: tests/test_jev.py), and its truncate.

use toktok::jev;

const KNOWN: &[(&str, u32)] = &[
    ("hello world", 2),
    ("helloworld", 4),
    ("1234567890", 10),
    ("I'm don't you'll we've", 8),
    ("Привет, как дела?", 10),
    ("日本語のテキストです。", 11),
    ("emoji 🚀🔥👍🏽 test", 11),
    ("<|endoftext|>", 7),
    ("https://example.com/path?q=1&b=2", 12),
    ("naïve café résumé", 5),
];

#[test]
fn never_under_known_counts() {
    for &(text, n) in KNOWN {
        let e = jev::count(text.as_bytes());
        assert!(e >= n, "{text:?}: estimate {e} < jev {n}");
    }
    assert_eq!(jev::count(b""), 0);
}

#[test]
fn batch_matches_single() {
    let texts: Vec<&[u8]> = KNOWN.iter().map(|(t, _)| t.as_bytes()).collect();
    let single: Vec<u32> = texts.iter().map(|t| jev::count(t)).collect();
    assert_eq!(jev::count_batch(&texts, 1), single);
    assert_eq!(jev::count_batch(&texts, 0), single);
}

#[test]
fn truncate_fits_and_cuts_on_char_boundaries() {
    let text = "Привет 🚀 hello world, ".repeat(50);
    let t = text.as_bytes();
    let total = jev::count(t) as usize;
    for max in [0, 1, 5, 10, 50, 200, total, total + 1] {
        let cut = jev::truncate(t, max);
        assert_eq!(cut.total_tokens, total);
        assert!(text.is_char_boundary(cut.bytes));
        assert_eq!(cut.kept_tokens, jev::count(&t[..cut.bytes]) as usize);
        assert!(
            cut.kept_tokens <= max || cut.bytes == 0,
            "max {max}: {cut:?}"
        );
        if total <= max {
            assert_eq!(cut.bytes, t.len());
        }
    }
    let batch = jev::truncate_batch(&[t, b"hi"], 10, 0);
    assert_eq!(batch, [jev::truncate(t, 10), jev::truncate(b"hi", 10)]);
}
