#!/usr/bin/env python3
"""Cross-encoder throughput benchmark: toktok vs quicktok (the C++ original) vs
tiktoken, on the corpora fetched by `fetch_corpus.py`.

Every encoder's output is verified token-for-token identical before any timing —
a mismatch is reported and that encoder is dropped from the run.

    python bench/compare.py                       # all corpora, both encodings
    python bench/compare.py --enc cl100k_base --corpus pile
    python bench/compare.py --threads 8           # batch/parallel scaling too
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")


def encoders(enc_name):
    """(label, encode_fn) for every encoder available in this environment."""
    out = []
    import toktok

    tt = toktok.get_encoding(enc_name)
    out.append(("toktok", lambda s: tt.encode_ordinary(s)))
    try:
        import numpy as np  # noqa: F401

        out.append(("toktok (numpy)", lambda s: toktok.encode_to_numpy(tt, s)))
    except ImportError:
        pass
    try:
        import quicktok

        qt = quicktok.get_encoding(enc_name)
        out.append(("quicktok (C++)", lambda s: qt.encode_ordinary(s)))
        if hasattr(qt, "encode_to_numpy"):
            out.append(("quicktok (numpy)", lambda s: qt.encode_to_numpy(s)))
    except Exception:
        pass
    try:
        import tiktoken

        tk = tiktoken.get_encoding(enc_name)
        out.append(("tiktoken", lambda s: tk.encode_ordinary(s)))
    except Exception:
        pass
    return out


def verify(encs, text):
    """Drop any encoder whose ids differ from the first one's."""
    ref_label, ref_fn = encs[0]
    ref = list(ref_fn(text))
    kept = [encs[0]]
    for label, fn in encs[1:]:
        got = list(fn(text))
        if got != ref:
            i = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
            print(f"  !! {label} differs from {ref_label} at token {i} "
                  f"({ref[i:i+3]} vs {got[i:i+3]}) — excluded", file=sys.stderr)
            continue
        kept.append((label, fn))
    return kept, len(ref)


def bench_one(fn, text, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(text)
        best = min(best, time.perf_counter() - t0)
    return best


def run(corpus_path, enc_name, repeats, threads):
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    mb = len(text.encode("utf-8")) / 1e6
    encs = encoders(enc_name)
    print(f"\n=== {os.path.basename(corpus_path)} · {enc_name} · {mb:.1f} MB "
          f"· best of {repeats} ===")
    encs, ntok = verify(encs, text)
    print(f"    {ntok} tokens, all {len(encs)} encoders byte-exact\n")
    rows = []
    for label, fn in encs:
        secs = bench_one(fn, text, repeats)
        rows.append((label, mb / secs))
    base = dict(rows).get("tiktoken")
    width = max(len(r[0]) for r in rows)
    for label, mbs in sorted(rows, key=lambda r: -r[1]):
        rel = f"   {mbs / base:5.2f}x tiktoken" if base else ""
        print(f"  {label:<{width}}  {mbs:7.1f} MB/s{rel}")

    if threads > 1:
        import toktok

        tt = toktok.get_encoding(enc_name)
        docs = [d for d in text.split("\n\n") if d]
        print(f"\n  batch, {len(docs)} docs, {threads} threads:")
        t0 = time.perf_counter()
        tt.encode_batch(docs, threads, False)
        print(f"    toktok encode_batch   {mb / (time.perf_counter() - t0):7.1f} MB/s")
        try:
            import quicktok

            qt = quicktok.get_encoding(enc_name)
            t0 = time.perf_counter()
            qt.encode_batch(docs, threads, False)
            print(f"    quicktok encode_batch {mb / (time.perf_counter() - t0):7.1f} MB/s")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="cl100k_base,o200k_base")
    ap.add_argument("--corpus", default="", help="corpus name(s), comma separated")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if a.corpus:
        want = a.corpus.split(",")
        paths = [p for p in paths if os.path.basename(p)[:-4] in want]
    if not paths:
        sys.exit(f"no corpora in {CORPUS_DIR} — run: python bench/fetch_corpus.py")
    for enc in a.enc.split(","):
        for p in paths:
            run(p, enc, a.repeats, a.threads)


if __name__ == "__main__":
    main()
