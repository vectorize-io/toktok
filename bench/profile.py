#!/usr/bin/env python3
"""Resource profile: memory, CPU and per-document latency for each encoder.

`compare.py` answers "how many MB/s on one big string". This answers the
questions that matter when you actually deploy a tokenizer:

  * **memory**  — resident bytes the loaded tables cost, and peak RSS while encoding
  * **CPU**     — CPU-seconds per MB, and cores used (CPU time / wall time)
  * **latency** — per-document encode latency: p50 / p90 / p99 / p999 / max

Every encoder runs in its **own subprocess**, so RSS is attributable to that
encoder alone and no other library's tables are resident. Output of every
encoder is checked against toktok before anything is measured.

    python bench/profile.py                                  # all encoders, defaults
    python bench/profile.py --enc o200k_base --corpus code
    python bench/profile.py --threads 8                      # add the parallel-batch row
"""

import argparse
import gc
import glob
import json
import os
import resource
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")

# encoder id -> label
ENCODERS = {
    "toktok": "toktok",
    "toktok-numpy": "toktok (numpy)",
    "quicktok": "quicktok (C++)",
    "tiktoken": "tiktoken",
}

# how many documents to time individually for the latency distribution
LATENCY_DOCS = 4000


# --------------------------------------------------------------------------
# worker: runs inside a fresh interpreter, one encoder only
# --------------------------------------------------------------------------


def rss_bytes():
    """Current resident set size, in bytes."""
    # ru_maxrss is a high-water mark: for the *current* RSS read the OS directly.
    if sys.platform == "linux":
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGESIZE")
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout.strip()
    return int(out) * 1024


def peak_rss_bytes():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if sys.platform == "darwin" else v * 1024  # macOS: bytes, Linux: KB


def cpu_seconds():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def load_encoder(which, enc_name):
    """-> (encode_fn, batch_fn_or_None, count_batch_fn_or_None).

    Called with only this encoder imported, so RSS is attributable to it."""
    if which in ("toktok", "toktok-numpy"):
        import toktok

        e = toktok.get_encoding(enc_name)
        # the numpy path returns one uint32 buffer instead of a list[int] — same
        # ids, ~8x less memory for the result
        enc_fn = (
            e.encode_ordinary if which == "toktok" else (lambda s: toktok.encode_to_numpy(e, s))
        )
        return (
            enc_fn,
            lambda docs, n: e.encode_batch(docs, n, False),
            lambda docs, n: e.count_batch(docs, n, False),
        )
    if which == "quicktok":
        import quicktok

        e = quicktok.get_encoding(enc_name)
        batch = (
            (lambda docs, n: e.encode_batch(docs, n, False)) if hasattr(e, "encode_batch") else None
        )
        return e.encode_ordinary, batch, None
    if which == "tiktoken":
        import tiktoken

        e = tiktoken.get_encoding(enc_name)
        return (
            e.encode_ordinary,
            lambda docs, n: e.encode_ordinary_batch(docs, num_threads=n),
            None,
        )
    raise SystemExit(f"unknown encoder {which}")


def mem_worker(which, enc_name):
    """A process that does nothing but load one encoding, so its RSS growth is
    attributable to that encoding alone (measuring this inside the benchmark
    process gives garbage: the corpus and its buffers churn RSS by tens of MB)."""
    gc.collect()
    base = rss_bytes()
    t0 = time.perf_counter()
    load_encoder(which, enc_name)
    load_s = time.perf_counter() - t0
    gc.collect()
    print(
        json.dumps(
            {
                "rss_load": rss_bytes() - base,
                "peak_rss": peak_rss_bytes(),
                "load_s": load_s,
                "tables_exact": exact_table_bytes(which, enc_name),
            }
        )
    )


def exact_table_bytes(which, enc_name):
    """Exact live table footprint, when the encoder can report it. An RSS delta
    also counts construction scratch the allocator hasn't returned to the OS, so
    the two numbers answer different questions and we print both."""
    if which.startswith("toktok"):
        import toktok

        return toktok.get_encoding(enc_name).memory_bytes
    return None


def worker(which, enc_name, corpus_path, threads):
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    docs = [d for d in text.split("\n\n") if d][:LATENCY_DOCS]
    nbytes = len(text.encode("utf-8"))

    base_rss = rss_bytes()
    t0 = time.perf_counter()
    encode, encode_batch, count_batch = load_encoder(which, enc_name)
    load_s = time.perf_counter() - t0
    tables_rss = rss_bytes() - base_rss

    ids = encode(text)  # warm + the ids the parent checks for exactness

    # --- throughput + CPU accounting on the whole corpus ---
    best_wall, best_cpu = float("inf"), None
    for _ in range(3):
        c0, w0 = cpu_seconds(), time.perf_counter()
        encode(text)
        w, c = time.perf_counter() - w0, cpu_seconds() - c0
        if w < best_wall:
            best_wall, best_cpu = w, c

    # --- per-document latency distribution ---
    for d in docs[:50]:
        encode(d)  # warm the memo/caches on this shape of input
    lat_ns = []
    for d in docs:
        t = time.perf_counter_ns()
        encode(d)
        lat_ns.append(time.perf_counter_ns() - t)

    # --- parallel batch: wall time and how many cores it actually used ---
    batch = None
    if threads > 1 and encode_batch is not None:
        encode_batch(docs[:100], threads)  # warm threads
        c0, w0 = cpu_seconds(), time.perf_counter()
        encode_batch(docs, threads)
        bw, bc = time.perf_counter() - w0, cpu_seconds() - c0
        dbytes = sum(len(d.encode("utf-8")) for d in docs)
        batch = {"wall_s": bw, "cpu_s": bc, "bytes": dbytes}
    # RSS cost of holding a whole batch of ids. Measured on its own (so the timings
    # above aren't perturbed) and as the max of three rounds with the collector
    # settled either side — a single round can read negative when the allocator
    # hands pages back mid-measurement.
    batch_rss = None
    if encode_batch is not None:
        for _ in range(3):
            gc.collect()
            rss0 = rss_bytes()
            held = encode_batch(docs, max(1, threads))
            delta = rss_bytes() - rss0
            del held
            gc.collect()
            if batch_rss is None or delta > batch_rss:
                batch_rss = delta
        batch_rss = max(batch_rss, 0)

    # --- counting-only batch: the ids are never materialized ---
    count = None
    if count_batch is not None:
        count_batch(docs[:100], max(1, threads))  # warm
        rss0 = rss_bytes()
        c0, w0 = cpu_seconds(), time.perf_counter()
        counts = count_batch(docs, max(1, threads))
        count = {
            "wall_s": time.perf_counter() - w0,
            "cpu_s": cpu_seconds() - c0,
            "rss_delta": rss_bytes() - rss0,
            "total": int(sum(counts)),
        }
        del counts

    print(
        json.dumps(
            {
                "encoder": which,
                "load_s": load_s,
                "tables_rss": tables_rss,
                "peak_rss": peak_rss_bytes(),
                "bytes": nbytes,
                "wall_s": best_wall,
                "cpu_s": best_cpu,
                "n_tokens": len(ids),
                "checksum": sum(int(i) for i in ids),
                "doc_bytes": sum(len(d.encode("utf-8")) for d in docs),
                "n_docs": len(docs),
                "lat_ns": lat_ns,
                "batch": batch,
                "batch_rss": batch_rss,
                "count": count,
            }
        )
    )


# --------------------------------------------------------------------------
# parent: run every encoder in its own process, then tabulate
# --------------------------------------------------------------------------


def pct(sorted_vals, q):
    """Nearest-rank percentile (no interpolation — p99 of a latency sample
    should be an observed value, not an average of two)."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, round(q * len(sorted_vals) + 0.5) - 1))
    return sorted_vals[k]


def mb(n):
    return n / (1024 * 1024)


def run_worker(which, enc_name, corpus_path, threads, mode="--worker"):
    r = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            mode,
            which,
            "--enc",
            enc_name,
            "--corpus-path",
            corpus_path,
            "--threads",
            str(threads),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None, (r.stderr.strip().splitlines() or ["failed"])[-1]
    return json.loads(r.stdout), None


def report(enc_name, corpus_path, threads, only):
    print(
        f"\n=== {os.path.basename(corpus_path)} · {enc_name} "
        f"· {mb(os.path.getsize(corpus_path)):.1f} MiB ==="
    )
    results = []
    for which, label in ENCODERS.items():
        if only and which not in only:
            continue
        res, err = run_worker(which, enc_name, corpus_path, threads)
        if res is None:
            print(f"  (skipping {label}: {err})", file=sys.stderr)
            continue
        res["label"] = label
        # load cost measured in its own bare process (best of 3 — RSS accounting
        # on macOS wobbles by a few MB between runs)
        mem = [
            m
            for m in (
                run_worker(which, enc_name, corpus_path, threads, "--memworker")[0]
                for _ in range(3)
            )
            if m
        ]
        if mem:
            best = min(mem, key=lambda m: m["rss_load"])
            res["rss_load"] = best["rss_load"]
            res["load_s"] = min(m["load_s"] for m in mem)
            res["tables_exact"] = best["tables_exact"]
            res["load_peak_rss"] = best["peak_rss"]
        results.append(res)
    if not results:
        sys.exit("no encoders available")

    ref = next((r for r in results if r["encoder"] == "toktok"), results[0])
    for r in results:
        if r["checksum"] != ref["checksum"] or r["n_tokens"] != ref["n_tokens"]:
            print(
                f"  !! {r['label']} ids differ from {ref['label']} — results not comparable",
                file=sys.stderr,
            )
    print(f"    {ref['n_tokens']} tokens, {len(results)} encoders agree byte-for-byte")

    w = max(len(r["label"]) for r in results)

    print("\n  memory")
    print(
        f"    {'encoder':<{w}}  {'tables':>8}  {'RSS@load':>9}  {'load':>7}  "
        f"{'RSS peak':>9}  {'ids held':>9}"
    )
    for r in sorted(results, key=lambda r: r.get("rss_load", 0)):
        exact = f"{mb(r['tables_exact']):7.1f}M" if r.get("tables_exact") else "      —"
        held = f"{mb(r['batch_rss']):8.1f}M" if r.get("batch_rss") is not None else "      n/a"
        print(
            f"    {r['label']:<{w}}  {exact}  {mb(r.get('rss_load', 0)):8.1f}M  "
            f"{r['load_s'] * 1e3:6.0f}ms  {mb(r['load_peak_rss']):8.1f}M  {held}"
        )
    print("      tables  = exact live table bytes (where the encoder reports them)")
    print("      RSS@load = RSS growth of a bare process that only loads the encoding —")
    print("                 also counts construction scratch the allocator kept")
    print(f"      ids held = RSS to hold the ids of all {results[0]['n_docs']} docs at once")

    print("\n  CPU (single thread, whole corpus)")
    print(f"    {'encoder':<{w}}  {'MB/s':>7}  {'CPU-s/MB':>9}  {'cores':>6}")
    for r in sorted(results, key=lambda r: -r["bytes"] / r["wall_s"]):
        mbs = mb(r["bytes"]) / r["wall_s"]
        print(
            f"    {r['label']:<{w}}  {mbs:7.1f}  {r['cpu_s'] / mb(r['bytes']):9.4f}  "
            f"{r['cpu_s'] / r['wall_s']:6.2f}"
        )

    med_doc = statistics.median(r["doc_bytes"] / r["n_docs"] for r in results)
    print(f"\n  per-document latency, µs ({ref['n_docs']} docs, mean {med_doc / 1024:.1f} KiB)")
    print(f"    {'encoder':<{w}}  {'p50':>9}  {'p90':>9}  {'p99':>9}  {'p99.9':>9}  {'max':>9}")
    for r in sorted(results, key=lambda r: pct(sorted(r["lat_ns"]), 0.99)):
        s = sorted(r["lat_ns"])
        cells = [pct(s, q) / 1e3 for q in (0.50, 0.90, 0.99, 0.999)] + [s[-1] / 1e3]
        print(f"    {r['label']:<{w}}  " + "  ".join(f"{c:9.1f}" for c in cells))

    if any(r.get("count") for r in results):
        thr = max(1, threads)
        print(f"\n  count-only batch, {thr} thread(s) — ids never materialized")
        print(f"    {'encoder':<{w}}  {'MB/s':>7}  {'RSS growth':>10}  {'encode_batch RSS':>16}")
        for r in results:
            c = r.get("count")
            if not c:
                continue
            vs = ""
            if r.get("batch_rss"):
                vs = f"{mb(r['batch_rss']):15.1f}M"
            print(
                f"    {r['label']:<{w}}  {mb(r['doc_bytes']) / c['wall_s']:7.1f}  "
                f"{mb(c['rss_delta']):9.1f}M  {vs:>15}"
            )

    if threads > 1 and any(r["batch"] for r in results):
        print(f"\n  batch encode, {threads} threads")
        print(f"    {'encoder':<{w}}  {'MB/s':>7}  {'CPU-s/MB':>9}  {'cores':>6}")
        for r in sorted(
            results,
            key=lambda r: -(r["batch"]["bytes"] / r["batch"]["wall_s"]) if r["batch"] else 0,
        ):
            b = r["batch"]
            if not b:
                continue
            print(
                f"    {r['label']:<{w}}  {mb(b['bytes']) / b['wall_s']:7.1f}  "
                f"{b['cpu_s'] / mb(b['bytes']):9.4f}  {b['cpu_s'] / b['wall_s']:6.2f}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="cl100k_base,o200k_base")
    ap.add_argument("--corpus", default="", help="corpus name(s), comma separated")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", default="", help="encoder id(s) to run: toktok,quicktok,tiktoken")
    # worker-side flags
    ap.add_argument("--worker", default="")
    ap.add_argument("--memworker", default="")
    ap.add_argument("--corpus-path", default="")
    a = ap.parse_args()

    if a.memworker:
        mem_worker(a.memworker, a.enc)
        return
    if a.worker:
        worker(a.worker, a.enc, a.corpus_path, a.threads)
        return

    paths = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if a.corpus:
        want = a.corpus.split(",")
        paths = [p for p in paths if os.path.basename(p)[:-4] in want]
    if not paths:
        sys.exit(f"no corpora in {CORPUS_DIR} — run: python bench/fetch_corpus.py")
    only = set(a.only.split(",")) if a.only else None
    for enc in a.enc.split(","):
        for p in paths:
            report(enc, p, a.threads, only)


if __name__ == "__main__":
    main()
