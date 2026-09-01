"""Free-threaded (no-GIL) behaviour.

The extension declares `gil_used = false`, so on a free-threaded interpreter
CPython must NOT re-enable the GIL for us — if it did, every count would
serialize on it and the whole point of shipping a `cp314t` wheel would be lost.
These tests fail loudly if that regresses.

On a GIL build they still exercise concurrent counting, which is worth having
either way: the Tokenizer is shared across threads by design.
"""

import os
import sys
import threading

import pytest

import toktok

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()

DOCS = [f"doc {i}: hello 日本語 world, how many tokens? " * 5 for i in range(500)]


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded interpreter")
def test_gil_stays_disabled_after_import():
    # importing an extension that does not declare free-threading support makes
    # CPython switch the GIL back on for the whole process
    assert sys._is_gil_enabled() is False


def test_concurrent_batch_count_agrees():
    """The same work from many threads must produce identical counts."""
    expected = toktok.batch_count(DOCS, threads=1)
    results, errors = [], []

    def work():
        try:
            results.append(toktok.batch_count(DOCS, "cl100k_base", threads=4))
        except Exception as e:  # noqa: BLE001 — surface it in the assertion
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(r == expected for r in results)


def test_concurrent_mixed_encodings():
    """Different encodings in parallel: separate tokenizers, shared cache."""
    want = {e: toktok.batch_count(DOCS[:100], e, threads=1)
            for e in ("cl100k_base", "o200k_base", "o200k_harmony")}
    out, errors = {}, []
    lock = threading.Lock()

    def work(enc):
        try:
            got = toktok.batch_count(DOCS[:100], enc, threads=2)
            with lock:
                out.setdefault(enc, []).append(got)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(e,))
               for e in want for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for enc, runs in out.items():
        assert all(r == want[enc] for r in runs), enc


@pytest.mark.skipif(os.cpu_count() < 2, reason="needs more than one core")
def test_counting_actually_runs_in_parallel():
    """Prove the worker threads run *concurrently*, not that they are faster.

    Comparing two wall-clock timings is hopeless on a shared runner (the first
    attempt compared 9 ms against 9 ms and failed). Instead compare CPU time to
    wall time: process_time() sums every thread's CPU, so burning more CPU
    seconds than wall seconds is only possible if several threads ran at once.

    This holds on a GIL build too — counting releases the GIL — and it is
    exactly the property the cp314t wheel promises.
    """
    import time

    # ~14 MB of text. Sizing this is awkward precisely because the tokenizer is
    # fast — it clears that in well under a tenth of a second — so the check
    # below only needs the window to be long enough to time, not to be long.
    docs = [d * 40 for d in DOCS] * 4
    threads = min(4, os.cpu_count())

    toktok.batch_count(docs[:50], threads=threads)  # warm

    ratio = 0.0
    for _ in range(3):  # best of 3: a busy runner can starve one attempt
        w0, c0 = time.perf_counter(), time.process_time()
        toktok.batch_count(docs, threads=threads)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        assert wall > 0.003, f"workload too small to measure ({wall:.4f}s)"
        ratio = max(ratio, cpu / wall)

    assert ratio > 1.3, (
        f"only {ratio:.2f} CPU-seconds per wall-second with {threads} threads — "
        "the work did not run in parallel"
    )
