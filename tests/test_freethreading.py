"""Free-threaded (no-GIL) behaviour.

The extension declares `gil_used = false`, so on a free-threaded interpreter
CPython must NOT re-enable the GIL for us — if it did, every count would
serialize on it and the whole point of shipping a `cp314t` wheel would be lost.
These tests fail loudly if that regresses.

On a GIL build they still exercise concurrent counting, which is worth having
either way: the Tokenizer is shared across threads by design.
"""

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


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded interpreter")
def test_scales_without_the_gil():
    """Counting releases the GIL, so 4 threads should beat 1 on a no-GIL build.

    Deliberately loose (>1.3x): CI runners are shared and this is a smoke test
    for 'the work actually runs in parallel', not a benchmark.
    """
    import time

    big = [d * 20 for d in DOCS]

    def timed(threads):
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            toktok.batch_count(big, threads=threads)
            best = min(best, time.perf_counter() - t0)
        return best

    one, four = timed(1), timed(4)
    assert one / four > 1.3, f"no speedup from 4 threads: {one:.3f}s vs {four:.3f}s"
