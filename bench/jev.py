#!/usr/bin/env python3
"""Throughput of the 'jev' estimate next to cl100k_base, on the corpora fetched
by `fetch_corpus.py`, through the public `toktok.batch_count`. There is no Jev
tokenizer to compare against: accuracy is checked in tests/test_jev.py.

    python bench/jev.py              # 1 thread, then every core
"""

import os
import time

import toktok

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
DOC = 20_000  # chars per text: payload-sized


def best_mbps(docs, mb, encoding, threads, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        toktok.batch_count(docs, encoding, threads=threads)
        best = min(best, time.perf_counter() - t)
    return mb / best


def main():
    print("| corpus | threads | jev | cl100k_base |\n|---|---:|---:|---:|")
    for name in ["pile", "code", "commoncrawl"]:
        with open(os.path.join(HERE, f"{name}.txt"), encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs = [text[i : i + DOC] for i in range(0, len(text), DOC)]
        mb = len(text.encode()) / 1e6
        for threads in (1, 0):
            jev = best_mbps(docs, mb, "jev", threads)
            cl = best_mbps(docs, mb, "cl100k_base", threads)
            print(f"| {name} | {threads or 'all'} | {jev:.0f} MB/s | {cl:.0f} MB/s |")


if __name__ == "__main__":
    main()
