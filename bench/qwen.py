#!/usr/bin/env python3
"""Throughput of 'qwen3.5' against Hugging Face `tokenizers` (the reference),
with cl100k_base alongside, on the corpora fetched by `fetch_corpus.py`.

Ids are checked identical to `tokenizers` before timing. Pass bigpickle's
tokenizer.json:

    python bench/qwen.py path/to/tokenizer.json
"""

import os
import sys
import time
import unicodedata

import toktok
from tokenizers import Tokenizer

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
DOC = 20_000  # chars per text: payload-sized


def best_mbps(fn, mb, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return mb / best


def main(path):
    hf = Tokenizer.from_file(path)
    enc = toktok._encoding("qwen3.5")
    print("| corpus | threads | qwen3.5 | tokenizers | cl100k_base |\n|---|---:|---:|---:|---:|")
    for name in ["pile", "code", "commoncrawl"]:
        with open(os.path.join(HERE, f"{name}.txt"), encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs = [text[i : i + DOC] for i in range(0, len(text), DOC)]
        mb = len(text.encode()) / 1e6
        for d, r in zip(docs, hf.encode_batch(docs, add_special_tokens=False), strict=True):
            if "<|" not in d:  # tokenizers always splits special strings
                assert enc.encode_ordinary(unicodedata.normalize("NFC", d)) == r.ids, name
        for threads in (1, 0):
            time.sleep(1)  # let tokenizers' worker pool go idle first
            q = best_mbps(lambda d=docs, t=threads: toktok.batch_count(d, "qwen3.5", threads=t), mb)
            cl = best_mbps(
                lambda d=docs, t=threads: toktok.batch_count(d, "cl100k_base", threads=t), mb
            )
            if threads == 1:
                h = best_mbps(
                    lambda ds=docs: [hf.encode(d, add_special_tokens=False) for d in ds], mb, 1
                )
            else:
                h = best_mbps(lambda ds=docs: hf.encode_batch(ds, add_special_tokens=False), mb, 1)
            print(f"| {name} | {threads or 'all'} | {q:.0f} MB/s | {h:.1f} MB/s | {cl:.0f} MB/s |")


if __name__ == "__main__":
    main(sys.argv[1])
