#!/usr/bin/env python3
"""Fetch the three benchmark corpora used in the README tables, streamed from
their real sources and cut at 25 MB each (no multi-GB downloads):

  pile         monology/pile-uncopyrighted     — The Pile (diverse English)
  code         codeparrot/codeparrot-clean     — real GitHub Python
  commoncrawl  data.commoncrawl.org WET shard  — raw multilingual web text

Needs: pip install datasets zstandard.   Usage:

  python bench/fetch_corpus.py            # all three -> bench/corpus/<name>.txt
  python bench/fetch_corpus.py pile       # just one

Documents are joined with blank lines; the resulting text blob is what gets
benchmarked (see bench/compare.py). Streaming order is deterministic, so two
fetches produce the same bytes.
"""
import gzip, os, sys, time, urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
MAX_MB = 25.0
CC_CRAWL = "CC-MAIN-2024-10"


def stream_hf(repo, name=None):
    from datasets import load_dataset
    ds = load_dataset(repo, name=name, split="train", streaming=True)
    for ex in ds:
        t = ex.get("text") or ex.get("content") or ""
        if t:
            yield t


def stream_commoncrawl(crawl=CC_CRAWL):
    paths_url = f"https://data.commoncrawl.org/crawl-data/{crawl}/wet.paths.gz"
    with urllib.request.urlopen(paths_url, timeout=60) as r:
        first = gzip.decompress(r.read()).splitlines()[0].decode()
    wet_url = "https://data.commoncrawl.org/" + first
    print(f"  CC WET shard: {wet_url}", file=sys.stderr)
    resp = urllib.request.urlopen(wet_url, timeout=120)
    gz = gzip.GzipFile(fileobj=resp)
    while True:
        line = gz.readline()
        if not line:
            break
        if not line.startswith(b"WARC/"):
            continue
        wtype, clen = b"", 0
        while True:
            h = gz.readline()
            if h in (b"\r\n", b"\n", b""):
                break
            k, _, v = h.partition(b":")
            k = k.strip().lower()
            if k == b"warc-type":
                wtype = v.strip()
            elif k == b"content-length":
                clen = int(v.strip())
        payload = gz.read(clen)
        if wtype == b"conversion" and payload:
            yield payload.decode("utf-8", "ignore")


SOURCES = {
    "pile":        lambda: stream_hf("monology/pile-uncopyrighted"),
    "code":        lambda: stream_hf("codeparrot/codeparrot-clean-valid"),
    "commoncrawl": lambda: stream_commoncrawl(),
}


def fetch(name):
    out = os.path.join(OUT_DIR, name + ".txt")
    cap = int(MAX_MB * 1e6)
    t0, n, docs = time.time(), 0, 0
    with open(out, "wb") as f:
        for text in SOURCES[name]():
            b = text.encode("utf-8")
            f.write(b)
            f.write(b"\n\n")
            n += len(b) + 2
            docs += 1
            if n >= cap:
                break
    print(f"{name}: {n/1e6:.1f} MB, {docs} docs -> {out}  ({time.time()-t0:.1f}s)")


def main():
    names = sys.argv[1:] or sorted(SOURCES)
    bad = [n for n in names if n not in SOURCES]
    if bad:
        sys.exit(f"unknown corpus {bad}; choose from {sorted(SOURCES)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in names:
        fetch(n)
    sys.stdout.flush()
    os._exit(0)  # skip atexit (some streaming backends abort noisily at cleanup)


if __name__ == "__main__":
    main()
