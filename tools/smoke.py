#!/usr/bin/env python3
"""Post-install smoke test: is the wheel that would ship actually usable here?

Run against an installed toktok (not the source tree) on every interpreter and
platform we publish for. Deliberately small and dependency-free — it answers
"does the binary load and tokenize correctly on this Python", not "is the
tokenizer correct", which the real test suite covers.

    uv run --python 3.12 --with dist/toktok_rs-*.whl --no-project python tools/smoke.py
"""

import platform
import sys

import toktok

# ids/counts verified against tiktoken; if the extension loads but misbehaves on
# some platform (alignment, endianness, a bad build), these change
EXPECTED = [
    (["hello world"], "cl100k_base", [2]),
    (["how many tokens is this?"], "cl100k_base", [6]),
    (["hello world"], "o200k_base", [2]),
    (["hello world"], "gpt-4o", [2]),           # model-name resolution
    ([""], "cl100k_base", [0]),
    (["日本語のテキストです"], "cl100k_base", [9]),    # multibyte path
    (["a<|endoftext|>b"], "cl100k_base", [9]),  # specials as ordinary text
]

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()


def main() -> int:
    print(
        f"python {sys.version.split()[0]} ({'free-threaded' if FREE_THREADED else 'gil'}) "
        f"on {platform.system()} {platform.machine()} · toktok {toktok.__version__}"
    )

    failures = []
    for texts, encoding, want in EXPECTED:
        got = toktok.batch_count(texts, encoding)
        if got != want:
            failures.append(f"batch_count({texts!r}, {encoding!r}) = {got}, want {want}")

    # threading actually works from this interpreter
    docs = ["hello world " * 50] * 200
    if toktok.batch_count(docs, threads=4) != toktok.batch_count(docs, threads=1):
        failures.append("threaded and single-threaded counts disagree")

    # a free-threaded build must stay free-threaded after importing us
    if FREE_THREADED and sys._is_gil_enabled():
        failures.append("importing toktok re-enabled the GIL")

    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"ok — {len(EXPECTED)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
