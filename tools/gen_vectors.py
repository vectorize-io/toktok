#!/usr/bin/env python3
"""Generate the byte-exactness fixtures used by the Rust tests.

Reference ids come from tiktoken, so a passing test means toktok is
byte-identical to it. Run from the repo root:

    python tools/gen_vectors.py

Format of tests/vectors_<encoding>.bin (all little-endian):
    u32 n_cases
    per case: u32 text_len, text bytes, u32 n_ids, n_ids * u32
"""

import os
import struct
import sys

CASES = [
    "",
    " ",
    "\n",
    "\t\t",
    "hello world",
    "Hello, World!",
    "hello  world",
    "The quick brown fox jumps over the lazy dog.",
    "it's I'M they'RE won't we'll you've he'd",
    "'s 'S 'll 'LL 've 'RE 'd 'm 'x '",
    "def main(argv):\n    if x == 1:  # comment\n        return {'a': [1, 2, 3]}\n",
    "SELECT * FROM users WHERE id = 42 AND name LIKE '%foo%';",
    "0 1 12 123 1234 12345 007 3.14159 1e-9 0x1F 1_000_000",
    "     leading spaces",
    "trailing spaces     ",
    "  \n\n  \r\n\t mixed \r whitespace \n",
    "line1\nline2\r\nline3\rline4",
    "a\n\n\n\n\n\nb",
    " " * 40,
    "\n" * 20,
    "https://example.com/path?query=1&other=2#frag",
    "email@example.com, +1 (555) 010-9999",
    "ÀÉÎÕÜ àéîõü ĄĆĘŁŃÓŚŹŻ Ññ ßẞ",
    "Ελληνικά κείμενο δοκιμή",
    "Привет, мир! Как дела?",
    "العربية نص تجريبي",
    "עברית טקסט לבדיקה",
    "日本語のテキストです。これはテストです。",
    "中文测试文本，包含标点符号。",
    "한국어 텍스트 테스트입니다",
    "ไทย ภาษาไทย ทดสอบ",
    "亚洲AV 中文AB混合Test",
    "ＡＢＣ ｆｕｌｌｗｉｄｔｈ １２３",
    "🚀🎉😀 emoji test 👨‍👩‍👧‍👦 🇺🇸🇯🇵 🏳️‍🌈",
    "combining: é à ñ ö",
    "zero​width​space",
    "math: ∑∫∂√∞ ≠ ≤ ≥ α β γ",
    "box: ┌─┬─┐ │ └─┴─┘",
    "\x00\x01\x02 control bytes \x7f",
    "MiXeDcAsE CamelCase snake_case SCREAMING_SNAKE kebab-case",
    "ALLUPPER lower Mixed123Digits",
    "AAA bbb CCCddd EEEfff",
    "a" * 300,
    "ab" * 200,
    "word " * 100,
    "日本" * 100,
    "<|endoftext|>",
    "text <|endoftext|> more",
    "<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>",
    "{'json': [1, 2, {'nested': null}], \"other\": true}",
    "\\n\\t escaped \\\\ backslash",
    "```python\nprint('hi')\n```",
    "# Heading\n\n- bullet\n- list\n\n> quote\n",
    "tabs\tbetween\twords",
    "trailing newline\n",
    "\n leading newline",
    "one/two/three//four",
    "path\\to\\file.txt",
    "a,b,,c,,,d",
    "!!!???...---___+++",
    "(((nested)))[[[brackets]]]{{{braces}}}",
    "🚀start",
    "end🚀",
    "mid🚀dle",
    " 🚀 ",
]


def main():
    import tiktoken

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "tests")
    os.makedirs(out_dir, exist_ok=True)
    for enc in ("cl100k_base", "o200k_base"):
        tk = tiktoken.get_encoding(enc)
        blob = bytearray(struct.pack("<I", len(CASES)))
        for s in CASES:
            b = s.encode("utf-8")
            ids = tk.encode_ordinary(s)
            blob += struct.pack("<I", len(b)) + b
            blob += struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids)
        path = os.path.join(out_dir, f"vectors_{enc}.bin")
        with open(path, "wb") as f:
            f.write(blob)
        print(f"{path}: {len(CASES)} cases, {len(blob)} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
