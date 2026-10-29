#!/usr/bin/env python3
"""Fail if any file tracked by git contains CJK characters.

The repository's language policy is English-only. Fullwidth forms in the
U+FF00 block are deliberately excluded because the DeepSeek chat template
uses a fullwidth vertical bar inside its special tokens
(e.g. <|end of sentence|>).
"""

from __future__ import annotations

import subprocess
import sys

# Ranges that only appear in CJK text: punctuation, kana, ideographs, hangul.
CJK_RANGES = (
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
)


def first_offence(line: str) -> int | None:
    for char in line:
        codepoint = ord(char)
        for low, high in CJK_RANGES:
            if low <= codepoint <= high:
                return codepoint
    return None


def tracked_files() -> list[str]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True
    ).stdout
    return [name.decode("utf-8", "surrogateescape") for name in raw.split(b"\0") if name]


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary or non-UTF-8 asset
        if "\0" in text:
            continue  # binary file
        for number, line in enumerate(text.splitlines(), start=1):
            codepoint = first_offence(line)
            if codepoint is not None:
                failures.append(f"{path}:{number}: U+{codepoint:04X} in {line.strip()[:100]}")

    if failures:
        print("CJK characters are not allowed in this repository:")
        for failure in failures:
            print(f"  {failure}")
        print(f"\n{len(failures)} offending line(s). Translate to English or remove them.")
        return 1

    print("OK: no CJK characters in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
