#!/usr/bin/env python3
"""
sanitize_external_corpus.py
──────────────────────────
Sanitize an existing external markdown corpus without modifying the original.

Goal:
- Preserve true structural markdown headers (# source group, ## record title)
- Neutralize line-leading heading markers in body text so malformed scraped
  prose is not treated as section boundaries during parsing.

Default I/O:
- input:  data/processed/external_ophthalmic_resources_clean.md
- output: data/processed/external_ophthalmic_resources_sanitized.md
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("data/processed/external_ophthalmic_resources_clean.md")
DEFAULT_OUTPUT = Path("data/processed/external_ophthalmic_resources_sanitized.md")

ALLOWED_H1 = {
    "external ophthalmic resources",
    "eyewiki (aao)",
    "pmc open access",
    "hugging face - eye-lit complete",
    "hugging face - medrag textbooks",
    "aao preferred practice patterns",
    "statpearls (ncbi bookshelf)",
    "merck manual professional",
    "wikipedia (ophthalmology)",
}

METADATA_LINE_RE = re.compile(r"^\*(source|url|metadata)\*\s*:", flags=re.I)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _is_allowed_h1(title: str) -> bool:
    return title.strip().lower() in ALLOWED_H1


def _is_plausible_h2(title: str) -> bool:
    t = title.strip()
    if not t:
        return False
    if len(t) > 140:
        return False
    if len(t.split()) > 20:
        return False
    if re.search(r"https?://|www\.", t, flags=re.I):
        return False
    if METADATA_LINE_RE.match(t):
        return False
    sentence_punct = len(re.findall(r"[.!?;]", t))
    if sentence_punct >= 3:
        return False
    if re.search(r"[.!?]\s+[A-Z]", t):
        return False
    return True


def _neutralize_heading_line(line: str) -> str:
    # Escape the first marker to force markdown to treat it as literal text.
    return "\\" + line


def sanitize_markdown(text: str) -> tuple[str, dict]:
    lines = text.splitlines(keepends=True)

    in_record = False
    state = "outside"  # outside | metadata | body

    neutralized_by_level: Counter[int] = Counter()
    untouched_structural_h1 = 0
    untouched_structural_h2 = 0

    out_lines: list[str] = []

    for line in lines:
        stripped_nl = line.rstrip("\n")

        heading = HEADING_RE.match(stripped_nl)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()

            # Preserve only intended structural boundaries.
            if level == 1 and _is_allowed_h1(title):
                untouched_structural_h1 += 1
                in_record = False
                state = "outside"
                out_lines.append(line)
                continue

            if level == 2 and _is_plausible_h2(title):
                untouched_structural_h2 += 1
                in_record = True
                state = "metadata"
                out_lines.append(line)
                continue

            # Non-structural heading-like lines become safe literal text in body.
            if in_record:
                if state != "body":
                    state = "body"
                out_lines.append(_neutralize_heading_line(line))
                neutralized_by_level[level] += 1
                continue

            # Outside records, keep as-is to avoid unexpected destructive changes.
            out_lines.append(line)
            continue

        # Metadata block handling immediately after ## title.
        if in_record and state == "metadata":
            s = stripped_nl.strip()
            if not s:
                out_lines.append(line)
                continue
            if METADATA_LINE_RE.match(s):
                out_lines.append(line)
                continue

            # First non-metadata content line starts body block.
            state = "body"
            out_lines.append(line)
            continue

        out_lines.append(line)

    report = {
        "neutralized_by_level": dict(sorted(neutralized_by_level.items())),
        "neutralized_total": sum(neutralized_by_level.values()),
        "untouched_structural_h1": untouched_structural_h1,
        "untouched_structural_h2": untouched_structural_h2,
    }
    return "".join(out_lines), report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize existing external corpus markdown for robust parsing.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", type=Path, default=None, help="Optional backup copy of input.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}")
        return 1

    text = args.input.read_text(encoding="utf-8")
    sanitized, report = sanitize_markdown(text)

    in_bytes = len(text.encode("utf-8", errors="ignore"))
    out_bytes = len(sanitized.encode("utf-8", errors="ignore"))

    print("Sanitization summary")
    print("-" * 72)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Input size:  {in_bytes / 1_000_000:.1f} MB")
    print(f"Output size: {out_bytes / 1_000_000:.1f} MB")
    print(f"Structural H1 preserved: {report['untouched_structural_h1']:,}")
    print(f"Structural H2 preserved: {report['untouched_structural_h2']:,}")
    print(f"Neutralized body heading-like lines: {report['neutralized_total']:,}")

    if report["neutralized_by_level"]:
        levels = ", ".join(
            f"H{lvl}:{count:,}" for lvl, count in report["neutralized_by_level"].items()
        )
        print(f"By level: {levels}")

    if args.dry_run:
        print("\nDry run enabled. No file written.")
        return 0

    if args.backup is not None:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.input, args.backup)
        print(f"Backup created: {args.backup}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sanitized, encoding="utf-8")
    print(f"\nSanitized corpus written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
