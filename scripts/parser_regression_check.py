#!/usr/bin/env python3
"""
parser_regression_check.py
─────────────────────────
Lightweight regression checks for markdown section parsing behavior.

Usage:
  python scripts/parser_regression_check.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNK_DATA = ROOT / "scripts" / "chunk_data.py"


def _load_chunk_data_module():
    spec = importlib.util.spec_from_file_location("chunk_data_module", CHUNK_DATA)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {CHUNK_DATA}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case_1(mod) -> tuple[bool, str]:
    md = """
# External Ophthalmic Resources

# EyeWiki (AAO)

## Acanthamoeba Keratitis

*Source:* QIAIUNCC/EYE-lit-complete
*URL:* https://huggingface.co/datasets/QIAIUNCC/EYE-lit-complete
*Metadata:* row_idx=1

# This is malformed scraped prose with enough words to look like a paragraph, not a chapter boundary.
Symptoms include pain, redness, and watering in the affected eye over multiple days.

## Next True Section

This is true body text for another section and should remain separate.
""".strip()

    parents = mod.parse_markdown_sections(md, "External Ophthalmic Resources")
    if len(parents) != 2:
        return False, f"case1 expected 2 sections, got {len(parents)}"

    first = parents[0]
    if "malformed scraped prose" not in first.page_content:
        return False, "case1 malformed heading line was not preserved as body content"

    if first.metadata.get("chapter") != "EyeWiki (AAO)":
        return False, f"case1 chapter mismatch: {first.metadata.get('chapter')}"

    return True, "case1 passed"


def run_case_2(mod) -> tuple[bool, str]:
    md = """
# Chapter One

## Intro Section

This starts with good content and should remain one section.

## This line looks like a long paragraph sentence with punctuation and should not become a real section boundary because it is malformed and too prose-like.

Additional content still belongs to Intro Section.
""".strip()

    parents = mod.parse_markdown_sections(md, "Any Source")
    if len(parents) != 1:
        return False, f"case2 expected 1 section, got {len(parents)}"

    if "Additional content still belongs" not in parents[0].page_content:
        return False, "case2 content after malformed H2 did not stay in same section"

    return True, "case2 passed"


def main() -> int:
    try:
        mod = _load_chunk_data_module()
    except Exception as exc:
        print(f"ERROR: failed to load chunk parser module: {exc}")
        return 1

    checks = [run_case_1, run_case_2]
    failures = []

    for check in checks:
        ok, msg = check(mod)
        print(msg)
        if not ok:
            failures.append(msg)

    if failures:
        print("\nParser regression checks FAILED")
        for fail in failures:
            print(f"- {fail}")
        return 1

    print("\nParser regression checks PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
