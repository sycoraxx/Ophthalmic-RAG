#!/usr/bin/env python3
"""
purge_pmc_from_corpus.py
────────────────────────
Strip the '# PMC Open Access' section (and all its ## sub-entries) from
external_ophthalmic_resources_clean.md so the rest of the clean corpus is
preserved while PMC articles are re-fetched with the fixed HTML pipeline.

Usage:
    python scripts/purge_pmc_from_corpus.py
    # Re-fetch PMC only:
    python scripts/fetch_external_resources.py \\
        --sources pmc --max-pmc 5000 \\
        --output-md data/processed/pmc_clean_refetch.md
    # Then merge:
    python scripts/merge_pmc_corpus.py
"""
from pathlib import Path
import re, sys

CORPUS = Path("data/processed/external_ophthalmic_resources_clean.md")
BACKUP = CORPUS.with_suffix(".md.bak")

if not CORPUS.exists():
    print(f"ERROR: corpus not found at {CORPUS}")
    sys.exit(1)

text = CORPUS.read_text(encoding="utf-8")

# Split at every top-level H1 section boundary
# Sections start with '\n# ' (or at the very start with '# ')
sections = re.split(r"(?=\n# |\A# )", text)

before, pmc_section, after = [], None, []
in_pmc = False
for sec in sections:
    header = re.match(r"\A#?\n?# (.+)", sec)
    if header and header.group(1).strip() == "PMC Open Access":
        pmc_section = sec
        in_pmc = True
    elif in_pmc:
        after.append(sec)
        in_pmc = False
    else:
        before.append(sec)

if pmc_section is None:
    print("No '# PMC Open Access' section found — nothing to purge.")
    sys.exit(0)

pmc_lines = pmc_section.count("\n")
print(f"Found PMC section ({pmc_lines} lines). Backing up to {BACKUP} ...")
CORPUS.rename(BACKUP)

clean = "".join(before + after)
CORPUS.write_text(clean, encoding="utf-8")

remaining = clean.count("## ")
print(f"Purge complete. {remaining} entries remain in corpus.")
print(f"Backup saved at: {BACKUP}")
print()
print("Next steps:")
print("  1) conda run -n rag python scripts/fetch_external_resources.py \\")
print("         --sources pmc --max-pmc 5000 \\")
print("         --output-md data/processed/pmc_clean_refetch.md")
print("  2) cat data/processed/pmc_clean_refetch.md >> data/processed/external_ophthalmic_resources_clean.md")
print("  3) conda run -n rag python scripts/chunk_data.py")
print("  4) conda run -n rag python scripts/ingest_db.py")
