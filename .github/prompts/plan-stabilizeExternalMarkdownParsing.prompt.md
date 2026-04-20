## Plan: Sanitize Existing Corpus And Harden Markdown Parser

### Objective
Sanitize the already-downloaded external corpus and harden section parsing so malformed heading-like lines in body text are never treated as structural section boundaries.

### Scope Lock
- No refetching of external resources.
- Preserve the original corpus file unchanged.
- Parse from a sanitized copy.
- Strengthen parser behavior even if malformed lines still appear.

### Why Two Layers
1. Sanitization fixes the current large corpus safely and deterministically.
2. Parser hardening prevents future malformed content from breaking section boundaries, even if sanitization is skipped or imperfect.

For starter these type of lines are present:

*Source:* QIAIUNCC/EYE-lit-complete  
*URL:* https://huggingface.co/datasets/QIAIUNCC/EYE-lit-complete  
*Metadata:* row_idx=233404; content_hash=acd7c4cb76be142a8229ec024d8aa3c4e9a6a8f94463ee5b4e553176a611a7e1; authority_rank=5  

I don't want this meta data/source/url line to be treated as a section header, but the current parser does not differentiate it from a real H1 boundary, which causes section counts to explode and metadata to be lost.


---

### Step 1: Add A Dedicated Sanitization Script
Create [scripts/sanitize_external_corpus.py](scripts/sanitize_external_corpus.py) that reads the existing corpus and writes a sanitized output file.

#### Inputs And Outputs
- Input default: [data/processed/external_ophthalmic_resources_clean.md](data/processed/external_ophthalmic_resources_clean.md)
- Output default: [data/processed/external_ophthalmic_resources_sanitized.md](data/processed/external_ophthalmic_resources_sanitized.md)
- Optional flags:
  - --input
  - --output
  - --dry-run
  - --backup (optional path for explicit backup copy)

#### Sanitization Rules
- Keep true structural headers intact:
  - Top-level source-group lines that are intended H1 boundaries.
  - Record title lines that are intended H2 boundaries.
- Inside record body content only:
  - Detect line-leading heading markers with pattern equivalent to start-of-line followed by one to six # characters and whitespace.
  - Neutralize by escaping the first marker so the line remains literal text, not markdown structure.
- Do not alter metadata lines that begin with Source, URL, or Metadata markers.
- Preserve line order and content semantics.

#### Script Reporting
Print a summary:
- Input size and output size.
- Number of body lines neutralized by level (H1 through H6 style markers).
- Count of untouched structural headers.

---

### Step 2: Harden Header Boundary Detection In Parser
Update [scripts/chunk_data.py](scripts/chunk_data.py), especially [parse_markdown_sections](scripts/chunk_data.py#L208), so H1 and H2 boundaries are accepted only when structurally plausible.

#### Concrete Parser Changes
1. Add helper(s) for header validation, for example:
	- is_plausible_header_text
	- is_allowed_external_h1
2. Reject boundary headers when they look like prose paragraphs, such as:
	- Excessive character length.
	- Sentence-like punctuation density.
	- Suspiciously high token count.
3. For rejected heading-like lines:
	- Treat them as normal body content.
	- Do not flush section or reset chapter.
4. Add source-aware H1 allowlist for external corpus parsing so only known source-group headings are accepted as H1 chapter boundaries.
5. Keep existing metadata schema unchanged so downstream ingestion remains compatible.

---

### Step 3: Add Preflight Structure Diagnostics Before Chunking
In [scripts/chunk_data.py](scripts/chunk_data.py), add a lightweight profile print before parsing each file:
- File size.
- Raw H1 and H2 counts.
- Ratio of suspicious heading-like lines to accepted structural headers.

If suspicious ratio is high, emit a warning that parsing should use sanitized corpus.

---

### Step 4: Wire Chunking To Sanitized Corpus Path
Use sanitized file for the external corpus run without changing textbook or other source workflows.

Recommended behavior:
- If input file is [data/processed/external_ophthalmic_resources_clean.md](data/processed/external_ophthalmic_resources_clean.md) and sanitized sibling exists, prefer sanitized file and log that substitution.
- If sanitized file does not exist, continue with explicit warning (do not silently fail).

This keeps behavior explicit and avoids accidental destructive overwrite.

---

### Step 5: Add Targeted Regression Checks
Add a compact parser regression test file or script, for example under [scripts](scripts) or [evaluation](evaluation), containing synthetic markdown cases:
- Valid H1 and H2 boundaries.
- Body lines that start with # but are sentence-like prose.
- Long pseudo-heading lines that should remain body content.

Pass criteria:
- Parent section count matches expected boundaries.
- Pseudo-heading prose does not split sections.

---

### Step 6: Validate End To End On Existing Corpus
1. Run sanitization script in dry-run mode to confirm neutralization volume.
2. Run sanitization write mode to produce sanitized copy.
3. Chunk only the sanitized external corpus file.
4. Confirm parent section count is in realistic range and no longer inflated by pseudo-heading boundaries.
5. Spot-check several previously bad lines to verify they are preserved as literal body text.

---

### Specific Code Changes Summary
- New file: [scripts/sanitize_external_corpus.py](scripts/sanitize_external_corpus.py)
  - Structured-state corpus pass.
  - Body-only heading neutralization.
  - Dry-run and reporting.
- Update: [scripts/chunk_data.py](scripts/chunk_data.py)
  - Header plausibility validation helpers.
  - Source-aware external H1 allowlist.
  - Rejected heading fallback to body text.
  - Preflight structure diagnostics.
  - Optional sanitized-file preference for external corpus path.

No changes required in [scripts/fetch_external_resources.py](scripts/fetch_external_resources.py) for this scoped pass.

---

### Verification Commands (Planned)
1. Run sanitization dry-run:
	- python scripts/sanitize_external_corpus.py --dry-run
2. Generate sanitized corpus:
	- python scripts/sanitize_external_corpus.py
3. Chunk only sanitized corpus:
	- python scripts/chunk_data.py data/processed/external_ophthalmic_resources_sanitized.md
4. Optional ingestion compatibility smoke check (only if artifacts regenerated):
	- python scripts/ingest_db.py

---

### Decisions Needed Before Implementation
1. Sanitized output filename:
	- Use [data/processed/external_ophthalmic_resources_sanitized.md](data/processed/external_ophthalmic_resources_sanitized.md) by default, or a different naming convention.
2. Chunker behavior if sanitized file is missing:
	- Hard fail, or continue with warning.
3. Header plausibility thresholds:
	- Conservative strictness versus lenient acceptance for uncommon but valid long titles.
4. Regression test placement:
	- Lightweight script under [scripts](scripts) or evaluation-style check under [evaluation](evaluation).

---

### Final Scope Statement
- Included: sanitize existing corpus copy, harden parser boundaries, preflight diagnostics, and regression verification.
- Excluded: external refetch, fetch volume changes, and upstream extractor redesign.
