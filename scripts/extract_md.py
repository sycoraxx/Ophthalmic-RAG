"""
markdown_extractor.py
─────────────────────────────────────────────────────────────────────────────
SOURCE-SPECIFIC: PDF → Clean Markdown

Converts ophthalmology textbook PDFs into clean, well-structured markdown
with proper # and ## headers. Handles OCR noise unique to each book.

**Book-specific patterns detected:**

Kanski:
  - Real chapter headers exist as `#### Topic Name` (21 unique chapters)
  - Running page headers: "CHAPTER\\n\\nExamination Techniques 5" (noise)
  - Page numbers emitted as `#### 1`, `#### 2` (noise)

Khurana:
  - No useful markdown headers (all OCR garbage)
  - Running page headers: "CHAPTER N TOPIC_NAME **PageNum**"
  - Section headings in bold uppercase: "**VISUAL PATHWAY**"

Output: *_clean.md files ready for the GENERIC chunker (chunker.py)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import fitz
import pymupdf4llm


# ─── PDF → Raw Markdown ──────────────────────────────────────────────────────
def extract_markdown_from_pdf(pdf_path: str, start_page: int = 0) -> str:
    """Extract raw markdown from a PDF, skipping front-matter pages."""
    print(f"  Extracting '{pdf_path}' from page {start_page}...")
    total_pages = fitz.open(pdf_path).page_count
    pages = list(range(start_page, total_pages))
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        pages=pages,
        write_images=False,     # skip image extraction (causes crashes)
    )
    return md_text


# ─── Kanski-Specific Cleaning ────────────────────────────────────────────────
def clean_kanski(md_text: str) -> str:
    """
    Kanski has:
      - Real chapter headers: `#### Topic Name` (e.g. "#### Glaucoma")
      - Junk headers: `#### 1`, `#### 2` (page numbers as H4)
      - Running page headers: "CHAPTER\\n\\nTopic PageNum"
    """
    print("  Cleaning Kanski markdown...")

    # 1. Extract real #### chapter names BEFORE stripping headers
    #    These are #### lines with alphabetic content (not just numbers)
    real_chapters = re.findall(
        r'^####\s+([A-Za-z][A-Za-z\s\-\']+)$',
        md_text, flags=re.MULTILINE
    )
    print(f"  Found {len(real_chapters)} real chapter headers: {real_chapters[:5]}...")

    # 2. Remove ALL markdown headers that are just numbers: #### 1, #### 2, etc.
    md_text = re.sub(r'^#{1,6}\s+\d+\s*$', '', md_text, flags=re.MULTILINE)

    # 3. Promote real #### chapter headers to # (H1)
    def promote_chapter(m):
        name = m.group(1).strip()
        return f"\n# {name}\n"

    md_text = re.sub(
        r'^####\s+([A-Za-z][A-Za-z\s\-\'\,\(\)]+)$',
        promote_chapter,
        md_text,
        flags=re.MULTILINE,
    )

    # 4. Remove remaining noise headers (any # lines that aren't our promoted ones)
    md_text = re.sub(r'^#{1,6}\s+[^A-Za-z#].*$', '', md_text, flags=re.MULTILINE)

    # 5. Remove "CHAPTER" running page header lines and the following topic line
    md_text = re.sub(r'^CHAPTER\s*$', '', md_text, flags=re.MULTILINE)
    # Remove lines like "Examination Techniques 5" (topic + page number)
    md_text = re.sub(r'^[A-Z][a-z].*\d+\s*$', '', md_text, flags=re.MULTILINE)

    # 6. Detect section headings — capitalized lines followed by body text
    #    In Kanski, section headings appear as standalone capitalized words
    #    e.g. "Introduction", "Technique", "Principles"
    def to_section_header(m):
        return f"\n## {m.group(1).strip()}\n"

    md_text = re.sub(
        r'^([A-Z][a-z]+(?:\s+[a-z]+)*)\s*$',
        to_section_header,
        md_text,
        flags=re.MULTILINE,
    )

    # 7. Remove figure captions
    md_text = re.sub(
        r'^(?:_?\*?\*?)?Fig\.?\s*\*?\*?\s*[\d\.]+.*$',
        '', md_text, flags=re.MULTILINE
    )

    # 8. Clean whitespace
    md_text = re.sub(r'\*\*\s*\*\*', '', md_text)
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)

    return md_text.strip()


# ─── Khurana-Specific Cleaning ────────────────────────────────────────────────
def clean_khurana(md_text: str) -> str:
    """
    Khurana has:
      - No useful markdown headers (all OCR noise)
      - Running page headers: "CHAPTER N TOPIC_NAME **PageNum**"
      - Section headings: "**BOLD UPPERCASE TEXT**"
    """
    print("  Cleaning Khurana markdown...")

    # 1. Remove ALL markdown header lines (they are all garbage)
    md_text = re.sub(r'^#{1,6}\s+.*$', '', md_text, flags=re.MULTILINE)

    # 2. Extract unique chapter names from running page headers
    #    Format: "CHAPTER 10 GLAUCOMA **253**" or "CHAPTER 10 GLAUCOMA 253"
    chapter_matches = re.findall(
        r'^CHAPTER\s+(\d+)\s+(.+?)(?:\s+(?:_?\*\*|\d+\s*$))',
        md_text, flags=re.MULTILINE
    )
    unique_chapters = {}
    for num, name in chapter_matches:
        # Clean trailing bold markers, numbers, OCR artifacts
        name_clean = re.sub(r'[\*_]+$', '', name).strip()
        name_clean = re.sub(r'\s+\d+\s*$', '', name_clean).strip()
        if len(name_clean) > 3 and num not in unique_chapters:
            unique_chapters[num] = name_clean.title()

    print(f"  Found {len(unique_chapters)} unique chapters: {list(unique_chapters.values())[:5]}...")

    # 3. Remove ALL running page header lines
    md_text = re.sub(r'^CHAPTER\s+\d+\s+.+$', '', md_text, flags=re.MULTILINE)

    # 4. Inject proper # chapter headers at first occurrence of chapter content
    #    We do this by finding transitions — where the chapter topic first appears
    for num, name in sorted(unique_chapters.items(), key=lambda x: int(x[0])):
        # Find the first substantial content after the chapter header was removed
        # Look for bold section headings that might start the chapter
        pattern = rf'(?:^Section\s+.*$\s*)*'
        first_occurrence = md_text.find(f"CHAPTER {num}")
        if first_occurrence == -1:
            # If all running headers were removed, insert at the position
            # where we first saw this chapter number
            pass

    # Since running headers are already removed, we need a different approach:
    # Split by where the chapter content actually flows. 
    # Use the section headings as anchors and group them under chapters.

    # 5. Promote bold uppercase lines to ## section headers
    def bold_to_header(m):
        title = m.group(1).strip()
        return f"\n## {title.title()}\n"

    md_text = re.sub(
        r'^\*\*([A-Z][A-Z\s\(\)\-\,\.]{3,})\*\*$',
        bold_to_header,
        md_text,
        flags=re.MULTILINE,
    )

    # 6. Now inject chapter headers based on the chapter_matches positions
    #    We'll use the FIRST line number where each chapter appeared
    chapter_first_pos = {}
    for match in re.finditer(
        r'^CHAPTER\s+(\d+)\s+',
        pymupdf4llm.to_markdown.__doc__ or '',  # dummy — won't match
        flags=re.MULTILINE
    ):
        pass  # placeholder

    # Instead, let's do a simpler approach: find known chapter topic names
    # as standalone text and inject chapter headers before them
    for num, name in sorted(unique_chapters.items(), key=lambda x: int(x[0])):
        # Look for the topic name appearing standalone or as a Section header
        # This is imperfect but catches most transitions
        search_pat = re.escape(name.upper())
        match = re.search(rf'## {re.escape(name.title())}', md_text)
        if match:
            # Insert a chapter header just before this section
            insert_pos = match.start()
            header = f"\n# Chapter {num}: {name}\n\n"
            md_text = md_text[:insert_pos] + header + md_text[insert_pos:]

    # 7. Remove figure captions
    md_text = re.sub(
        r'^(?:_?\*?\*?)?Fig\.?\s*\*?\*?\s*[\d\.]+.*$',
        '', md_text, flags=re.MULTILINE
    )

    # 8. Clean
    md_text = re.sub(r'\*\*\s*\*\*', '', md_text)
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)

    return md_text.strip()


# ─── Main ─────────────────────────────────────────────────────────────────────
BOOKS = [
    {
        "pdf_path": "data/knowledge_base/kanski.pdf",
        "start_page": 14,
        "source_name": "Kanski",
        "cleaner": clean_kanski,
    },
    {
        "pdf_path": "data/knowledge_base/khurana.pdf",
        "start_page": 5,
        "source_name": "Khurana",
        "cleaner": clean_khurana,
    },
]

if __name__ == "__main__":
    for book in BOOKS:
        print(f"\n{'='*60}")
        print(f"Processing: {book['source_name']}")
        print(f"{'='*60}")

        raw_md = extract_markdown_from_pdf(
            book["pdf_path"], start_page=book["start_page"]
        )
        clean_md = book["cleaner"](raw_md)

        out_path = f"data/{book['source_name'].lower()}_clean.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(clean_md)

        # Count actual headers in the output
        h1_count = len(re.findall(r'^# ', clean_md, flags=re.MULTILINE))
        h2_count = len(re.findall(r'^## ', clean_md, flags=re.MULTILINE))
        print(f"  ✓ Saved: {out_path}")
        print(f"    {len(clean_md):,} chars, {clean_md.count(chr(10)):,} lines")
        print(f"    {h1_count} H1 chapters, {h2_count} H2 sections")

    print("\n✓ Markdown extraction complete. Run chunker.py next.")
