"""
chunker.py
─────────────────────────────────────────────────────────────────────────────
GENERIC: Clean Markdown → Parent Docs + Child Chunks

Works with ANY properly structured markdown that has:
  - # Chapter headers  (H1)
  - ## Section headers  (H2)
  - Body text

This module is REUSABLE across diverse document sources (textbooks, guidelines,
papers) — only the upstream markdown extractor is source-specific.

Pipeline:
  1. Parse markdown headers (H1 = chapter, H2 = section)
  2. Build parent docs (one per section) with hierarchical metadata
  3. Semantically split each parent into child chunks (GPU-accelerated)
  4. Filter noise fragments
  5. Save parent_docs.pkl + chunks.pkl

Usage:
    python chunker.py                          # chunks all *_clean.md files
    python chunker.py khurana_clean.md          # chunks a specific file
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import sys
import uuid
import pickle

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# ─── GPU Configuration ────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# ─── Embedding Model ─────────────────────────────────────────────────────────
# abhinand/MedEmbed-large-v0.1: Medical fine-tune of BGE-large-en-v1.5,
# trained on clinical data from PubMed Central. MUST match the model used in
# data_ingestion.py and retriever.py.
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"

print(f"Loading embedding model ({EMBEDDING_MODEL}) on GPU...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
)
print("Embedding model loaded.")

# Minimum chars for a chunk to be considered meaningful prose
MIN_CHUNK_SIZE = 60


# ─── Parse Markdown Structure ────────────────────────────────────────────────
def parse_markdown_sections(md_text: str, source_name: str) -> list[Document]:
    """
    Parse clean markdown into parent Documents, one per section.

    Detects:
      # H1 → chapter
      ## H2 → section within chapter

    Returns parent Documents with metadata:
      source, chapter, section, section_path, parent_id, doc_type
    """
    lines = md_text.split('\n')
    parent_docs = []

    current_chapter = source_name   # fallback if no chapter header found
    current_section = None
    current_content = []

    def flush_section():
        """Save accumulated content as a parent doc."""
        nonlocal current_content
        text = '\n'.join(current_content).strip()
        if len(text) < MIN_CHUNK_SIZE:
            current_content = []
            return

        section = current_section or current_chapter
        breadcrumb_parts = [current_chapter]
        if current_section and current_section != current_chapter:
            breadcrumb_parts.append(current_section)

        doc = Document(
            page_content=text,
            metadata={
                "source":       source_name,
                "chapter":      current_chapter,
                "section":      section,
                "section_path": " > ".join(breadcrumb_parts),
                "parent_id":    str(uuid.uuid4()),
                "doc_type":     "parent",
            },
        )
        parent_docs.append(doc)
        current_content = []

    for line in lines:
        # Detect chapter header: # Chapter N: Topic  or  # Topic
        h1_match = re.match(r'^#\s+(.+)$', line)
        if h1_match and not line.startswith('##'):
            flush_section()
            current_chapter = h1_match.group(1).strip()
            current_section = None
            continue

        # Detect section header: ## Section Name
        h2_match = re.match(r'^##\s+(.+)$', line)
        if h2_match and not line.startswith('###'):
            flush_section()
            current_section = h2_match.group(1).strip()
            continue

        # Accumulate body text (skip empty lines at the start of a section)
        if line.strip() or current_content:
            current_content.append(line)

    # Flush the last section
    flush_section()

    return parent_docs


# ─── Semantic Child Chunking ──────────────────────────────────────────────────
def create_child_chunks(parent_docs: list[Document]) -> list[Document]:
    """
    Split each parent doc into smaller semantic child chunks.
    Each child inherits the parent's metadata + gets its own UUID link.
    """
    print(f"  Semantically splitting {len(parent_docs)} parent sections on GPU...")
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    child_chunks = []
    for parent in parent_docs:
        try:
            children = semantic_splitter.split_documents([parent])
        except Exception:
            # If semantic splitting fails (e.g. too short), treat parent as single chunk
            children = [parent]

        for child in children:
            text = child.page_content.strip()
            if len(text) < MIN_CHUNK_SIZE:
                continue

            child.metadata = {
                **parent.metadata,
                "doc_type": "child",
            }
            child_chunks.append(child)

    return child_chunks


# ─── Content Filters ─────────────────────────────────────────────────────────
def filter_noise(docs: list[Document]) -> list[Document]:
    """Remove documents that are mostly figure captions, tables, or boilerplate."""
    filtered = []
    for doc in docs:
        text = doc.page_content.strip()

        # Skip if mostly special characters (table/figure fragments)
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.4:
            continue

        # Skip if too short
        if len(text) < MIN_CHUNK_SIZE:
            continue

        filtered.append(doc)

    removed = len(docs) - len(filtered)
    if removed > 0:
        print(f"  Filtered out {removed} noise fragments.")
    return filtered


# ─── Main Orchestrator ────────────────────────────────────────────────────────
def chunk_file(md_path: str) -> tuple[list[Document], list[Document]]:
    """
    Full chunking pipeline for a single clean markdown file.
    Returns (parent_docs, child_chunks).
    """
    source_name = os.path.splitext(os.path.basename(md_path))[0]
    source_name = source_name.replace("_clean", "").title()

    print(f"\n  Parsing sections from '{md_path}'...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    parent_docs = parse_markdown_sections(md_text, source_name)
    parent_docs = filter_noise(parent_docs)
    print(f"  → {len(parent_docs)} parent sections")

    child_chunks = create_child_chunks(parent_docs)
    child_chunks = filter_noise(child_chunks)
    print(f"  → {len(child_chunks)} child chunks")

    return parent_docs, child_chunks


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    # Accept specific files as args, or default to all *_clean.md files
    if len(sys.argv) > 1:
        md_files = sys.argv[1:]
    else:
        md_files = sorted(glob.glob("data/processed/*_clean.md"))

    if not md_files:
        print("No *_clean.md files found. Run markdown_extractor.py first.")
        sys.exit(1)

    all_parents = []
    all_children = []

    for md_file in md_files:
        print(f"\n{'='*60}")
        print(f"Chunking: {md_file}")
        print(f"{'='*60}")

        parents, children = chunk_file(md_file)
        all_parents.extend(parents)
        all_children.extend(children)

    # Save outputs
    print(f"\n{'='*60}")
    print("Saving artifacts...")

    with open("data/cache/parent_docs.pkl", "wb") as f:
        pickle.dump(all_parents, f)
    print(f"  Saved: parent_docs.pkl  ({len(all_parents)} parent docs)")

    with open("data/cache/chunks.pkl", "wb") as f:
        pickle.dump(all_children, f)
    print(f"  Saved: chunks.pkl  ({len(all_children)} child chunks)")

    # Inspect samples
    if all_children:
        s = all_children[0]
        print(f"\n--- SAMPLE CHILD CHUNK ---")
        print(f"Metadata: {s.metadata}")
        print(f"Content:\n{s.page_content[:300]}...")

    if all_parents:
        s = all_parents[0]
        print(f"\n--- SAMPLE PARENT DOC ---")
        print(f"Metadata: {s.metadata}")
        print(f"Content:\n{s.page_content[:300]}...")

    print(f"\n✓ Chunking complete. Run data_ingestion.py next.")
