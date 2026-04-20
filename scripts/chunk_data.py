"""
chunker.py
─────────────────────────────────────────────────────────────────────────────
GENERIC: Clean Markdown → Parent Docs + Child Chunks

Works with ANY properly structured markdown that has:
  - # Chapter headers  (H1)
  - ## Section headers (H2)
  - Body text

This module is REUSABLE across diverse document sources (textbooks, guidelines,
papers) — only the upstream markdown extractor is source-specific.

Pipeline:
  1. Parse markdown headers (H1 = chapter, H2 = section)
  2. Build parent docs (one per section) with hierarchical metadata
  3. Semantically split each parent into child chunks (multi-GPU accelerated)
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
import hashlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from tqdm import tqdm

# ─── GPU Configuration ────────────────────────────────────────────────────────
# Explicitly expose all 4 A5000s.  Change if you want a subset.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ─── Embedding Model ─────────────────────────────────────────────────────────
# abhinand/MedEmbed-large-v0.1: Medical fine-tune of BGE-large-en-v1.5,
# trained on clinical data from PubMed Central.
# MUST match the model used in data_ingestion.py and retriever.py.
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"

# Per-GPU batch size fed to each worker process.
# A5000 (24 GB VRAM) can comfortably handle 512 for large models.
PER_GPU_BATCH_SIZE = 512

# Number of sentences per embedding progress step.
# Larger values improve throughput; smaller values give more frequent updates.
EMBED_PROGRESS_SENTENCE_BATCH = 50_000

# Upper bound for sentences held in RAM at once while semantic chunking.
# Keeps peak host memory stable for very large corpora.
STREAM_SENTENCE_BUDGET = 20_000

# Minimum chars for a chunk to be considered meaningful prose
MIN_CHUNK_SIZE = 60

# Threshold percentile for semantic sentence splitting
SEMANTIC_BREAK_PERCENTILE = 75

# External corpus handling
EXTERNAL_CLEAN_BASENAME = "external_ophthalmic_resources_clean.md"
EXTERNAL_SANITIZED_BASENAME = "external_ophthalmic_resources_sanitized.md"
EXTERNAL_ALLOWED_H1 = {
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


# ─── Multi-GPU Embeddings ─────────────────────────────────────────────────────
class MultiGPUEmbeddings:
    """
    Wraps SentenceTransformer's native multi-process pool.

    start_multi_process_pool() spawns one independent process per device,
    each with its own copy of the model loaded directly onto that GPU.
    encode_multi_process() shards the sentence list across those processes
    and collects results — true data-parallel embedding, not fake "auto-select".

    Lifecycle:
        emb = MultiGPUEmbeddings(MODEL_NAME)
        ...
        emb.stop()          # or use as context manager
    """

    def __init__(self, model_name: str, batch_size: int = PER_GPU_BATCH_SIZE):
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            self.devices = ["cpu"]
            print("  ⚠  No CUDA devices found — falling back to CPU.")
        else:
            self.devices = [f"cuda:{i}" for i in range(n_gpus)]
            print(f"  MultiGPUEmbeddings: spawning workers on {self.devices}")

        # One subprocess per GPU — each loads a full model copy.
        # This MUST be called inside if __name__ == "__main__" on Windows/macOS
        # to avoid recursive spawning; Linux fork-start is safe from any scope.
        self.pool = self.model.start_multi_process_pool(target_devices=self.devices)
        print(f"  Worker pool ready ({len(self.devices)} workers).")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings.  Automatically sharded across all GPUs.
        Returns a plain Python list of float lists for LangChain compatibility.
        """
        assert self.pool is not None, "Embedding pool has been stopped."
        arr = self.model.encode(
            sentences=texts,
            pool=self.pool,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return arr.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Single-string embedding for query time (uses GPU 0 directly)."""
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            device=self.devices[0],
        ).tolist()[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Gracefully terminate worker processes."""
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None
            print("  Worker pool stopped.")

    # Support `with MultiGPUEmbeddings(...) as emb:` usage
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


# Global handle — set in main() so worker subprocesses never re-enter this code
embeddings: MultiGPUEmbeddings | None = None


# ─── Utilities ────────────────────────────────────────────────────────────────

def _normalize_for_hash(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedupe_documents(docs: list[Document], label: str) -> list[Document]:
    """Remove duplicate documents by normalized content hash."""
    seen: set[str] = set()
    out: list[Document] = []
    removed = 0

    for doc in docs:
        normalized = _normalize_for_hash(doc.page_content)
        if not normalized:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
        if digest in seen:
            removed += 1
            continue
        seen.add(digest)
        doc.metadata = {**doc.metadata, "content_hash": digest}
        out.append(doc)

    if removed:
        print(f"  Deduplicated {removed} duplicate {label} documents.")
    return out


def _clean_md_text(text: str) -> str:
    """Strip excess whitespace and noise from raw markdown."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no NLTK dependency)."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip()]


def _normalize_source_name_from_path(md_path: str) -> str:
    base = os.path.splitext(os.path.basename(md_path))[0]
    base = re.sub(r'_(clean|sanitized)$', '', base, flags=re.I)
    return base.replace("_", " ").title()


def _is_external_corpus_source(source_name: str) -> bool:
    return "external ophthalmic resources" in source_name.lower()


def _is_plausible_header_text(text: str, level: int) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    max_len = 120 if level == 1 else 140
    if len(t) > max_len:
        return False

    words = t.split()
    max_words = 18 if level == 1 else 20
    if len(words) > max_words:
        return False

    if re.search(r'https?://|www\.', t, flags=re.I):
        return False

    if re.match(r'^\*?(source|url|metadata)\*?\s*:', t, flags=re.I):
        return False

    sentence_punct = len(re.findall(r'[.!?;]', t))
    if sentence_punct >= 3:
        return False

    # Reject sentence-like headings (likely scraped prose lines)
    if re.search(r'[.!?]\s+[A-Z]', t):
        return False

    return True


def _is_allowed_external_h1(text: str) -> bool:
    return text.strip().lower() in EXTERNAL_ALLOWED_H1


def _is_structural_h1(title: str, source_name: str) -> bool:
    if not _is_plausible_header_text(title, level=1):
        return False
    if _is_external_corpus_source(source_name):
        return _is_allowed_external_h1(title)
    return True


def _is_structural_h2(title: str) -> bool:
    return _is_plausible_header_text(title, level=2)


def _print_markdown_profile(md_text: str, source_name: str, md_path: str) -> None:
    lines = md_text.split('\n')
    raw_h1 = 0
    raw_h2 = 0
    accepted_h1 = 0
    accepted_h2 = 0

    for line in lines:
        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        if not heading:
            continue

        level = len(heading.group(1))
        title = heading.group(2).strip()

        if level == 1:
            raw_h1 += 1
            if _is_structural_h1(title, source_name):
                accepted_h1 += 1
        elif level == 2:
            raw_h2 += 1
            if _is_structural_h2(title):
                accepted_h2 += 1

    raw_total = raw_h1 + raw_h2
    accepted_total = accepted_h1 + accepted_h2
    suspicious = raw_total - accepted_total

    size_mb = len(md_text.encode("utf-8", errors="ignore")) / 1_000_000
    print(
        f"  Structure profile [{os.path.basename(md_path)}]: "
        f"size={size_mb:.1f}MB | "
        f"H1 raw/accepted={raw_h1:,}/{accepted_h1:,} | "
        f"H2 raw/accepted={raw_h2:,}/{accepted_h2:,} | "
        f"suspicious={suspicious:,}"
    )

    if suspicious >= 100 and suspicious > accepted_total:
        print(
            "  ⚠ High suspicious heading count detected. "
            "Prefer parsing sanitized corpus for this source."
        )


def _resolve_chunk_input_path(md_path: str) -> str:
    basename = os.path.basename(md_path)
    if basename != EXTERNAL_CLEAN_BASENAME:
        return md_path

    sanitized_path = os.path.join(os.path.dirname(md_path), EXTERNAL_SANITIZED_BASENAME)
    if os.path.exists(sanitized_path):
        print(f"  Using sanitized corpus for parsing: '{sanitized_path}'")
        return sanitized_path

    print(
        "  ⚠ Sanitized corpus not found for external resources. "
        "Proceeding with original file."
    )
    return md_path


# ─── Content Filters ─────────────────────────────────────────────────────────

def filter_noise(docs: list[Document]) -> list[Document]:
    """Remove figure captions, table fragments, and boilerplate."""
    filtered = []
    for doc in docs:
        text = doc.page_content.strip()
        if len(text) < MIN_CHUNK_SIZE:
            continue
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.4:
            continue
        filtered.append(doc)

    removed = len(docs) - len(filtered)
    if removed:
        print(f"  Filtered out {removed} noise fragments.")
    return filtered


# ─── Parse Markdown Structure ─────────────────────────────────────────────────

def parse_markdown_sections(md_text: str, source_name: str) -> list[Document]:
    """
    Parse clean markdown into parent Documents, one per H2 section.

    Hierarchy:
      # H1  → chapter
      ## H2 → section within chapter

    Metadata keys: source, chapter, section, section_path, parent_id, doc_type
    """
    md_text = _clean_md_text(md_text)
    lines = md_text.split('\n')
    parent_docs: list[Document] = []
    current_chapter = source_name
    current_section: str | None = None
    current_content: list[str] = []

    def flush_section() -> None:
        nonlocal current_content
        text = '\n'.join(current_content).strip()
        if len(text) < MIN_CHUNK_SIZE:
            current_content = []
            return

        section = current_section or current_chapter
        breadcrumb = [current_chapter]
        if current_section and current_section != current_chapter:
            breadcrumb.append(current_section)

        parent_docs.append(Document(
            page_content=text,
            metadata={
                "source":       source_name,
                "chapter":      current_chapter,
                "section":      section,
                "section_path": " > ".join(breadcrumb),
                "parent_id":    str(uuid.uuid4()),
                "doc_type":     "parent",
            },
        ))
        current_content = []

    for line in lines:
        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()

            if level == 1 and _is_structural_h1(title, source_name):
                flush_section()
                current_chapter = title
                current_section = None
                continue

            if level == 2 and _is_structural_h2(title):
                flush_section()
                current_section = title
                continue

        if line.strip() or current_content:
            current_content.append(line)

    flush_section()
    return parent_docs


# ─── Semantic Chunking ────────────────────────────────────────────────────────

def create_child_chunks(parent_docs: list[Document]) -> list[Document]:
    """
    Semantically split parent docs into child chunks.

    Strategy:
        1. Stream parent docs into bounded sentence groups to cap RAM usage.
        2. Embed each sentence group via the multi-GPU pool.
        3. Per-doc: compute cosine distances between adjacent sentence embeddings,
         break at the SEMANTIC_BREAK_PERCENTILE threshold.
    """
    assert embeddings is not None, "Global `embeddings` must be initialised before chunking."

    def _append_short_doc_chunk(doc: Document, sents: list[str], out: list[Document]) -> None:
        text = " ".join(sents).strip()
        if len(text) >= MIN_CHUNK_SIZE:
            out.append(Document(
                page_content=text,
                metadata={**doc.metadata, "doc_type": "child", "chunk_id": str(uuid.uuid4())},
            ))

    def _append_semantic_chunks_for_doc(
        doc: Document,
        sents: list[str],
        doc_embs: np.ndarray,
        out: list[Document],
    ) -> None:
        # Embeddings are normalized; adjacent cosine is dot product.
        sims = np.sum(doc_embs[:-1] * doc_embs[1:], axis=1)
        sims = np.clip(sims, -1.0, 1.0)
        distances = 1.0 - sims
        threshold = float(np.percentile(distances, SEMANTIC_BREAK_PERCENTILE))

        current: list[str] = [sents[0]]
        for i, dist in enumerate(distances):
            if dist > threshold:
                text = " ".join(current).strip()
                if len(text) >= MIN_CHUNK_SIZE:
                    out.append(Document(
                        page_content=text,
                        metadata={**doc.metadata, "doc_type": "child", "chunk_id": str(uuid.uuid4())},
                    ))
                current = [sents[i + 1]]
            else:
                current.append(sents[i + 1])

        if current:
            text = " ".join(current).strip()
            if len(text) >= MIN_CHUNK_SIZE:
                out.append(Document(
                    page_content=text,
                    metadata={**doc.metadata, "doc_type": "child", "chunk_id": str(uuid.uuid4())},
                ))

    def _flush_sentence_group(
        grouped_docs: list[tuple[Document, list[str]]],
        grouped_sentences: list[str],
        out: list[Document],
    ) -> None:
        if not grouped_docs:
            return

        assert embeddings is not None, "Global `embeddings` must be initialised before chunking."
        emb = embeddings

        emb_rows: list[list[float]] = []
        for start in range(0, len(grouped_sentences), EMBED_PROGRESS_SENTENCE_BATCH):
            batch = grouped_sentences[start:start + EMBED_PROGRESS_SENTENCE_BATCH]
            emb_rows.extend(emb.embed_documents(batch))

        group_embs = np.array(emb_rows, dtype=np.float32)

        cursor = 0
        for doc, sents in grouped_docs:
            n = len(sents)
            doc_embs = group_embs[cursor:cursor + n]
            cursor += n
            _append_semantic_chunks_for_doc(doc, sents, doc_embs, out)

    print(f"  Extracting + embedding sentences from {len(parent_docs)} parent sections...")
    child_chunks: list[Document] = []
    sentence_group_docs: list[tuple[Document, list[str]]] = []
    sentence_group: list[str] = []
    total_sentences = 0

    for doc in tqdm(parent_docs, desc="  Streaming sentence groups"):
        sents = _split_sentences(doc.page_content) or [doc.page_content]
        total_sentences += len(sents)

        # Trivially short sections do not need semantic splitting.
        if len(sents) <= 2:
            _append_short_doc_chunk(doc, sents, child_chunks)
            continue

        if sentence_group_docs and len(sentence_group) + len(sents) > STREAM_SENTENCE_BUDGET:
            _flush_sentence_group(sentence_group_docs, sentence_group, child_chunks)
            sentence_group_docs.clear()
            sentence_group.clear()

        sentence_group_docs.append((doc, sents))
        sentence_group.extend(sents)

    _flush_sentence_group(sentence_group_docs, sentence_group, child_chunks)

    print(f"  {total_sentences:,} total sentences processed in streaming mode.")
    return child_chunks


# ─── Per-File Orchestrator ────────────────────────────────────────────────────

def chunk_file(md_path: str) -> tuple[list[Document], list[Document]]:
    """Full chunking pipeline for a single clean markdown file."""
    effective_md_path = _resolve_chunk_input_path(md_path)
    source_name = _normalize_source_name_from_path(md_path)

    print(f"\n  Parsing sections from '{effective_md_path}'...")
    with open(effective_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    _print_markdown_profile(md_text, source_name, effective_md_path)

    parent_docs = parse_markdown_sections(md_text, source_name)
    parent_docs = filter_noise(parent_docs)
    parent_docs = dedupe_documents(parent_docs, "parent")
    print(f"  → {len(parent_docs)} parent sections")

    child_chunks = create_child_chunks(parent_docs)
    child_chunks = filter_noise(child_chunks)
    child_chunks = dedupe_documents(child_chunks, "child")
    print(f"  → {len(child_chunks)} child chunks")

    return parent_docs, child_chunks


# ─── Entry Point ──────────────────────────────────────────────────────────────
# The `if __name__ == "__main__"` guard is *mandatory* here.
# SentenceTransformer's multi-process pool uses Python multiprocessing under
# the hood.  On Linux (fork start), spawning from module scope would cause each
# worker to re-execute the top-level code and recursively spawn more workers.
# Keeping pool creation inside this block is the correct pattern.

if __name__ == "__main__":
    import glob

    n_visible = torch.cuda.device_count()
    print(f"CUDA devices visible: {n_visible}")
    for i in range(n_visible):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"\nLoading embedding model ({EMBEDDING_MODEL}) — spawning {n_visible} GPU workers...")
    embeddings = MultiGPUEmbeddings(EMBEDDING_MODEL, batch_size=PER_GPU_BATCH_SIZE)

    # Accept specific files as args, or default to all *_clean.md files
    md_files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("data/processed/*_clean.md"))

    if not md_files:
        print("No *_clean.md files found. Run markdown_extractor.py first.")
        embeddings.stop()
        sys.exit(1)

    all_parents: list[Document] = []
    all_children: list[Document] = []

    try:
        for md_file in tqdm(md_files, desc="Chunking markdown files", unit="file"):
            print(f"\n{'='*60}")
            print(f"Chunking: {md_file}")
            print(f"{'='*60}")
            parents, children = chunk_file(md_file)
            all_parents.extend(parents)
            all_children.extend(children)
    finally:
        # Always shut down worker pool, even if an exception occurs mid-run
        embeddings.stop()

    # Final global dedupe across all sources
    all_parents = dedupe_documents(all_parents, "parent")
    all_children = dedupe_documents(all_children, "child")

    # Persist
    print(f"\n{'='*60}")
    print("Saving artifacts...")
    os.makedirs("data/cache", exist_ok=True)

    with open("data/cache/parent_docs.pkl", "wb") as f:
        pickle.dump(all_parents, f)
    print(f"  Saved: parent_docs.pkl  ({len(all_parents)} parent docs)")

    with open("data/cache/chunks.pkl", "wb") as f:
        pickle.dump(all_children, f)
    print(f"  Saved: chunks.pkl  ({len(all_children)} child chunks)")

    # Spot-check samples
    for label, docs in [("CHILD CHUNK", all_children), ("PARENT DOC", all_parents)]:
        if docs:
            s = docs[0]
            print(f"\n--- SAMPLE {label} ---")
            print(f"Metadata: {s.metadata}")
            print(f"Content:\n{s.page_content[:300]}...")

    print(f"\n✓ Chunking complete. Run data_ingestion.py next.")