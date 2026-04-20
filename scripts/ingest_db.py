"""
data_ingestion.py
─────────────────────────────────────────────────────────────────────────────
Ingests preprocessed chunks into the retrieval stack:

  1. ChromaDB           — dense vector store (MedEmbed-large embeddings)
  2. BM25Retriever      — sparse keyword index (rank_bm25)
  3. EnsembleRetriever  — RRF fusion of 1 + 2 (60% dense, 40% sparse)
  4. ParentDocStore     — in-memory dict of full parent sections keyed by parent_id

Run AFTER chunker.py has produced:
  - chunks.pkl       (child chunks with parent_id + section_path metadata)
  - parent_docs.pkl  (full header sections for LLM context)

Multi-GPU strategy:
  - All chunk embeddings computed in ONE encode_multi_process() call across 4 GPUs
  - Pre-computed vectors inserted directly into ChromaDB (no re-embedding in the loop)
  - BM25 corpus tokenization parallelised across all CPU cores
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ─── GPU Configuration ────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ─── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "abhinand/MedEmbed-large-v0.1"
CHROMA_DB_PATH   = "./data/vectorstore/ophthalmology_db"
CHROMA_COLL_NAME = "ophthalmic_child_chunks"
PER_GPU_BATCH    = 512          # per-worker batch size fed to each A5000
CHROMA_BATCH     = 4000         # max docs per ChromaDB .add() call
EMBED_PROGRESS_BATCH = 50_000   # chunks per embedding progress step


# ─── Multi-GPU Embeddings (LangChain-compatible) ──────────────────────────────
class MultiGPUEmbeddings(Embeddings):
    """
    LangChain Embeddings subclass backed by SentenceTransformer's multi-process
    pool.  One subprocess per GPU; encode_multi_process() shards the input list
    across all workers for true data-parallel throughput.

    Usage:
        with MultiGPUEmbeddings(MODEL_NAME) as emb:
            vectors = emb.embed_documents(texts)   # multi-GPU
            qvec    = emb.embed_query(query)        # GPU 0, single string
    """

    def __init__(self, model_name: str, batch_size: int = PER_GPU_BATCH):
        self.batch_size = batch_size
        self.model      = SentenceTransformer(model_name)

        n_gpus      = torch.cuda.device_count()
        self.devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus else ["cpu"]
        print(f"  MultiGPUEmbeddings: workers on {self.devices}")

        self.pool = self.model.start_multi_process_pool(target_devices=self.devices)
        print(f"  Pool ready ({len(self.devices)} workers).")

    # ── LangChain Embeddings interface ────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert self.pool is not None, "Embedding pool has been stopped."
        pool = self.pool
        arr = self.model.encode(
            sentences=texts,
            pool=pool,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return arr.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            device=self.devices[0],
        ).tolist()[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None
            print("  Worker pool stopped.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


# ─── ChromaDB: Pre-computed Insertion ────────────────────────────────────────

def build_chromadb(
    chunks: list[Document],
    all_embeddings: np.ndarray,
    chroma_client,
    embeddings_obj: MultiGPUEmbeddings,
) -> Chroma:
    """
    Insert pre-computed embeddings directly into ChromaDB.
    Bypasses re-embedding entirely — Chroma just stores the vectors we give it.
    Returns a LangChain Chroma object wired to the same collection for querying.
    """
    # Purge stale collection
    try:
        chroma_client.delete_collection(CHROMA_COLL_NAME)
        print("  Cleared existing ChromaDB collection.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=CHROMA_COLL_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    print(f"  Inserting {total:,} chunks into ChromaDB (batch={CHROMA_BATCH})...")

    for batch_start in tqdm(range(0, total, CHROMA_BATCH), desc="  ChromaDB insert"):
        batch_end  = min(batch_start + CHROMA_BATCH, total)
        batch      = chunks[batch_start:batch_end]
        batch_embs = all_embeddings[batch_start:batch_end]

        # Use chunk_id from metadata as the stable ChromaDB ID
        ids = [
            doc.metadata.get("chunk_id", str(uuid.uuid4()))
            for doc in batch
        ]

        # Strip any keys ChromaDB can't serialise (e.g. nested dicts)
        safe_meta = []
        for doc in batch:
            m = {k: str(v) for k, v in doc.metadata.items()}
            safe_meta.append(m)

        collection.add(
            embeddings=batch_embs.tolist(),   # pre-computed — no GPU work here
            documents=[d.page_content for d in batch],
            metadatas=safe_meta,
            ids=ids,
        )

    print(f"  ChromaDB collection '{CHROMA_COLL_NAME}' ready.")

    # Wrap in LangChain object so .as_retriever() works at query time.
    # embed_query() in embeddings_obj is called only for incoming queries (GPU 0).
    return Chroma(
        client=chroma_client,
        collection_name=CHROMA_COLL_NAME,
        embedding_function=embeddings_obj,
    )


# ─── BM25: Parallel Tokenisation ─────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Module-level function required for multiprocessing pickling."""
    return text.lower().split()


def build_bm25(chunks: list[Document]) -> BM25Retriever:
    """
    Build BM25 retriever with corpus tokenised in parallel across all CPU cores.
    BM25Retriever.from_documents() tokenises serially; we do it ourselves first.
    """
    n_workers = cpu_count()
    print(f"  Tokenising {len(chunks):,} docs across {n_workers} CPU cores...")

    texts = [doc.page_content for doc in chunks]
    with Pool(n_workers) as pool:
        tokenized = list(
            tqdm(pool.imap(_tokenize, texts, chunksize=500),
                 total=len(texts), desc="  BM25 tokenise")
        )

    # Build retriever from pre-tokenized corpus
    retriever        = BM25Retriever.from_texts(texts, metadatas=[d.metadata for d in chunks])
    retriever.k      = 10
    # Overwrite the internal corpus with our pre-tokenized version so BM25Okapi
    # doesn't re-tokenize internally
    retriever.vectorizer.corpus = tokenized   # rank_bm25 stores corpus here
    return retriever


# ─── Parent DocStore ──────────────────────────────────────────────────────────

def build_parent_store(parent_docs: list[Document]) -> dict[str, Document]:
    return {doc.metadata["parent_id"]: doc for doc in parent_docs}


# ─── Hybrid Retrieval Helper ──────────────────────────────────────────────────

def make_retrieve_fn(
    hybrid_retriever: EnsembleRetriever,
    parent_store: dict[str, Document],
):
    """Returns a closure that runs child retrieval + parent expansion."""
    def retrieve_with_parents(query: str, k: int = 5) -> list[Document]:
        child_results = hybrid_retriever.invoke(query)[:k]
        seen: set[str] = set()
        parents: list[Document] = []
        for child in child_results:
            p_id = child.metadata.get("parent_id")
            if p_id and p_id not in seen:
                parent = parent_store.get(p_id)
                if parent:
                    parents.append(parent)
                    seen.add(p_id)
        return parents
    return retrieve_with_parents


# ─── Entry Point ──────────────────────────────────────────────────────────────
# `if __name__ == "__main__"` guard is mandatory: MultiGPUEmbeddings uses
# Python multiprocessing, and pool workers must not re-execute top-level code.

if __name__ == "__main__":

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("Loading preprocessed artifacts...")
    with open("data/cache/chunks.pkl", "rb") as f:
        child_chunks: list[Document] = pickle.load(f)
    with open("data/cache/parent_docs.pkl", "rb") as f:
        parent_docs: list[Document] = pickle.load(f)
    print(f"  {len(child_chunks):,} child chunks | {len(parent_docs):,} parent docs")

    n_visible = torch.cuda.device_count()
    print(f"\nCUDA devices visible: {n_visible}")
    for i in range(n_visible):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    with MultiGPUEmbeddings(EMBEDDING_MODEL, batch_size=PER_GPU_BATCH) as med_embeddings:

        # ── 1. Embed ALL chunks in one multi-GPU call ─────────────────────────
        # This is the only GPU-heavy step. All four A5000s work in parallel,
        # each processing its shard at batch_size=512.
        print(f"\nEmbedding {len(child_chunks):,} chunks across {n_visible} GPUs...")
        chunk_texts = [doc.page_content for doc in child_chunks]
        emb_rows: list[list[float]] = []
        for start in tqdm(
            range(0, len(chunk_texts), EMBED_PROGRESS_BATCH),
            desc="  Embedding chunks",
            unit="batch",
        ):
            batch = chunk_texts[start:start + EMBED_PROGRESS_BATCH]
            emb_rows.extend(med_embeddings.embed_documents(batch))

        all_embeddings = np.array(emb_rows, dtype=np.float32)
        print(f"  Embedding matrix: {all_embeddings.shape}  (chunks × dims)")

        # ── 2. ChromaDB — insert pre-computed vectors (no re-embedding) ───────
        print("\nBuilding ChromaDB dense vector store...")
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        chroma_client    = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        dense_vectorstore = build_chromadb(
            child_chunks, all_embeddings, chroma_client, med_embeddings
        )

        # ── 3. BM25 — parallel CPU tokenisation ───────────────────────────────
        print("\nBuilding BM25 sparse retriever...")
        bm25_retriever = build_bm25(child_chunks)
        with open("data/cache/bm25_retriever.pkl", "wb") as f:
            pickle.dump(bm25_retriever, f)
        print("  BM25 saved → bm25_retriever.pkl")

        # ── 4. EnsembleRetriever ──────────────────────────────────────────────
        print("\nWiring EnsembleRetriever (BM25 40% + Dense 60%)...")
        dense_retriever  = dense_vectorstore.as_retriever(search_kwargs={"k": 10})
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.4, 0.6],
        )
        print("  EnsembleRetriever ready.")

        # ── 5. Parent DocStore ────────────────────────────────────────────────
        parent_store = build_parent_store(parent_docs)
        with open("data/cache/parent_store.pkl", "wb") as f:
            pickle.dump(parent_store, f)
        print(f"  ParentDocStore saved: {len(parent_store):,} entries → parent_store.pkl")

        retrieve_with_parents = make_retrieve_fn(hybrid_retriever, parent_store)

        # ── Smoke Test ────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SMOKE TEST — Hybrid Retrieval")
        print("=" * 60)

        test_queries = [
            "what is phacoemulsification",
            "my eye hurts in bright light",
            "ranibizumab injection for wet AMD",
        ]
        for q in test_queries:
            print(f"\nQuery: '{q}'")
            results = retrieve_with_parents(q, k=3)
            for i, doc in enumerate(results, 1):
                sp  = doc.metadata.get("section_path", "N/A")
                src = doc.metadata.get("source",       "N/A")
                print(f"  [{i}] [{src}] {sp}")
                print(f"       {doc.page_content[:150].strip()}...")

    # Pool is stopped here by the context manager __exit__

    print("\n✓ data_ingestion.py complete.")
    print("  Artifacts: ophthalmology_db/ | parent_store.pkl | bm25_retriever.pkl")