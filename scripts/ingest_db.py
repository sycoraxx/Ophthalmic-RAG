"""
data_ingestion.py
─────────────────────────────────────────────────────────────────────────────
Ingests preprocessed chunks into the retrieval stack:

  1. ChromaDB           — dense vector store (MedEmbed-large embeddings)
  2. BM25Retriever      — sparse keyword index (rank_bm25)
  3. EnsembleRetriever  — RRF fusion of 1 + 2 (60% dense, 40% sparse)
  4. ParentDocStore     — in-memory dict of full parent sections keyed by parent_id

Run AFTER data_preprocessing.py has produced:
  - chunks.pkl       (child chunks with parent_id + section_path metadata)
  - parent_docs.pkl  (full header sections for LLM context)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ─── GPU Configuration ────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# ─── Load Preprocessed Artifacts ─────────────────────────────────────────────
print("Loading preprocessed artifacts...")
with open("data/cache/chunks.pkl", "rb") as f:
    child_chunks = pickle.load(f)

with open("data/cache/parent_docs.pkl", "rb") as f:
    parent_docs = pickle.load(f)

print(f"  Loaded {len(child_chunks)} child chunks and {len(parent_docs)} parent docs.")

# ─── Medical Embedding Model ──────────────────────────────────────────────────
# MUST match the model used in chunk_data.py and retriever.py for consistent vector space.
print("\nLoading MedEmbed medical embedding model on GPU...")
med_embeddings = HuggingFaceEmbeddings(
    model_name="abhinand/MedEmbed-large-v0.1",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
)
print("Embedding model loaded.")

# ─── 1. ChromaDB: Dense Vector Store ─────────────────────────────────────────
print("\nBuilding ChromaDB dense vector store...")
CHROMA_DB_PATH = "./data/vectorstore/ophthalmology_db"

# Purge existing collection to avoid stale data on re-runs
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    chroma_client.delete_collection("ophthalmic_child_chunks")
    print("  Cleared existing ChromaDB collection.")
except Exception:
    pass

# Build fresh Chroma vectorstore from child chunks (batched to stay under max 5461)
BATCH_SIZE = 4000
dense_vectorstore = Chroma.from_documents(
    documents=child_chunks[:BATCH_SIZE],
    embedding=med_embeddings,
    collection_name="ophthalmic_child_chunks",
    persist_directory=CHROMA_DB_PATH,
)
# Add remaining chunks in batches
for i in range(BATCH_SIZE, len(child_chunks), BATCH_SIZE):
    batch = child_chunks[i:i + BATCH_SIZE]
    dense_vectorstore.add_documents(batch)
    print(f"  Added batch {i//BATCH_SIZE + 1}: {len(batch)} chunks")
print(f"  ChromaDB populated with {len(child_chunks)} child chunks.")

# ─── 2. BM25: Sparse Keyword Index ───────────────────────────────────────────
print("\nBuilding BM25 sparse retriever...")
bm25_retriever = BM25Retriever.from_documents(child_chunks)
bm25_retriever.k = 10   # fetch top-10 candidates from BM25

# Persist the BM25 retriever for reuse (avoid rebuilding on every query)
with open("data/cache/bm25_retriever.pkl", "wb") as f:
    pickle.dump(bm25_retriever, f)
print("  BM25 index built and saved to bm25_retriever.pkl")

# ─── 3. EnsembleRetriever: Hybrid RRF Fusion ─────────────────────────────────
print("\nWiring EnsembleRetriever (BM25 40% + Dense 60%)...")
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": 10})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],   # 40% keyword precision, 60% semantic recall
)
print("  EnsembleRetriever ready.")

# ─── 4. Parent DocStore ───────────────────────────────────────────────────────
# Build an in-memory dict: { parent_id -> parent Document }
# Used at query time to fetch the full clinical section for the LLM.
parent_store = {doc.metadata["parent_id"]: doc for doc in parent_docs}

with open("data/cache/parent_store.pkl", "wb") as f:
    pickle.dump(parent_store, f)
print(f"\n  ParentDocStore saved: {len(parent_store)} entries → parent_store.pkl")


# ─── Helper: Retrieve with Parent Expansion ───────────────────────────────────
def retrieve_with_parents(query: str, k: int = 5):
    """
    1. Run hybrid retrieval to get top-k child chunks.
    2. Expand each child chunk to its full parent section.
    3. Deduplicate (multiple children may share the same parent).
    Returns a list of parent Documents (full clinical sections).
    """
    child_results = hybrid_retriever.invoke(query)[:k]

    seen_parent_ids = set()
    parent_results  = []
    for child in child_results:
        p_id = child.metadata.get("parent_id")
        if p_id and p_id not in seen_parent_ids:
            parent_doc = parent_store.get(p_id)
            if parent_doc:
                parent_results.append(parent_doc)
                seen_parent_ids.add(p_id)

    return parent_results


# ─── Smoke Test ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SMOKE TEST — Hybrid Retrieval (no query refinement yet)")
print("="*60)

test_queries = [
    "what is phacoemulsification",
    "my eye hurts in bright light",
    "ranibizumab injection for wet AMD",
]

for q in test_queries:
    print(f"\nQuery: '{q}'")
    results = retrieve_with_parents(q, k=3)
    for i, doc in enumerate(results, 1):
        sp = doc.metadata.get("section_path", "N/A")
        src = doc.metadata.get("source", "N/A")
        print(f"  [{i}] [{src}] {sp}")
        print(f"       {doc.page_content[:150].strip()}...")

print("\n✓ data_ingestion.py complete.")
print("  Artifacts: ophthalmology_db/  |  parent_store.pkl  |  bm25_retriever.pkl")