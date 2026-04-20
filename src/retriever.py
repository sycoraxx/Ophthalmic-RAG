"""
retriever.py — Handles all search and ranking logic.
────────────────────────────────────────────────────
Loads BM25, ChromaDB (dense), ParentDocStore, and MedCPT Cross-Encoder.
Provides a `search()` method to replace the old `retrieve()` logic.

Models:
  Embeddings:  abhinand/MedEmbed-large-v0.1  (medical fine-tune of BGE-large)
  Reranker:    ncbi/MedCPT-Cross-Encoder     (255M PubMed query-article pairs)
"""

import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

MEDEMBED_MODEL   = "abhinand/MedEmbed-large-v0.1"
RERANKER_MODEL   = "ncbi/MedCPT-Cross-Encoder"
CHROMA_DB_PATH   = "./data/vectorstore/ophthalmology_db"


class RetinaRetriever:
    """Manages the hybrid retrieval and re-ranking stack."""
    
    def __init__(self):
        self._load_reranker()
        self._load_retriever()

    def _load_reranker(self):
        print(f"Loading MedCPT cross-encoder reranker ({RERANKER_MODEL})...")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL
        )
        self.reranker_model.eval()
        # Move to GPU if available
        self.reranker_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.reranker_model.to(self.reranker_device)
        print("MedCPT reranker loaded.")

    def _load_retriever(self):
        print(f"Loading MedEmbed ({MEDEMBED_MODEL}) + ChromaDB dense retriever...")
        embed_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        med_embeddings = HuggingFaceEmbeddings(
            model_name=MEDEMBED_MODEL,
            model_kwargs={"device": embed_device},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
        )
        dense_vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=med_embeddings,
        )
        dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": 10})

        print("Loading BM25 sparse retriever...")
        with open("./data/cache/bm25_retriever.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 10

        print("Creating hybrid EnsembleRetriever...")
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.4, 0.6],
        )

        print("Loading parent docstore...")
        with open("./data/cache/parent_store.pkl", "rb") as f:
            self.parent_store = pickle.load(f)

        print("Retriever stack ready.")

    def _is_noisy_aao_doc(self, doc: Document) -> bool:
        chapter = str(doc.metadata.get("chapter", "")).lower()
        section = str(doc.metadata.get("section", "")).lower()
        text = (doc.page_content or "").lower()

        if chapter != "aao preferred practice patterns":
            return False

        noisy_sections = (
            "supplement",
            "committee/disclosures",
            "keeping current",
            "revisions & publication schedule",
            "ppp translations",
            "about preferred practice patterns",
        )
        if any(tag in section for tag in noisy_sections):
            return True

        noisy_text_signals = (
            "literature searches for this ppp",
            "my dashboard my education find an ophthalmologist",
            "all rights reserved",
            "registered trademarks of the american academy of ophthalmology",
        )
        return any(sig in text for sig in noisy_text_signals)

    def _rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        """Score each (query, document) pair with MedCPT Cross-Encoder."""
        if not docs:
            return docs
        if top_k <= 0:
            return []

        # Let the tokenizer handle truncation in token space; character clipping
        # can discard relevant terms and destabilize ranking.
        pairs = [[query, doc.page_content] for doc in docs]

        with torch.no_grad():
            encoded = self.reranker_tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.reranker_device)
            logits = self.reranker_model(**encoded).logits

            # MedCPT checkpoints may emit either:
            # - [batch] / [batch, 1] regression-style scores
            # - [batch, 2] (or more) classification logits
            # Always collapse to one scalar relevance value per document.
            if logits.dim() == 1:
                relevance = logits
            elif logits.size(-1) == 1:
                relevance = logits.squeeze(dim=-1)
            else:
                # For classification heads, use probability of the final label
                # as relevance (for binary heads, this is the positive class).
                relevance = torch.softmax(logits, dim=-1)[..., -1]

            if relevance.dim() == 0:
                relevance = relevance.unsqueeze(0)
            scores = relevance.cpu().tolist()

        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        ranked_docs: list[Document] = []
        for rank, (score, doc) in enumerate(scored_docs[:top_k], start=1):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = rank
            ranked_docs.append(doc)

        return ranked_docs

    def search(self, query: str, k: int = 5, verbose: bool = True) -> list[Document]:
        """
        Hybrid retrieval → Re-ranking → Parent expansion.
        Returns up to `k` deduplicated parent Documents.
        """
        # Over-fetch children from hybrid retriever
        child_hits = self.hybrid_retriever.invoke(query)[:k * 4]

        # Drop known AAO boilerplate/supplement chunks before re-ranking.
        filtered_child_hits = [doc for doc in child_hits if not self._is_noisy_aao_doc(doc)]
        if filtered_child_hits:
            if verbose and len(filtered_child_hits) != len(child_hits):
                print(
                    f"[Retriever] Filtered {len(child_hits) - len(filtered_child_hits)} "
                    "noisy AAO child chunks."
                )
            child_hits = filtered_child_hits

        # Re-rank children by cross-encoder relevance
        if verbose:
            print(f"[Retriever] Re-ranking {len(child_hits)} child hits...")
        reranked_children = self._rerank(query, child_hits, top_k=k * 2)

        # Expand to parent docs (deduplicated)
        seen, parents = set(), []
        for child in reranked_children:
            p_id = child.metadata.get("parent_id")
            if p_id and p_id not in seen:
                parent_doc = self.parent_store.get(p_id)
                if parent_doc:
                    if self._is_noisy_aao_doc(parent_doc):
                        continue
                    parents.append(parent_doc)
                    seen.add(p_id)
            if len(parents) >= k:
                break

        if verbose:
            print(f"[Retriever] Retrieved {len(parents)} parent sections:")
            for i, doc in enumerate(parents, 1):
                sp  = doc.metadata.get("section_path", "N/A")
                src = doc.metadata.get("source", "N/A")
                print(f"  [{i}] [{src}] {sp}")

        return parents

