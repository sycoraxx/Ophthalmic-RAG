"""
retrieval_metrics.py — Retrieval Evaluation Metrics
─────────────────────────────────────────────────────
Provides Recall@k, Precision@k, MRR, and keyword-hit-rate metrics
for evaluating the RAG retrieval component.

Input contract:
    retrieved_docs : list[langchain_core.documents.Document]
    question_entry : dict  (one record from ophthalmic_eval_questions.json)
"""

from __future__ import annotations
from typing import List, Dict, Any


def _doc_is_relevant(doc_text: str, keywords: List[str], threshold: int = 1) -> bool:
    """Return True if at least `threshold` keywords appear in the doc text."""
    doc_lower = doc_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in doc_lower)
    return hits >= threshold


def recall_at_k(
    retrieved_docs: list,
    ground_truth_keywords: List[str],
    k: int = 5,
) -> float:
    """
    Recall@k — fraction of ground-truth keywords covered by top-k docs.

    A keyword is "covered" if it appears in at least one of the top-k
    retrieved document texts.

    Returns: float in [0, 1]
    """
    if not ground_truth_keywords or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    combined = " ".join(doc.page_content.lower() for doc in top_k)
    hits = sum(1 for kw in ground_truth_keywords if kw.lower() in combined)
    return hits / len(ground_truth_keywords)


def precision_at_k(
    retrieved_docs: list,
    ground_truth_keywords: List[str],
    k: int = 5,
) -> float:
    """
    Precision@k — fraction of top-k docs that contain at least one keyword.

    Returns: float in [0, 1]
    """
    if not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    relevant = sum(
        1 for doc in top_k
        if _doc_is_relevant(doc.page_content, ground_truth_keywords)
    )
    return relevant / len(top_k)


def mean_reciprocal_rank(
    retrieved_docs: list,
    ground_truth_keywords: List[str],
) -> float:
    """
    MRR — reciprocal rank of the first relevant document.

    Returns: float in [0, 1]  (0 if no relevant doc found)
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        if _doc_is_relevant(doc.page_content, ground_truth_keywords):
            return 1.0 / rank
    return 0.0


def keyword_hit_rate(
    retrieved_docs: list,
    ground_truth_keywords: List[str],
) -> Dict[str, bool]:
    """
    Per-keyword presence map across all retrieved documents.

    Returns: dict mapping keyword -> bool (found in any retrieved doc)
    """
    combined = " ".join(doc.page_content.lower() for doc in retrieved_docs)
    return {kw: kw.lower() in combined for kw in ground_truth_keywords}


def compute_retrieval_metrics(
    retrieved_docs: list,
    question_entry: Dict[str, Any],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Compute all retrieval metrics for a single question.

    Args:
        retrieved_docs: List of LangChain Document objects from retriever.
        question_entry: One record from ophthalmic_eval_questions.json.
        k: Number of top docs to consider.

    Returns:
        dict with keys: recall_at_k, precision_at_k, mrr, keyword_hits,
                        keyword_hit_rate_pct, num_retrieved
    """
    keywords = question_entry.get("ground_truth_keywords", [])
    kw_hits = keyword_hit_rate(retrieved_docs, keywords)
    hit_count = sum(kw_hits.values())

    return {
        "question_id": question_entry.get("id", ""),
        "num_retrieved": len(retrieved_docs),
        "recall_at_k": round(recall_at_k(retrieved_docs, keywords, k), 4),
        "precision_at_k": round(precision_at_k(retrieved_docs, keywords, k), 4),
        "mrr": round(mean_reciprocal_rank(retrieved_docs, keywords), 4),
        "keyword_hits": kw_hits,
        "keyword_hit_rate_pct": round(100 * hit_count / len(keywords), 1) if keywords else 0.0,
    }
