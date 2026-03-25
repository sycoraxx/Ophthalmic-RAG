"""
generation_metrics.py — Generation Quality Metrics
────────────────────────────────────────────────────
Provides semantic similarity, ROUGE-L, keyword coverage,
and LLM-as-judge scoring for generated answers.

Designed to work WITHOUT heavy dependencies when possible:
  - ROUGE-L: pure Python (no nltk required)
  - Semantic similarity: sentence-transformers (lazy loaded)
  - LLM-as-judge: uses MedGemmaGenerator passed in from caller
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional


# ── ROUGE-L (pure Python) ────────────────────────────────────────────────────

def _lcs_length(s1: List[str], s2: List[str]) -> int:
    """Compute LCS length using DP."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 score based on longest common subsequence.
    Returns float in [0, 1].
    """
    def tokenize(text: str) -> List[str]:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ── Keyword Coverage ─────────────────────────────────────────────────────────

def keyword_coverage(answer: str, ground_truth_keywords: List[str]) -> float:
    """
    Fraction of ground-truth keywords found in the generated answer.
    Returns float in [0, 1].
    """
    if not ground_truth_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in ground_truth_keywords if kw.lower() in answer_lower)
    return round(hits / len(ground_truth_keywords), 4)


# ── Semantic Similarity (lazy-loaded sentence-transformers) ──────────────────

_sim_model = None

def _get_sim_model():
    global _sim_model
    if _sim_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            _sim_model = None
    return _sim_model


def semantic_similarity(answer: str, reference: str) -> float:
    """
    Cosine similarity between answer and reference using all-MiniLM-L6-v2.
    Returns float in [0, 1], or -1.0 if model unavailable.
    """
    model = _get_sim_model()
    if model is None:
        return -1.0
    try:
        import numpy as np
        embs = model.encode([answer, reference], normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        return round(max(0.0, sim), 4)
    except Exception:
        return -1.0


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

LLM_JUDGE_PROMPT = """You are a medical QA evaluator for ophthalmology.

Score the GENERATED ANSWER against the REFERENCE ANSWER on these 4 dimensions.
Each dimension: integer 1 (poor) to 5 (excellent).

QUESTION: {question}

REFERENCE ANSWER: {reference}

GENERATED ANSWER: {answer}

Output ONLY this format (no explanation):
ACCURACY: <1-5>
COMPLETENESS: <1-5>
SAFETY: <1-5>
CLARITY: <1-5>"""


def llm_judge_score(
    question: str,
    answer: str,
    reference: str,
    generator,       # MedGemmaGenerator instance
) -> Dict[str, Any]:
    """
    Use MedGemma as an LLM judge to score the generated answer.

    Args:
        question: The original patient query.
        answer: The generated answer to evaluate.
        reference: Ground truth answer from the dataset.
        generator: MedGemmaGenerator instance (from src.generator).

    Returns:
        dict with keys: accuracy, completeness, safety, clarity, overall_avg
        Returns None values if parsing fails.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        question=question, reference=reference, answer=answer
    )
    messages = [
        {"role": "system", "content": "You are a strict medical QA evaluator. Follow the output format exactly."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = generator._generate(
            messages,
            max_new_tokens=80,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
            skip_thought=True,
        )
        scores = {}
        for dim in ["ACCURACY", "COMPLETENESS", "SAFETY", "CLARITY"]:
            m = re.search(rf"{dim}:\s*([1-5])", response, re.IGNORECASE)
            scores[dim.lower()] = int(m.group(1)) if m else None

        valid = [v for v in scores.values() if v is not None]
        scores["overall_avg"] = round(sum(valid) / len(valid), 2) if valid else None
        scores["raw_response"] = response
        return scores
    except Exception as e:
        return {
            "accuracy": None, "completeness": None,
            "safety": None, "clarity": None,
            "overall_avg": None, "error": str(e),
        }


# ── Combined Entry Point ─────────────────────────────────────────────────────

def compute_generation_metrics(
    answer: str,
    question_entry: Dict[str, Any],
    generator=None,
    run_llm_judge: bool = True,
) -> Dict[str, Any]:
    """
    Compute all generation metrics for a single answer.

    Args:
        answer: Generated answer string.
        question_entry: One record from ophthalmic_eval_questions.json.
        generator: MedGemmaGenerator (required for LLM-judge, optional otherwise).
        run_llm_judge: Whether to run the expensive LLM judge step.

    Returns:
        dict with all generation metric scores.
    """
    reference = question_entry.get("ground_truth_answer", "")
    keywords = question_entry.get("ground_truth_keywords", [])
    question = question_entry.get("question", "")

    result = {
        "question_id": question_entry.get("id", ""),
        "rouge_l": rouge_l(answer, reference),
        "keyword_coverage": keyword_coverage(answer, keywords),
        "semantic_similarity": semantic_similarity(answer, reference),
    }

    if run_llm_judge and generator is not None and reference:
        result["llm_judge"] = llm_judge_score(question, answer, reference, generator)
    else:
        result["llm_judge"] = None

    return result
