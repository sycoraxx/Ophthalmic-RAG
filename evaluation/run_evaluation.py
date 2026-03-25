"""
run_evaluation.py — End-to-end RAG Pipeline Evaluation
────────────────────────────────────────────────────────
Runs each question in the ophthalmic eval dataset through the full
RAG pipeline and computes retrieval + generation metrics.

Usage:
    # Full evaluation (requires GPU + loaded models):
    python evaluation/run_evaluation.py

    # Retrieval only (faster, no LLM):
    python evaluation/run_evaluation.py --retrieval-only

    # Run on N questions only (quick smoke test):
    python evaluation/run_evaluation.py --max-questions 10

    # Skip LLM-as-judge (saves time):
    python evaluation/run_evaluation.py --no-llm-judge
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── MCQ accuracy helper ───────────────────────────────────────────────────────

def _mcq_answer_correct(answer: str, entry: dict) -> bool:
    """Check if generated answer selects the correct MCQ option."""
    if entry["category"] != "mcq" or entry["correct_option_idx"] is None:
        return False
    idx = entry["correct_option_idx"]
    label = ["A", "B", "C", "D"][idx]
    correct_text = entry["correct_answer"].lower()
    answer_lower = answer.lower()
    # Accept: "A", "option A", contains correct text
    return (
        re.search(rf"\b{label}\b", answer, re.IGNORECASE) is not None
        or correct_text in answer_lower
    )


# ── Per-question pipeline run ─────────────────────────────────────────────────

def run_single(
    entry: dict,
    engine,
    k: int = 3,
    retrieval_only: bool = False,
    run_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Run the pipeline on one question and return all metrics."""
    import re as _re
    global re
    re = _re

    from evaluation.metrics.retrieval_metrics import compute_retrieval_metrics
    from evaluation.metrics.generation_metrics import compute_generation_metrics

    question = entry["question"]
    if entry["category"] == "mcq":
        opts_block = "\n".join(entry["options"])
        full_question = f"{question}\n\nOptions:\n{opts_block}"
    else:
        full_question = question

    result = {
        "id": entry["id"],
        "source": entry["source"],
        "category": entry["category"],
        "question": question,
        "topic": entry["topic"],
    }

    t0 = time.time()

    # ── Step 1: Query refinement ─────────────────────────────────────────────
    try:
        refined_query = engine.refine_query(full_question)
    except Exception as e:
        refined_query = full_question
        result["refine_error"] = str(e)

    result["refined_query"] = refined_query

    # ── Step 2: Retrieval ────────────────────────────────────────────────────
    try:
        retrieved_docs = engine.retriever.search(refined_query, k=k, verbose=False)
    except Exception as e:
        retrieved_docs = []
        result["retrieval_error"] = str(e)

    result["num_retrieved"] = len(retrieved_docs)
    result["retrieved_sources"] = [
        {
            "source": d.metadata.get("source", "?"),
            "section_path": d.metadata.get("section_path", "?"),
        }
        for d in retrieved_docs
    ]

    # ── Retrieval metrics ────────────────────────────────────────────────────
    retrieval_metrics = compute_retrieval_metrics(retrieved_docs, entry, k=k)
    result["retrieval_metrics"] = retrieval_metrics

    if retrieval_only or not retrieved_docs:
        result["elapsed_sec"] = round(time.time() - t0, 2)
        return result

    # ── Step 3: Generation ───────────────────────────────────────────────────
    try:
        answer = engine.generator.generate_answer(
            raw_query=full_question,
            context_docs=retrieved_docs,
        )
        result["answer"] = answer
    except Exception as e:
        result["answer"] = ""
        result["generation_error"] = str(e)
        result["elapsed_sec"] = round(time.time() - t0, 2)
        return result

    # ── Step 4: Grounding verification ───────────────────────────────────────
    try:
        context_block = engine.generator.build_context_block(retrieved_docs)
        grounding = engine.generator.verify_grounding(answer, context_block, verbose=False)
        result["grounding"] = grounding
    except Exception as e:
        result["grounding"] = {"verdict": "ERROR", "error": str(e)}

    # ── Generation metrics ───────────────────────────────────────────────────
    gen_metrics = compute_generation_metrics(
        answer=answer,
        question_entry=entry,
        generator=engine.generator if run_llm_judge else None,
        run_llm_judge=run_llm_judge,
    )
    result["generation_metrics"] = gen_metrics

    # MCQ accuracy
    if entry["category"] == "mcq":
        result["mcq_correct"] = bool(re.search(
            rf"\b{['A','B','C','D'][entry['correct_option_idx']]}\b",
            answer, re.IGNORECASE
        ) or entry["correct_answer"].lower() in answer.lower())

    result["elapsed_sec"] = round(time.time() - t0, 2)
    return result


# ── Aggregate stats ───────────────────────────────────────────────────────────

def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics across all results."""
    def avg(vals):
        vals = [v for v in vals if v is not None and v >= 0]
        return round(sum(vals) / len(vals), 4) if vals else None

    recall_vals = [r["retrieval_metrics"]["recall_at_k"] for r in results if "retrieval_metrics" in r]
    mrr_vals = [r["retrieval_metrics"]["mrr"] for r in results if "retrieval_metrics" in r]
    precision_vals = [r["retrieval_metrics"]["precision_at_k"] for r in results if "retrieval_metrics" in r]
    kw_hit_vals = [r["retrieval_metrics"]["keyword_hit_rate_pct"] for r in results if "retrieval_metrics" in r]

    rouge_vals = [r["generation_metrics"]["rouge_l"] for r in results if "generation_metrics" in r]
    sem_vals = [r["generation_metrics"]["semantic_similarity"] for r in results if "generation_metrics" in r]
    kw_cov_vals = [r["generation_metrics"]["keyword_coverage"] for r in results if "generation_metrics" in r]

    judge_vals = []
    for r in results:
        jm = (r.get("generation_metrics") or {}).get("llm_judge") or {}
        if jm.get("overall_avg") is not None:
            judge_vals.append(jm["overall_avg"])

    mcq_results = [r for r in results if r.get("category") == "mcq" and "mcq_correct" in r]
    mcq_accuracy = sum(r["mcq_correct"] for r in mcq_results) / len(mcq_results) if mcq_results else None

    grounding_pass = sum(
        1 for r in results
        if (r.get("grounding") or {}).get("verdict") == "PASS"
    )
    grounding_total = sum(1 for r in results if "grounding" in r)

    return {
        "total_questions": len(results),
        "retrieval": {
            "avg_recall_at_k": avg(recall_vals),
            "avg_mrr": avg(mrr_vals),
            "avg_precision_at_k": avg(precision_vals),
            "avg_keyword_hit_rate_pct": avg(kw_hit_vals),
        },
        "generation": {
            "avg_rouge_l": avg(rouge_vals),
            "avg_semantic_similarity": avg(sem_vals),
            "avg_keyword_coverage": avg(kw_cov_vals),
            "avg_llm_judge_score": avg(judge_vals),
            "mcq_accuracy": round(mcq_accuracy, 4) if mcq_accuracy is not None else None,
            "grounding_pass_rate": round(grounding_pass / grounding_total, 4) if grounding_total else None,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to ophthalmic_eval_combined.json (auto-downloads if absent)")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Load dataset
    from evaluation.dataset_loader import load_eval_dataset
    dataset_path = Path(args.dataset) if args.dataset else None
    if dataset_path:
        with open(dataset_path) as f:
            questions = json.load(f)
    else:
        questions = load_eval_dataset()

    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"\n{'='*60}")
    print(f"  Ophthalmic RAG Evaluation — {len(questions)} questions")
    print(f"  k={args.k}  retrieval_only={args.retrieval_only}")
    print(f"{'='*60}\n")

    # Load engine
    print("Loading QueryEngine...")
    from src.engine import QueryEngine
    engine = QueryEngine(enable_session_state=False)
    print("Engine ready.\n")

    results = []
    for i, entry in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {entry['id']} — {entry['question'][:70]}...")
        try:
            r = run_single(
                entry, engine,
                k=args.k,
                retrieval_only=args.retrieval_only,
                run_llm_judge=not args.no_llm_judge,
            )
        except Exception as e:
            r = {"id": entry["id"], "error": str(e)}
        results.append(r)
        print(f"         recall@{args.k}={r.get('retrieval_metrics', {}).get('recall_at_k', 'N/A')}"
              f"  mrr={r.get('retrieval_metrics', {}).get('mrr', 'N/A')}")

    summary = aggregate_results(results)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"eval_results_{timestamp}.json"
    payload = {"summary": summary, "config": vars(args), "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    ret = summary["retrieval"]
    gen = summary["generation"]
    print(f"  RETRIEVAL  Recall@{args.k}: {ret['avg_recall_at_k']}  "
          f"MRR: {ret['avg_mrr']}  "
          f"P@{args.k}: {ret['avg_precision_at_k']}  "
          f"Keyword Hit: {ret['avg_keyword_hit_rate_pct']}%")
    if not args.retrieval_only:
        print(f"  GENERATION ROUGE-L: {gen['avg_rouge_l']}  "
              f"SemanticSim: {gen['avg_semantic_similarity']}  "
              f"KW-Coverage: {gen['avg_keyword_coverage']}")
        if gen["avg_llm_judge_score"] is not None:
            print(f"             LLM-Judge: {gen['avg_llm_judge_score']}/5.0")
        if gen["mcq_accuracy"] is not None:
            print(f"             MCQ Accuracy: {gen['mcq_accuracy']*100:.1f}%")
        if gen["grounding_pass_rate"] is not None:
            print(f"             Grounding Pass Rate: {gen['grounding_pass_rate']*100:.1f}%")
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    import re
    main()
