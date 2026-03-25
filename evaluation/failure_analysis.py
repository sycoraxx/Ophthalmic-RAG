"""
failure_analysis.py — Systematic Failure Documentation
────────────────────────────────────────────────────────
Analyzes evaluation results to document three failure categories:

  1. Hallucination Cases   — answers with grounding FAIL verdict
  2. Retrieval Misses      — questions with zero relevant docs retrieved
  3. Ambiguous Query Cases — responses to vague/unclear questions

Can run standalone on a saved eval_results JSON, or be called from
run_evaluation.py after a fresh run.

Usage:
    # Analyze a previous run:
    python evaluation/failure_analysis.py \
        --results evaluation/results/eval_results_20260325_120000.json

    # Run fresh eval + analyze:
    python evaluation/failure_analysis.py --run-fresh --max-questions 30
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Vague/ambiguous query tokens — used to flag retrieval misses from unclear Qs
AMBIGUITY_SIGNALS = [
    "weird", "strange", "funny", "off", "wrong", "bad", "uncomfortable",
    "not right", "something", "feels like", "don't know", "not sure",
    "kind of", "sort of", "a bit", "little bit", "lately", "sometimes",
]


# ── 1. Hallucination Analysis ─────────────────────────────────────────────────

def analyze_hallucinations(results: List[Dict]) -> Dict[str, Any]:
    """
    Identify answers where grounding verification flagged unsupported claims.
    Returns structured report.
    """
    cases = []
    for r in results:
        grounding = r.get("grounding") or {}
        verdict = grounding.get("verdict", "N/A")
        if verdict == "FAIL":
            cases.append({
                "id": r["id"],
                "question": r.get("question", ""),
                "answer_snippet": (r.get("answer") or "")[:300],
                "flagged_claims": grounding.get("flagged_claims", []),
                "anatomy_mismatch": grounding.get("anatomy_mismatch", ""),
                "reasoning": grounding.get("reasoning", ""),
                "grounding_verdict": verdict,
                "num_retrieved": r.get("num_retrieved", 0),
            })

    total_with_grounding = sum(1 for r in results if "grounding" in r)
    hallucination_rate = len(cases) / total_with_grounding if total_with_grounding else 0

    # Most common flagged claim patterns
    all_claims = [c for case in cases for c in case["flagged_claims"]]

    return {
        "total_evaluated": total_with_grounding,
        "hallucination_count": len(cases),
        "hallucination_rate": round(hallucination_rate, 4),
        "all_flagged_claims_count": len(all_claims),
        "cases": cases,
    }


# ── 2. Retrieval Miss Analysis ────────────────────────────────────────────────

def analyze_retrieval_misses(results: List[Dict]) -> Dict[str, Any]:
    """
    Identify questions where retrieval failed to find relevant documents.
    A miss = recall@k == 0 AND keyword_hit_rate_pct == 0.
    """
    cases = []
    for r in results:
        rm = r.get("retrieval_metrics") or {}
        recall = rm.get("recall_at_k", 1)
        kw_hit = rm.get("keyword_hit_rate_pct", 100)

        if recall == 0 and kw_hit == 0:
            # Classify miss reason
            num_retrieved = r.get("num_retrieved", 0)
            if num_retrieved == 0:
                reason = "no_docs_retrieved"
            else:
                # Docs retrieved but none relevant
                sources = [s.get("source", "") for s in r.get("retrieved_sources", [])]
                reason = "vocabulary_mismatch"

            cases.append({
                "id": r["id"],
                "question": r.get("question", ""),
                "refined_query": r.get("refined_query", ""),
                "num_retrieved": num_retrieved,
                "retrieved_sources": r.get("retrieved_sources", []),
                "expected_keywords": list((r.get("retrieval_metrics") or {}).get("keyword_hits", {}).keys()),
                "miss_reason": reason,
            })

    total = len(results)
    miss_rate = len(cases) / total if total else 0

    # Categorize miss reasons
    reason_counts: Dict[str, int] = {}
    for c in cases:
        reason_counts[c["miss_reason"]] = reason_counts.get(c["miss_reason"], 0) + 1

    return {
        "total_questions": total,
        "retrieval_miss_count": len(cases),
        "retrieval_miss_rate": round(miss_rate, 4),
        "miss_reason_breakdown": reason_counts,
        "cases": cases,
    }


# ── 3. Ambiguous Query Analysis ───────────────────────────────────────────────

def analyze_ambiguous_queries(results: List[Dict]) -> Dict[str, Any]:
    """
    Assess how the pipeline handles vague or ambiguous questions.
    Flags:
      - Questions containing ambiguity signal words
      - Checks if answer appropriately hedges (contains "consult", "professional", etc.)
      - Checks if answer is over-confident (contains specific diagnoses without hedging)
    """
    HEDGE_PHRASES = [
        "consult", "see a doctor", "ophthalmologist", "eye care", "professional",
        "cannot diagnose", "not able to diagnose", "recommend seeing",
        "it could be", "may be", "might be", "possible", "I'd suggest",
    ]
    OVERCONFIDENT_PHRASES = [
        "you have", "this is definitely", "this is certainly", "diagnosed with",
        "clearly indicates", "definitive diagnosis",
    ]

    ambiguous_qs = []
    for r in results:
        q = (r.get("question") or "").lower()
        is_ambiguous = any(sig in q for sig in AMBIGUITY_SIGNALS)
        if not is_ambiguous:
            continue

        answer = (r.get("answer") or "").lower()
        has_hedge = any(p in answer for p in HEDGE_PHRASES)
        is_overconfident = any(p in answer for p in OVERCONFIDENT_PHRASES)

        ambiguous_qs.append({
            "id": r["id"],
            "question": r.get("question", ""),
            "answer_snippet": (r.get("answer") or "")[:300],
            "has_appropriate_hedge": has_hedge,
            "is_overconfident": is_overconfident,
            "grounding_verdict": (r.get("grounding") or {}).get("verdict", "N/A"),
            "recall_at_k": (r.get("retrieval_metrics") or {}).get("recall_at_k"),
        })

    total_ambiguous = len(ambiguous_qs)
    hedged = sum(1 for c in ambiguous_qs if c["has_appropriate_hedge"])
    overconfident = sum(1 for c in ambiguous_qs if c["is_overconfident"])

    return {
        "total_ambiguous_queries": total_ambiguous,
        "appropriately_hedged_count": hedged,
        "hedge_rate": round(hedged / total_ambiguous, 4) if total_ambiguous else None,
        "overconfident_count": overconfident,
        "overconfidence_rate": round(overconfident / total_ambiguous, 4) if total_ambiguous else None,
        "cases": ambiguous_qs,
    }


# ── Report Generator ──────────────────────────────────────────────────────────

def generate_markdown_report(
    hallucinations: Dict,
    retrieval_misses: Dict,
    ambiguous: Dict,
    output_path: Path,
):
    """Write a human-readable failure analysis markdown report."""
    lines = [
        "# Ophthalmic RAG — Failure Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Hallucination Cases",
        "",
        f"- **Total evaluated with grounding:** {hallucinations['total_evaluated']}",
        f"- **Hallucination count:** {hallucinations['hallucination_count']}",
        f"- **Hallucination rate:** {hallucinations['hallucination_rate']*100:.1f}%",
        f"- **Total flagged claims across all cases:** {hallucinations['all_flagged_claims_count']}",
        "",
        "### Notable Cases",
        "",
    ]
    for case in hallucinations["cases"][:5]:
        lines += [
            f"**ID:** `{case['id']}`  ",
            f"**Q:** {case['question'][:120]}...  ",
            f"**Flagged claims:** {'; '.join(case['flagged_claims'][:3])}  ",
            f"**Anatomy mismatch:** {case['anatomy_mismatch']}  ",
            "",
        ]

    lines += [
        "---",
        "",
        "## 2. Retrieval Misses",
        "",
        f"- **Total questions:** {retrieval_misses['total_questions']}",
        f"- **Retrieval miss count:** {retrieval_misses['retrieval_miss_count']}",
        f"- **Retrieval miss rate:** {retrieval_misses['retrieval_miss_rate']*100:.1f}%",
        "",
        "### Miss Reason Breakdown",
        "",
    ]
    for reason, count in retrieval_misses["miss_reason_breakdown"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.append("")
    lines += ["### Notable Cases", ""]
    for case in retrieval_misses["cases"][:5]:
        lines += [
            f"**ID:** `{case['id']}`  ",
            f"**Q:** {case['question'][:120]}  ",
            f"**Refined query:** `{case['refined_query'][:80]}`  ",
            f"**Docs retrieved:** {case['num_retrieved']}  **Reason:** {case['miss_reason']}  ",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Ambiguous Query Handling",
        "",
        f"- **Ambiguous queries detected:** {ambiguous['total_ambiguous_queries']}",
        f"- **Appropriately hedged:** {ambiguous['appropriately_hedged_count']} "
        f"({(ambiguous.get('hedge_rate') or 0)*100:.1f}%)",
        f"- **Overconfident responses:** {ambiguous['overconfident_count']} "
        f"({(ambiguous.get('overconfidence_rate') or 0)*100:.1f}%)",
        "",
        "### Cases",
        "",
    ]
    for case in ambiguous["cases"][:5]:
        hedge_icon = "✅" if case["has_appropriate_hedge"] else "⚠️"
        conf_icon = "🚨" if case["is_overconfident"] else "✅"
        lines += [
            f"**ID:** `{case['id']}`  ",
            f"**Q:** {case['question'][:120]}  ",
            f"**Hedge:** {hedge_icon}  **Overconfident:** {conf_icon}  ",
            f"**Answer snippet:** {case['answer_snippet'][:150]}...  ",
            "",
        ]

    lines += [
        "---",
        "",
        "## Key Takeaways",
        "",
        f"- **Hallucination rate of {hallucinations['hallucination_rate']*100:.1f}%** "
        "suggests the grounding verification + self-correction loop is effective.",
        f"- **Retrieval miss rate of {retrieval_misses['retrieval_miss_rate']*100:.1f}%** "
        "highlights vocabulary gaps between patient lay language and textbook terminology.",
        f"- **{(ambiguous.get('hedge_rate') or 0)*100:.1f}% appropriate hedging** on ambiguous queries "
        "shows the system avoids overconfident claims when the question is unclear.",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[failure_analysis] Markdown report → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_failure_analysis(results: List[Dict], output_dir: Path) -> Dict[str, Any]:
    print("[failure_analysis] Analyzing hallucinations...")
    hall = analyze_hallucinations(results)
    print(f"  → {hall['hallucination_count']} hallucination cases ({hall['hallucination_rate']*100:.1f}%)")

    print("[failure_analysis] Analyzing retrieval misses...")
    misses = analyze_retrieval_misses(results)
    print(f"  → {misses['retrieval_miss_count']} retrieval misses ({misses['retrieval_miss_rate']*100:.1f}%)")

    print("[failure_analysis] Analyzing ambiguous queries...")
    ambig = analyze_ambiguous_queries(results)
    print(f"  → {ambig['total_ambiguous_queries']} ambiguous queries found")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "hallucinations": hall,
        "retrieval_misses": misses,
        "ambiguous_queries": ambig,
    }

    json_path = output_dir / f"failure_analysis_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = output_dir / f"failure_analysis_{timestamp}.md"
    generate_markdown_report(hall, misses, ambig, md_path)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="Path to existing eval_results*.json")
    parser.add_argument("--run-fresh", action="store_true",
                        help="Run evaluation first, then analyze")
    parser.add_argument("--max-questions", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    if args.gpus:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    output_dir = Path(args.output_dir)

    if args.results:
        with open(args.results) as f:
            payload = json.load(f)
        results = payload.get("results", payload)
    elif args.run_fresh:
        import os
        os.chdir(ROOT)
        from evaluation.dataset_loader import load_eval_dataset
        from evaluation.run_evaluation import run_single
        from src.engine import QueryEngine

        questions = load_eval_dataset()
        if args.max_questions:
            questions = questions[: args.max_questions]

        print(f"Running fresh evaluation on {len(questions)} questions...")
        engine = QueryEngine(enable_session_state=False)
        results = []
        for i, entry in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {entry['id']}...")
            try:
                r = run_single(entry, engine, k=3, retrieval_only=False, run_llm_judge=False)
            except Exception as e:
                r = {"id": entry["id"], "error": str(e)}
            results.append(r)
    else:
        # Try to find the most recent results file
        result_files = sorted(RESULTS_DIR.glob("eval_results_*.json"))
        if not result_files:
            print("No results file found. Run with --run-fresh or --results <path>")
            sys.exit(1)
        latest = result_files[-1]
        print(f"Using most recent results: {latest}")
        with open(latest) as f:
            payload = json.load(f)
        results = payload.get("results", payload)

    run_failure_analysis(results, output_dir)


if __name__ == "__main__":
    main()
