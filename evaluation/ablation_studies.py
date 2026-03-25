"""
ablation_studies.py — Pipeline Ablation Experiments
─────────────────────────────────────────────────────
Runs the RAG pipeline under different configurations to measure
the contribution of each component.

Ablation configurations:
  1. full_pipeline        — Baseline (query refinement + hybrid + rerank + verify)
  2. no_refinement        — Raw query sent directly to retriever
  3. no_reranking         — Hybrid retrieval without MedCPT cross-encoder
  4. dense_only           — ChromaDB dense retrieval only
  5. bm25_only            — BM25 sparse retrieval only
  6. no_grounding         — Skip grounding verification / self-correction
  7. eyeclip_augmented    — Prepend EyeCLIP-style condition terms to query
                            (simulates image retrieval augmentation for text queries)

Usage:
    python evaluation/ablation_studies.py --max-questions 20
    python evaluation/ablation_studies.py --configs no_refinement no_reranking
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_CONFIGS = [
    "full_pipeline",
    "no_refinement",
    "no_reranking",
    "dense_only",
    "bm25_only",
    "no_grounding",
    "eyeclip_augmented",
]

# Representative EyeCLIP terms to simulate image-driven query augmentation
EYECLIP_SIMULATION_TERMS = [
    "drusen AMD macular degeneration",
    "corneal ulcer keratitis",
    "diabetic retinopathy fundus",
    "glaucomatous optic disc cupping",
    "retinal detachment vitreous",
]


def _avg(vals):
    vals = [v for v in vals if v is not None and v >= 0]
    return round(sum(vals) / len(vals), 4) if vals else None


def _run_config(
    entry: dict,
    engine,
    config: str,
    k: int = 3,
) -> Dict[str, Any]:
    """Run a single question under one ablation configuration."""
    from evaluation.metrics.retrieval_metrics import compute_retrieval_metrics
    from evaluation.metrics.generation_metrics import (
        rouge_l, keyword_coverage, semantic_similarity
    )

    question = entry["question"]
    if entry["category"] == "mcq":
        opts = "\n".join(entry["options"])
        full_q = f"{question}\n\nOptions:\n{opts}"
    else:
        full_q = question

    result = {"id": entry["id"], "config": config}
    t0 = time.time()

    # ── Query preparation ────────────────────────────────────────────────────
    if config == "no_refinement":
        retrieval_query = full_q
    elif config == "eyeclip_augmented":
        # Simulate EyeCLIP augmentation: pick relevant terms by keyword overlap
        import random
        random.seed(hash(question) % 2**32)
        augment = random.choice(EYECLIP_SIMULATION_TERMS)
        try:
            refined = engine.refine_query(full_q)
        except Exception:
            refined = full_q
        retrieval_query = f"{augment} {refined}"
    else:
        try:
            retrieval_query = engine.refine_query(full_q)
        except Exception:
            retrieval_query = full_q

    result["retrieval_query"] = retrieval_query

    # ── Retrieval ────────────────────────────────────────────────────────────
    try:
        if config == "no_reranking":
            # Bypass MedCPT: use raw hybrid results
            child_hits = engine.hybrid_retriever.invoke(retrieval_query)[: k * 2]
            seen, docs = set(), []
            for ch in child_hits:
                pid = ch.metadata.get("parent_id")
                if pid and pid not in seen:
                    pd = engine.parent_store.get(pid)
                    if pd:
                        docs.append(pd)
                        seen.add(pid)
                if len(docs) >= k:
                    break
            retrieved_docs = docs

        elif config == "dense_only":
            # Dense ChromaDB only
            child_hits = engine.retriever.hybrid_retriever.retrievers[1].invoke(retrieval_query)[: k * 2]
            seen, docs = set(), []
            for ch in child_hits:
                pid = ch.metadata.get("parent_id")
                if pid and pid not in seen:
                    pd = engine.parent_store.get(pid)
                    if pd:
                        docs.append(pd)
                        seen.add(pid)
                if len(docs) >= k:
                    break
            retrieved_docs = docs

        elif config == "bm25_only":
            # BM25 sparse only
            child_hits = engine.retriever.hybrid_retriever.retrievers[0].invoke(retrieval_query)[: k * 2]
            seen, docs = set(), []
            for ch in child_hits:
                pid = ch.metadata.get("parent_id")
                if pid and pid not in seen:
                    pd = engine.parent_store.get(pid)
                    if pd:
                        docs.append(pd)
                        seen.add(pid)
                if len(docs) >= k:
                    break
            retrieved_docs = docs

        else:
            # full_pipeline / no_refinement / no_grounding / eyeclip_augmented
            retrieved_docs = engine.retriever.search(retrieval_query, k=k, verbose=False)

    except Exception as e:
        result["retrieval_error"] = str(e)
        retrieved_docs = []

    retrieval_metrics = compute_retrieval_metrics(retrieved_docs, entry, k=k)
    result["retrieval_metrics"] = retrieval_metrics

    if not retrieved_docs:
        result["elapsed_sec"] = round(time.time() - t0, 2)
        return result

    # ── Generation ───────────────────────────────────────────────────────────
    try:
        answer = engine.generator.generate_answer(
            raw_query=full_q,
            context_docs=retrieved_docs,
        )
        result["answer"] = answer
    except Exception as e:
        result["generation_error"] = str(e)
        result["elapsed_sec"] = round(time.time() - t0, 2)
        return result

    # ── Grounding (skip if no_grounding config) ───────────────────────────────
    if config != "no_grounding":
        try:
            ctx_block = engine.generator.build_context_block(retrieved_docs)
            grounding = engine.generator.verify_grounding(answer, ctx_block, verbose=False)
            result["grounding_verdict"] = grounding["verdict"]
        except Exception:
            result["grounding_verdict"] = "ERROR"
    else:
        result["grounding_verdict"] = "SKIPPED"

    # ── Quick generation metrics (no LLM judge in ablation for speed) ────────
    ref = entry.get("explanation") or entry.get("correct_answer") or ""
    result["rouge_l"] = rouge_l(answer, ref)
    result["keyword_coverage"] = keyword_coverage(answer, entry.get("ground_truth_keywords", []))
    result["semantic_similarity"] = semantic_similarity(answer, ref)
    result["elapsed_sec"] = round(time.time() - t0, 2)
    return result


def run_ablations(
    questions: List[dict],
    engine,
    configs: List[str],
    k: int = 3,
) -> Dict[str, Any]:
    """Run all configs over all questions; return comparative summary."""
    config_results: Dict[str, List[dict]] = {c: [] for c in configs}

    for i, entry in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {entry['id']} — {entry['question'][:60]}...")
        for config in configs:
            try:
                r = _run_config(entry, engine, config, k=k)
            except Exception as e:
                r = {"id": entry["id"], "config": config, "error": str(e)}
            config_results[config].append(r)
            rm = r.get("retrieval_metrics", {})
            print(f"  [{config:20s}] recall={rm.get('recall_at_k','?')}  "
                  f"mrr={rm.get('mrr','?')}  "
                  f"rouge={r.get('rouge_l','?')}")

    # Build comparative summary table
    summary = {}
    for config, results in config_results.items():
        recall_vals = [r["retrieval_metrics"]["recall_at_k"] for r in results if "retrieval_metrics" in r]
        mrr_vals = [r["retrieval_metrics"]["mrr"] for r in results if "retrieval_metrics" in r]
        prec_vals = [r["retrieval_metrics"]["precision_at_k"] for r in results if "retrieval_metrics" in r]
        rouge_vals = [r.get("rouge_l") for r in results if r.get("rouge_l") is not None]
        sem_vals = [r.get("semantic_similarity") for r in results if r.get("semantic_similarity", -1) >= 0]
        kw_vals = [r.get("keyword_coverage") for r in results if r.get("keyword_coverage") is not None]
        grounding_pass = sum(1 for r in results if r.get("grounding_verdict") == "PASS")
        grounding_total = sum(1 for r in results if r.get("grounding_verdict") not in (None, "SKIPPED", "ERROR"))

        summary[config] = {
            "avg_recall_at_k": _avg(recall_vals),
            "avg_mrr": _avg(mrr_vals),
            "avg_precision_at_k": _avg(prec_vals),
            "avg_rouge_l": _avg(rouge_vals),
            "avg_semantic_similarity": _avg(sem_vals),
            "avg_keyword_coverage": _avg(kw_vals),
            "grounding_pass_rate": round(grounding_pass / grounding_total, 4) if grounding_total else None,
        }

    return {"summary": summary, "per_config_results": config_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS, choices=ALL_CONFIGS)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from evaluation.dataset_loader import load_eval_dataset
    questions = load_eval_dataset(output_path=Path(args.dataset)) if args.dataset else load_eval_dataset()
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"\n{'='*60}")
    print(f"  Ablation Study — {len(questions)} Qs × {len(args.configs)} configs")
    print(f"  Configs: {args.configs}")
    print(f"{'='*60}")

    from src.engine import QueryEngine
    engine = QueryEngine(enable_session_state=False)

    ablation_data = run_ablations(questions, engine, args.configs, k=args.k)

    # Print comparative table
    print(f"\n{'='*70}")
    print(f"  ABLATION COMPARISON (k={args.k})")
    print(f"{'='*70}")
    print(f"  {'Config':<22} {'Recall@k':>9} {'MRR':>7} {'P@k':>7} {'ROUGE-L':>9} {'SemanticSim':>12} {'Ground%':>8}")
    print(f"  {'-'*22} {'-'*9} {'-'*7} {'-'*7} {'-'*9} {'-'*12} {'-'*8}")
    for config, stats in ablation_data["summary"].items():
        gr = f"{stats['grounding_pass_rate']*100:.1f}%" if stats['grounding_pass_rate'] is not None else "  N/A"
        print(f"  {config:<22} "
              f"{str(stats['avg_recall_at_k']):>9} "
              f"{str(stats['avg_mrr']):>7} "
              f"{str(stats['avg_precision_at_k']):>7} "
              f"{str(stats['avg_rouge_l']):>9} "
              f"{str(stats['avg_semantic_similarity']):>12} "
              f"{gr:>8}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"ablation_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), **ablation_data}, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
