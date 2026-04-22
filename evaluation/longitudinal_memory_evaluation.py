"""
longitudinal_memory_evaluation.py — Compare patient-memory backends on follow-up turns.

This script replays a small set of multi-turn ophthalmology cases with a fresh
session per visit and the same patient_id across visits. It compares the final
follow-up answer quality between the SQLite fallback store and the MemPalace
backend using the existing generation metrics.

Usage:
    python evaluation/longitudinal_memory_evaluation.py
    python evaluation/longitudinal_memory_evaluation.py --max-cases 2
    python evaluation/longitudinal_memory_evaluation.py --backends sqlite mempalace
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_CASES: List[Dict[str, Any]] = [
    {
        "id": "corneal_ulcer_followup",
        "patient_id": "LG-1001",
        "turns": [
            {
                "query": "I was told I have a corneal ulcer in my right eye and started moxifloxacin drops yesterday.",
                "reference": "You should keep using the prescribed drops exactly as directed and get urgent review if pain, redness, or vision gets worse.",
                "keywords": ["corneal ulcer", "right eye", "moxifloxacin", "drops"],
            },
            {
                "query": "The redness is a bit better, but the eye still hurts. Should I stop the drops now?",
                "reference": "Do not stop on your own. Continue the treatment for the right corneal ulcer/keratitis as prescribed and seek urgent review if symptoms worsen.",
                "keywords": ["right eye", "corneal ulcer", "keratitis", "drops"],
            },
        ],
    },
    {
        "id": "diabetic_retinopathy_followup",
        "patient_id": "LG-1002",
        "turns": [
            {
                "query": "My doctor said I have diabetic retinopathy and asked me to control my blood sugar.",
                "reference": "Blood sugar control is important because diabetic retinopathy can worsen when diabetes is poorly controlled.",
                "keywords": ["diabetic retinopathy", "blood sugar", "diabetes"],
            },
            {
                "query": "Now I am seeing more floaters in that eye. Does the same diabetes problem explain it?",
                "reference": "Yes, the same diabetic eye disease may be relevant, and new floaters need prompt ophthalmology review because they can signal bleeding or retinal involvement.",
                "keywords": ["diabetic retinopathy", "floaters", "retinal"],
            },
        ],
    },
    {
        "id": "cataract_surgery_followup",
        "patient_id": "LG-1003",
        "turns": [
            {
                "query": "I was scheduled for cataract surgery in my left eye next week.",
                "reference": "Cataract surgery is a common procedure and your doctor will explain the timing, drops, and precautions before the operation.",
                "keywords": ["cataract", "left eye", "surgery"],
            },
            {
                "query": "I forgot the exact timing. Should I still continue the pre-op drops before the surgery?",
                "reference": "Yes, keep following the pre-operative plan for the same left-eye cataract surgery and confirm the schedule with your surgeon.",
                "keywords": ["left eye", "cataract surgery", "drops"],
            },
        ],
    },
]


def _keyword_coverage(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
    return round(hits / len(keywords), 4)


def _build_engine_config(base_dir: Path, backend: str) -> Path:
    config = {
        "patient_memory": {
            "enabled": True,
            "backend": backend,
            "palace_path": str(base_dir / f"palace_{backend}"),
            "sqlite_path": str(base_dir / f"patient_memory_{backend}.sqlite"),
            "enable_kg": True,
        }
    }
    config_path = base_dir / f"config_{backend}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def _normalize_answer(result: tuple[Any, ...]) -> tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
    answer = result[0] if len(result) > 0 else ""
    visual = result[1] if len(result) > 1 else None
    session_id = result[2] if len(result) > 2 and isinstance(result[2], str) else None
    trace = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
    return str(answer), visual if isinstance(visual, str) else None, session_id, trace


def _run_case(engine, case: Dict[str, Any], backend: str) -> Dict[str, Any]:
    from evaluation.metrics.generation_metrics import compute_generation_metrics

    case_result: Dict[str, Any] = {
        "case_id": case["id"],
        "patient_id": case["patient_id"],
        "backend": backend,
        "turns": [],
    }

    for turn_index, turn in enumerate(case["turns"]):
        session_id = str(uuid.uuid4())
        raw = engine.ask(
            turn["query"],
            session_id=session_id,
            patient_id=case["patient_id"],
            verbose=False,
            fast_mode=True,
            use_session_state=True,
            return_trace=True,
        )
        answer, visual_findings, returned_session_id, trace = _normalize_answer(raw)
        turn_result: Dict[str, Any] = {
            "turn_index": turn_index,
            "query": turn["query"],
            "answer": answer,
            "visual_findings": visual_findings,
            "session_id": returned_session_id,
            "trace": trace,
        }

        if turn_index > 0:
            eval_entry = {
                "id": f"{case['id']}_turn_{turn_index}",
                "question": turn["query"],
                "correct_answer": turn.get("reference", ""),
                "ground_truth_keywords": turn.get("keywords", []),
            }
            metrics = compute_generation_metrics(
                answer=answer,
                question_entry=eval_entry,
                generator=None,
                run_llm_judge=False,
            )
            turn_result["generation_metrics"] = metrics
            turn_result["carryover_keyword_coverage"] = _keyword_coverage(answer, turn.get("keywords", []))

        case_result["turns"].append(turn_result)

    return case_result


def _summarize_backend_results(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored_turns = [
        turn
        for case in case_results
        for turn in case["turns"]
        if turn.get("turn_index", 0) > 0 and turn.get("generation_metrics")
    ]

    def _avg(values: List[float]) -> Optional[float]:
        values = [value for value in values if value is not None and value >= 0]
        return round(sum(values) / len(values), 4) if values else None

    rouge_vals = [turn["generation_metrics"]["rouge_l"] for turn in scored_turns]
    sem_vals = [turn["generation_metrics"]["semantic_similarity"] for turn in scored_turns if turn["generation_metrics"]["semantic_similarity"] >= 0]
    kw_vals = [turn["generation_metrics"]["keyword_coverage"] for turn in scored_turns]
    carry_vals = [turn.get("carryover_keyword_coverage") for turn in scored_turns]

    per_case_final_scores: Dict[str, float] = {}
    for case in case_results:
        final_turn = case["turns"][-1]
        metrics = final_turn.get("generation_metrics") or {}
        per_case_final_scores[case["case_id"]] = float(metrics.get("keyword_coverage") or 0.0)

    return {
        "avg_rouge_l": _avg(rouge_vals),
        "avg_semantic_similarity": _avg(sem_vals),
        "avg_keyword_coverage": _avg(kw_vals),
        "avg_carryover_keyword_coverage": _avg(carry_vals),
        "per_case_final_keyword_coverage": per_case_final_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare longitudinal memory backends on follow-up ophthalmology cases.")
    parser.add_argument("--backends", nargs="+", default=["sqlite", "mempalace"], choices=["sqlite", "mempalace"])
    parser.add_argument("--cases", type=str, default=None, help="Optional JSON file with longitudinal cases.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)
    else:
        cases = DEFAULT_CASES

    if args.max_cases:
        cases = cases[: args.max_cases]

    print(f"\n{'=' * 70}")
    print(f"  Longitudinal Memory Evaluation — {len(cases)} cases")
    print(f"  Backends: {args.backends}")
    print(f"{'=' * 70}\n")

    from src.engine import QueryEngine

    backend_outputs: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="longitudinal_memory_eval_", dir=str(RESULTS_DIR)) as temp_root:
        temp_root_path = Path(temp_root)
        for backend in args.backends:
            backend_dir = temp_root_path / backend
            backend_dir.mkdir(parents=True, exist_ok=True)
            config_path = _build_engine_config(backend_dir, backend)
            print(f"Loading engine for backend={backend}...")
            engine = QueryEngine(enable_session_state=True, config_path=str(config_path), patient_memory_backend=backend)
            case_results = []
            for index, case in enumerate(cases, 1):
                print(f"  [{backend}] case {index}/{len(cases)} — {case['id']}")
                case_results.append(_run_case(engine, case, backend))
            backend_outputs[backend] = {
                "summary": _summarize_backend_results(case_results),
                "cases": case_results,
            }

    comparison: Dict[str, Any] = {"backends": backend_outputs}
    if len(args.backends) == 2:
        first, second = args.backends
        a = backend_outputs[first]["summary"]
        b = backend_outputs[second]["summary"]
        comparison["delta"] = {
            "avg_rouge_l": None if a["avg_rouge_l"] is None or b["avg_rouge_l"] is None else round(b["avg_rouge_l"] - a["avg_rouge_l"], 4),
            "avg_semantic_similarity": None if a["avg_semantic_similarity"] is None or b["avg_semantic_similarity"] is None else round(b["avg_semantic_similarity"] - a["avg_semantic_similarity"], 4),
            "avg_keyword_coverage": None if a["avg_keyword_coverage"] is None or b["avg_keyword_coverage"] is None else round(b["avg_keyword_coverage"] - a["avg_keyword_coverage"], 4),
            "avg_carryover_keyword_coverage": None if a["avg_carryover_keyword_coverage"] is None or b["avg_carryover_keyword_coverage"] is None else round(b["avg_carryover_keyword_coverage"] - a["avg_carryover_keyword_coverage"], 4),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"longitudinal_memory_eval_{timestamp}.json"
    payload = {"config": vars(args), **comparison}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  LONGITUDINAL MEMORY SUMMARY")
    print(f"{'=' * 70}")
    for backend, data in backend_outputs.items():
        summary = data["summary"]
        print(
            f"  {backend:<10} "
            f"ROUGE-L={summary['avg_rouge_l']}  "
            f"SemSim={summary['avg_semantic_similarity']}  "
            f"KW={summary['avg_keyword_coverage']}  "
            f"Carry={summary['avg_carryover_keyword_coverage']}"
        )
    if "delta" in comparison:
        delta = comparison["delta"]
        print(
            f"  DELTA      "
            f"ROUGE-L={delta['avg_rouge_l']}  "
            f"SemSim={delta['avg_semantic_similarity']}  "
            f"KW={delta['avg_keyword_coverage']}  "
            f"Carry={delta['avg_carryover_keyword_coverage']}"
        )
    print(f"\n  Results saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())