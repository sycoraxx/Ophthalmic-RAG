"""
dataset_loader.py — Downloads and normalizes public ophthalmic QA datasets.
────────────────────────────────────────────────────────────────────────────

Sources:
  1. MedMCQA (openlifescienceai/medmcqa) — ophthalmology subset
     MCQ with 4 options + correct answer + expert explanation
     License: MIT

  2. QIAIUNCC/EYE-TEST-2 — 20 expert-validated open-ended ophthalmic QA
     License: CC BY 4.0

Output schema (one dict per question):
  {
    "id": str,
    "source": "medmcqa" | "eye_test2",
    "category": str,          # "mcq" | "open_ended"
    "question": str,
    "options": list[str],     # ["A. ...", ...] or [] for open-ended
    "correct_answer": str,    # full text of correct option or ground-truth answer
    "correct_option_idx": int | None,  # 0-based index (MCQ only)
    "explanation": str,       # expert explanation / ground truth
    "topic": str,
    "ground_truth_keywords": list[str],
  }

Usage:
    python -m evaluation.dataset_loader
    # writes evaluation/dataset/ophthalmic_eval_combined.json
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

OUTPUT_PATH = Path(__file__).parent / "dataset" / "ophthalmic_eval_combined.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OPHTHALMO_TOPICS = {
    "ophthalmology", "eye", "ocular", "retina", "cornea", "glaucoma",
    "cataract", "conjunctiva", "lens", "optic", "macula", "vitreous",
}


# ── Keyword extractor ────────────────────────────────────────────────────────

OPHTHAL_KEYWORD_CANDIDATES = [
    "retina", "cornea", "glaucoma", "cataract", "macula", "conjunctiva",
    "lens", "vitreous", "optic nerve", "optic disc", "choroid", "iris",
    "aqueous", "trabecular", "intraocular pressure", "IOP", "OCT", "fundus",
    "photophobia", "uveitis", "keratitis", "strabismus", "amblyopia",
    "drusen", "AMD", "macular degeneration", "diabetic retinopathy",
    "retinal detachment", "blepharitis", "pterygium", "myopia", "hyperopia",
    "astigmatism", "presbyopia", "lagophthalmos", "epiphora", "nystagmus",
    "papilledema", "optic neuritis", "ischemic optic neuropathy", "chalazion",
    "hordeolum", "dacryocystitis", "endophthalmitis", "hypopyon", "hyphema",
]


def _extract_keywords(text: str) -> List[str]:
    """Pull known ophthalmic terms from text."""
    text_lower = text.lower()
    return [kw for kw in OPHTHAL_KEYWORD_CANDIDATES if kw.lower() in text_lower]


# ── Source 1: MedMCQA ────────────────────────────────────────────────────────

def _is_ophthalmology(record: dict) -> bool:
    subject = (record.get("subject_name") or "").lower()
    topic = (record.get("topic_name") or "").lower()
    combined = subject + " " + topic
    return any(kw in combined for kw in OPHTHALMO_TOPICS)


OPTION_LABELS = ["A", "B", "C", "D"]
OPTION_KEYS = ["opa", "opb", "opc", "opd"]


def _load_medmcqa(n: int = 75) -> List[Dict[str, Any]]:
    """Load up to n ophthalmology questions from MedMCQA."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[dataset_loader] ⚠ `datasets` not installed. Run: pip install datasets")
        return []

    print("[dataset_loader] Streaming MedMCQA (ophthalmology subset)…")
    ds = load_dataset(
        "openlifescienceai/medmcqa",
        split="train",
        streaming=True,
    )

    results = []
    for record in ds:
        if not _is_ophthalmology(record):
            continue
        options = [f"{lbl}. {record.get(k, '')}" for lbl, k in zip(OPTION_LABELS, OPTION_KEYS)]
        cop = record.get("cop")  # 1-based
        correct_idx = int(cop) - 1 if cop is not None else 0
        correct_text = record.get(OPTION_KEYS[correct_idx], "")
        explanation = record.get("exp") or ""
        question = record.get("question", "")

        entry = {
            "id": f"MEDMCQA_{record.get('id', len(results)):}",
            "source": "medmcqa",
            "category": "mcq",
            "question": question,
            "options": options,
            "correct_answer": correct_text,
            "correct_option_idx": correct_idx,
            "explanation": explanation,
            "topic": record.get("topic_name", "Ophthalmology"),
            "ground_truth_keywords": _extract_keywords(question + " " + correct_text + " " + explanation),
        }
        results.append(entry)
        if len(results) >= n:
            break

    print(f"[dataset_loader] Loaded {len(results)} MedMCQA ophthalmology questions.")
    return results


# ── Source 2: EYE-TEST-2 ─────────────────────────────────────────────────────

def _load_eye_test2() -> List[Dict[str, Any]]:
    """Load the 20-question EYE-TEST-2 expert QA dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[dataset_loader] ⚠ `datasets` not installed.")
        return []

    print("[dataset_loader] Loading QIAIUNCC/EYE-TEST-2…")
    ds = load_dataset("QIAIUNCC/EYE-TEST-2", split="train")

    results = []
    for i, record in enumerate(ds):
        # EYE-TEST-2 schema: input (question), instruction, source
        # No ground-truth answer provided — used as evaluation prompts
        q = record.get("input") or record.get("question") or ""
        instruction = record.get("instruction", "")
        entry = {
            "id": f"EYE_TEST2_{i:02d}",
            "source": "eye_test2",
            "category": "open_ended",
            "question": q,
            "options": [],
            "correct_answer": "",   # No ground truth — evaluated via LLM judge only
            "correct_option_idx": None,
            "explanation": instruction,
            "topic": "Ophthalmology — Expert Level",
            "ground_truth_keywords": _extract_keywords(q),
        }
        results.append(entry)

    print(f"[dataset_loader] Loaded {len(results)} EYE-TEST-2 questions.")
    return results


# ── Combined Loader ──────────────────────────────────────────────────────────

def load_eval_dataset(
    n_medmcqa: int = 75,
    use_eye_test2: bool = True,
    output_path: Path = OUTPUT_PATH,
    force_reload: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load and combine all evaluation sources. Caches result to disk.

    Args:
        n_medmcqa: Max number of MedMCQA ophthalmology questions to pull.
        use_eye_test2: Include EYE-TEST-2 expert questions.
        output_path: Where to cache the combined JSON.
        force_reload: Ignore cache and re-download.

    Returns:
        Combined list of question dicts.
    """
    if output_path.exists() and not force_reload:
        print(f"[dataset_loader] Loading cached dataset from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    combined = []
    combined.extend(_load_medmcqa(n=n_medmcqa))
    if use_eye_test2:
        combined.extend(_load_eye_test2())

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"[dataset_loader] Saved {len(combined)} questions → {output_path}")
    return combined


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare ophthalmic eval dataset")
    parser.add_argument("--n-medmcqa", type=int, default=75)
    parser.add_argument("--no-eye-test2", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    data = load_eval_dataset(
        n_medmcqa=args.n_medmcqa,
        use_eye_test2=not args.no_eye_test2,
        output_path=Path(args.output),
        force_reload=args.force_reload,
    )
    print(f"\nTotal questions: {len(data)}")
    cats = {}
    for q in data:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    for cat, cnt in cats.items():
        print(f"  {cat}: {cnt}")
