"""
anatomy_graph_regression.py
──────────────────────────
Deterministic regression checks for anatomy grounding and anti-hallucination rules.

Usage:
    python evaluation/anatomy_graph_regression.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.anatomy import get_eye_anatomy_graph
from src.generator import MedGemmaGenerator
from src.triage import check_red_flags

RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    detail: str


def _triage_level(msg: str | None) -> str:
    if not msg:
        return "none"
    low = msg.lower()
    if "emergency alert" in low:
        return "emergency"
    if "urgent eye alert" in low or "same-day" in low:
        return "urgent"
    return "other"


def run_lay_mapping_checks(graph) -> list[CaseResult]:
    cases = [
        ("lay_01_black_part", "black part of the eye", {"pupil"}, {"sclera"}),
        ("lay_02_colored_part", "colored part of the eye", {"iris"}, {"sclera", "pupil"}),
        ("lay_03_white_part", "white part of the eye", {"sclera"}, {"pupil"}),
        ("lay_04_clear_front", "clear front of the eye", {"cornea"}, {"retina"}),
    ]
    out: list[CaseResult] = []

    for case_id, text, must_have, must_not_have in cases:
        detected = graph.detect_structures(text)
        missing = sorted([x for x in must_have if x not in detected])
        leaked = sorted([x for x in must_not_have if x in detected])
        passed = not missing and not leaked
        detail = f"detected={sorted(detected)}"
        if missing:
            detail += f" | missing={missing}"
        if leaked:
            detail += f" | leaked={leaked}"
        out.append(CaseResult(case_id=case_id, passed=passed, detail=detail))

    return out


def run_contradiction_checks(graph) -> list[CaseResult]:
    bad_cases = [
        ("contra_01", "The sclera is the black part of the eye."),
        ("contra_02", "The pupil is the white part of the eye."),
        ("contra_03", "The cornea is the colored part of the eye."),
        ("contra_04", "The retina is visible in the mirror from the front."),
    ]
    good_cases = [
        ("contra_ok_01", "The pupil appears black because it is an opening in the iris."),
        ("contra_ok_02", "The sclera is the white outer coat and the cornea is clear."),
    ]

    out: list[CaseResult] = []

    for case_id, text in bad_cases:
        contradictions = graph.find_anatomy_contradictions(text)
        passed = len(contradictions) > 0
        out.append(CaseResult(case_id=case_id, passed=passed, detail=str(contradictions)))

    for case_id, text in good_cases:
        contradictions = graph.find_anatomy_contradictions(text)
        passed = len(contradictions) == 0
        out.append(CaseResult(case_id=case_id, passed=passed, detail=str(contradictions)))

    return out


def run_generator_detection_checks(graph) -> list[CaseResult]:
    g = MedGemmaGenerator.__new__(MedGemmaGenerator)
    g.anatomy_graph = graph

    detected = g._detect_anatomy("I see a white spot on the black part of my eye")
    passed = "pupil" in detected and "cornea" not in detected
    return [CaseResult(case_id="gen_detect_01", passed=passed, detail=f"detected={sorted(detected)}")]


def run_triage_checks() -> list[CaseResult]:
    cases = [
        ("triage_01", "I have a white spot on the colored part of my eye with pain and redness", "urgent"),
        ("triage_02", "I suddenly went blind in one eye", "emergency"),
        ("triage_03", "My eyes feel dry and itchy", "none"),
    ]
    out: list[CaseResult] = []
    for case_id, query, expected in cases:
        actual = _triage_level(check_red_flags(query))
        out.append(CaseResult(case_id=case_id, passed=(actual == expected), detail=f"expected={expected} actual={actual}"))
    return out


def main() -> int:
    graph = get_eye_anatomy_graph()

    results: list[CaseResult] = []
    results.extend(run_lay_mapping_checks(graph))
    results.extend(run_contradiction_checks(graph))
    results.extend(run_generator_detection_checks(graph))
    results.extend(run_triage_checks())

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) * 100.0, 2) if total else 0.0,
        },
        "results": [asdict(r) for r in results],
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"anatomy_graph_regression_{ts}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Anatomy Graph Regression")
    print("=" * 72)
    print(f"Passed: {passed}/{total}")
    print(f"Pass rate: {report['summary']['pass_rate']}%")
    print(f"Report: {output_path}")

    failed = [r for r in results if not r.passed]
    if failed:
        print("\nFailures:")
        for item in failed:
            print(f"- {item.case_id}: {item.detail}")
        return 1

    print("\nAll anatomy graph regressions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
