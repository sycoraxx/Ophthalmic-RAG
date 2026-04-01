"""
triage.py — Emergency Red Flag Triage
──────────────────────────────────────
Fast heuristic classifier to catch ocular emergencies before LLM or engine loads.
Shared by both the Streamlit frontend (app/main.py) and the CLI pipeline (engine.py).
"""

import re


def _has_any(q: str, patterns: list[str]) -> bool:
    """Return True if any regex pattern matches the query."""
    return any(re.search(pattern, q) for pattern in patterns)


def _has_negated_symptom(q: str, symptom: str) -> bool:
    """Detect simple negated symptom phrases such as 'no pain' or 'without redness'."""
    return bool(re.search(rf"\b(no|without)\s+{symptom}\b", q))


def check_red_flags(query: str) -> str | None:
    """
    Fast regex-based classifier for ocular emergencies.
    Returns a hardcoded emergency response if triggered, otherwise None.
    """
    q = query.lower()

    # Patterns that allow intervening words (e.g. "suddenly went blind")
    loose_patterns = [
        r'sudden\w*\s+\w*\s*blind',       # suddenly went blind, sudden blindness
        r"can'?t\s+see\s+(anything|at all|nothing)",
        r"cannot\s+see\s+(anything|at all|nothing)",
        r'lost\s+\w*\s*vision',            # lost my vision, lost all vision
        r'loss\s+of\s+vision',
        r'went\s+blind',
        r'chemical\s*(burn|splash|injury)?',
        r'acid\s+(in|into|splash)',
        r'bleach\s+(in|into|splash)',
        r'metal\s+in\s+(my\s+)?eye',
        r'metal\s+went\s+into\s+(my\s+)?eye',
        r'metal.*\sinto\s+.*eye',
        r'nail\s+in\s+(my\s+)?eye',
        r'penetrating\s+(eye|ocular|injury)',
        r'puncture[ds]?\s+(eye|wound)',
        r'retinal\s+detach',
        r'curtain\s+(falling|over|across)',
        r'flash(es)?\s+(and|with)\s+floater',
        r'extreme\s+pain',
        r'eye\s+burst',
        r'globe\s+rupture',
    ]

    if _has_any(q, loose_patterns):
        return (
            "🚨 **EMERGENCY ALERT:** Based on your description, this sounds like a "
            "medical emergency that requires immediate attention to prevent permanent "
            "vision loss. **Please go to the nearest emergency room or eye casualty "
            "ward immediately.** Do not wait."
        )

    # Sight-threatening corneal infection pattern (often described as
    # "white spot on black part of eye" + red-eye symptoms).
    corneal_white_spot_patterns = [
        r'white\s+(spot|patch|dot|mark|opacity|ulcer|lesion)',
        r'black\s+part\s+of\s+(the\s+)?eye',
        r'cornea\w*',
        r'front\s+of\s+(?:(?:the|my|your)\s+)?eye',
        r'pupil\w*',
        r'see\s+(it\s+)?in\s+(the\s+)?mirror|visible\s+in\s+(the\s+)?mirror',
        r'on\s+(?:(?:the|my|your)\s+)?(eye|cornea|iris|conjunctiva)',
    ]
    inflammatory_red_eye_patterns = [
        r'red\w*',
        r'water\w*',
        r'tear\w*',
        r'pain\w*',
        r'photophobia|light\s+sensitive|sensitivity\s+to\s+light|bright\s+light',
        r'blur\w*\s+vision|vision\s+blur\w*',
        r'discharge',
    ]
    contact_lens_patterns = [
        r'contact\s+lens',
        r'lens\s+wearer',
    ]

    has_white_spot = _has_any(q, [corneal_white_spot_patterns[0]])
    has_corneal_location = _has_any(q, corneal_white_spot_patterns[1:5])
    has_visible_surface_language = _has_any(q, corneal_white_spot_patterns[5:])
    inflammatory_hits = sum(bool(re.search(p, q)) for p in inflammatory_red_eye_patterns)
    has_red_eye_features = inflammatory_hits >= 1
    has_negative_redness = _has_negated_symptom(q, r"red\w*")
    has_negative_pain = _has_negated_symptom(q, r"pain\w*")
    has_combined_negation = bool(
        re.search(r"\bno\s+red\w*\s+or\s+pain\w*\b", q)
        or re.search(r"\bno\s+pain\w*\s+or\s+red\w*\b", q)
    )
    has_contact_lens_risk = _has_any(q, contact_lens_patterns) and (
        _has_any(q, [r'red\w*|pain\w*|water\w*|tear\w*|photophobia|white\s+(spot|patch|dot|ulcer)'])
    ) and not (has_combined_negation or (has_negative_redness and has_negative_pain))

    if (has_white_spot and has_red_eye_features and (has_corneal_location or has_visible_surface_language)) or has_contact_lens_risk:
        return (
            "⚠️ **URGENT EYE ALERT (Same-Day):** A **white spot on the front/black part of the eye** "
            "with redness/watering can indicate **infectious keratitis or a corneal ulcer**, which may "
            "threaten vision if treatment is delayed. **Please seek same-day in-person care (within 24 hours) at an eye "
            "hospital/eye casualty.**\n\n"
            "Until examined: avoid contact lenses, avoid steroid eye drops unless prescribed, and do not rub the eye."
        )

    return None
