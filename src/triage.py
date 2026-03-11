"""
triage.py — Emergency Red Flag Triage
──────────────────────────────────────
Fast heuristic classifier to catch ocular emergencies before LLM or engine loads.
Shared by both the Streamlit frontend (app/main.py) and the CLI pipeline (engine.py).
"""

import re


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
        r'lost\s+\w*\s*vision',            # lost my vision, lost all vision
        r'loss\s+of\s+vision',
        r'went\s+blind',
        r'chemical\s*(burn|splash|injury)?',
        r'acid\s+(in|into|splash)',
        r'bleach\s+(in|into|splash)',
        r'metal\s+in\s+(my\s+)?eye',
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

    for pattern in loose_patterns:
        if re.search(pattern, q):
            return (
                "🚨 **EMERGENCY ALERT:** Based on your description, this sounds like a "
                "medical emergency that requires immediate attention to prevent permanent "
                "vision loss. **Please go to the nearest emergency room or eye casualty "
                "ward immediately.** Do not wait."
            )
    return None
