"""
prompt_templates.py — Modality-Aware Ophthalmic Prompt Templates
────────────────────────────────────────────────────────────────────────
Templates mapped by modality (anterior, posterior, both, external).
This prevents "modality drag" (e.g., averaging an anterior disease
with a "fundus photograph" template, which degrades the embedding).
"""

# Generic templates applied to ALL labels.
# Since labels like 'OCT, drusen' or 'CFP, glaucoma' explicitly include
# their modality, we should avoid redundant or contradictory wrapping.
GENERIC_TEMPLATES: list[str] = [
    "{}",
]

# Modality-specific templates.
# Kept empty because the labels themselves handle modality definition.
MODALITY_TEMPLATES: dict[str, list[str]] = {
    "anterior": [],
    "posterior": [],
    "both": [],
    "external": [],
}


def get_templates_for_modality(modality: str) -> list[str]:
    """Return the combined generic + modality-specific templates."""
    templates = GENERIC_TEMPLATES.copy()
    if modality in MODALITY_TEMPLATES:
        templates.extend(MODALITY_TEMPLATES[modality])
    return list(set(templates))  # Remove duplicates if any
