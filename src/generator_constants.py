from __future__ import annotations

MEDICAL_ACRONYMS: set[str] = {
    "amd", "iop", "oct", "cnv", "dme", "erm", "pvd", "iol", "rd", "cscr",
    "faf", "ffa", "icga", "rpe", "onh", "pcv", "npdr", "pdr",
}

QUERY_NOISE_TOKENS: set[str] = {
    "patient", "question", "questions", "query", "context", "prior", "output",
    "rewrite", "rewritten", "keywords", "keyword", "clinical", "english",
    "only", "standalone", "search", "terms", "term", "include", "using",
    "what", "when", "where", "which", "this", "that", "there", "here",
    "from", "with", "into", "also", "have", "has", "had", "your", "you",
    "for", "and", "the", "to", "of", "is", "are", "be", "it",
    "differential", "diagnosis",
    "possible", "probable", "detected",
}

TOKEN_CANONICAL_MAP: dict[str, str] = {
    "retinal": "retina",
    "macular": "macula",
    "conjunctival": "conjunctiva",
    "corneal": "cornea",
    "lenticular": "lens",
    "watering": "tear",
    "tearing": "tear",
}

SURFACE_SPOT_PATTERNS: tuple[str, ...] = (
    r"white\s+(spot|patch|dot|mark|opacity|ulcer|lesion|infiltrate)",
    r"spot\s+on\s+(the\s+)?(black\s+part|front)\s+of\s+(the\s+)?eye",
)

SURFACE_LOCATION_PATTERNS: tuple[str, ...] = (
    r"black\s+part\s+of\s+(the\s+)?eye",
    r"front\s+of\s+(?:(?:the|my|your)\s+)?eye",
    r"on\s+(?:(?:the|my|your)\s+)?(eye|cornea|iris|conjunctiva)",
    r"cornea\w*|iris\w*|conjunctiva\w*",
    r"see\s+(it\s+)?in\s+(the\s+)?mirror|visible\s+in\s+(the\s+)?mirror",
)

INFLAMMATORY_FEATURE_PATTERNS: tuple[str, ...] = (
    r"red\w*",
    r"water\w*|tear\w*",
    r"pain\w*",
    r"photophobia|light\s+sensitive|sensitivity\s+to\s+light|bright\s+light",
    r"blur\w*\s+vision|vision\s+blur\w*",
    r"discharge",
)

FUNDUS_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"fundus|fundoscopy|fundoscopy|ophthalmoscopy",
    r"dilat(e|ed|ion|ing)\s+(exam|eye|pupil)",
    r"retinal\s+photo|retina\s+photo|fundus\s+photo",
    r"oct|ffa|fluorescein\s+angiography",
)

POSTERIOR_SEGMENT_MISLEADING_TOKENS: set[str] = {
    "retina", "retinal", "fundus", "leukocoria", "roth", "roths", "vascular",
}
