from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    tokens = [re.escape(tok) for tok in phrase.lower().strip().split() if tok]
    if not tokens:
        return re.compile(r"^$")
    return re.compile(r"\b" + r"\s+".join(tokens) + r"\b", re.I)


class EyeAnatomyGraph:
    """Deterministic eye-anatomy graph with lay-term mapping and contradiction checks."""

    def __init__(self, graph_path: str | Path | None = None):
        default_path = Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "eye_anatomy_graph.json"
        self.graph_path = Path(graph_path) if graph_path else default_path
        self.payload = self._load_payload(self.graph_path)

        self.nodes: Dict[str, dict] = {
            node["id"]: node for node in self.payload.get("nodes", []) if isinstance(node, dict) and node.get("id")
        }

        self._node_alias_patterns: Dict[str, List[re.Pattern[str]]] = {}
        for node_id, node in self.nodes.items():
            aliases = [node_id.replace("_", " ")] + list(node.get("aliases", []))
            self._node_alias_patterns[node_id] = [_phrase_pattern(a) for a in aliases if a]

        self._lay_alias_entries: List[dict] = []
        for entry in self.payload.get("lay_aliases", []):
            phrase = (entry or {}).get("phrase")
            targets = (entry or {}).get("targets", [])
            if not phrase or not targets:
                continue
            self._lay_alias_entries.append(
                {
                    "phrase": phrase,
                    "targets": [t for t in targets if t in self.nodes],
                    "pattern": _phrase_pattern(phrase),
                }
            )

        self._contradiction_rules = []
        for rule in self.payload.get("contradiction_rules", []):
            rid = (rule or {}).get("id")
            patterns = (rule or {}).get("patterns", [])
            msg = (rule or {}).get("message", "Anatomy contradiction detected.")
            if not rid or not patterns:
                continue
            compiled = [re.compile(p, re.I) for p in patterns]
            self._contradiction_rules.append((rid, compiled, msg))

        self._different_from = [
            tuple(pair)
            for pair in self.payload.get("different_from", [])
            if isinstance(pair, list) and len(pair) == 2
        ]

        self._fundus_context_pattern = re.compile(
            r"fundus|fundoscopy|ophthalmoscopy|retinal\s+photo|retina\s+photo|"
            r"dilat(e|ed|ion|ing)\s+(exam|eye|pupil)|oct|ffa|fluorescein\s+angiography",
            re.I,
        )
        self._surface_location_pattern = re.compile(
            r"front\s+of\s+(the\s+)?eye|surface\s+of\s+(the\s+)?eye|"
            r"on\s+(the\s+)?(eye|cornea|iris|pupil|conjunctiva|sclera)|"
            r"visible\s+in\s+(the\s+)?mirror|see\s+it\s+in\s+(the\s+)?mirror",
            re.I,
        )
        self._eye_reference_pattern = re.compile(r"\b(eye|ocular|cornea|iris|pupil|retina|conjunctiva|sclera|lens)\b", re.I)
        # Allow clinically plausible phrasing like "dark/black spot on sclera" without
        # treating it as the anatomy error "sclera is the black part of the eye".
        self._scleral_pigment_noncontradiction_pattern = re.compile(
            r"\b(?:black|dark|brown|pigmented)\b[^.]{0,40}\b(?:spot|dot|patch|lesion|nevus|naevus|pigment|melanosis|mole)\b[^.]{0,40}\bsclera\b"
            r"|\bsclera\b[^.]{0,40}\b(?:black|dark|brown|pigmented)\b[^.]{0,40}\b(?:spot|dot|patch|lesion|nevus|naevus|pigment|melanosis|mole)\b",
            re.I,
        )

    @staticmethod
    def _load_payload(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[EyeAnatomyGraph] Failed to load graph at {path}: {exc}")
        return {
            "nodes": [],
            "lay_aliases": [],
            "immutable_facts": [],
            "different_from": [],
            "contradiction_rules": [],
        }

    def resolve_lay_mentions(self, text: str) -> Dict[str, List[str]]:
        query = (text or "").lower()
        hits: Dict[str, List[str]] = {}
        for entry in self._lay_alias_entries:
            if entry["pattern"].search(query):
                hits[entry["phrase"]] = list(entry["targets"])
        return hits

    def detect_structures(self, text: str) -> Set[str]:
        query = (text or "").lower()
        detected: Set[str] = set()

        lay_hits = self.resolve_lay_mentions(query)
        for targets in lay_hits.values():
            detected.update(targets)

        for node_id, patterns in self._node_alias_patterns.items():
            node = self.nodes.get(node_id, {})
            if not node.get("extractable", False):
                continue
            if any(p.search(query) for p in patterns):
                detected.add(node_id)

        return detected

    def infer_query_profile(self, text: str) -> dict:
        query = (text or "").lower()
        lay_hits = self.resolve_lay_mentions(query)
        detected = self.detect_structures(query)

        surface_structures = {"cornea", "conjunctiva", "sclera", "iris", "pupil", "anterior_segment"}
        lay_targets = {t for targets in lay_hits.values() for t in targets}

        has_surface_location = bool(
            self._surface_location_pattern.search(query)
            or (lay_targets & surface_structures)
            or (detected & surface_structures)
        )

        return {
            "detected_structures": sorted(detected),
            "lay_mentions": lay_hits,
            "surface_targets": sorted((lay_targets | (detected & surface_structures)) & surface_structures),
            "has_surface_location": has_surface_location,
            "has_fundus_context": bool(self._fundus_context_pattern.search(query)),
            "has_eye_reference": bool(self._eye_reference_pattern.search(query)) or bool(detected),
        }

    def immutable_facts(self) -> List[str]:
        return [str(x) for x in self.payload.get("immutable_facts", []) if isinstance(x, str)]

    def grounding_facts_for_query(self, query: str, max_facts: int = 8) -> List[str]:
        facts: List[str] = []
        seen: Set[str] = set()

        # 1. First, get node-specific facts for mentioned structures
        mentioned = self.detect_structures(query)
        lay_hits = self.resolve_lay_mentions(query)
        for targets in lay_hits.values():
            mentioned.update(targets)

        for node_id in sorted(mentioned):
            node = self.nodes.get(node_id, {})
            fact = node.get("fact")
            if fact and fact not in seen:
                seen.add(fact)
                facts.append(fact)

        # 2. Then, add immutable facts only if we have space left
        if len(facts) < max_facts:
            for base_fact in self.immutable_facts():
                if base_fact not in seen:
                    # Check if the fact is relevant to the query to avoid clutter
                    # (Simple keyword check, excluding extremely common medical terms)
                    STOP_WORDS = {"eye", "ocular", "the", "and", "for", "with"}
                    query_words = {w for w in re.findall(r"\w+", query.lower()) if w not in STOP_WORDS}
                    fact_words = {w for w in re.findall(r"\w+", base_fact.lower()) if w not in STOP_WORDS}
                    if query_words & fact_words:
                        seen.add(base_fact)
                        facts.append(base_fact)
                        if len(facts) >= max_facts:
                            break
        
        # 3. Final fallback: If still empty, add top immutable facts
        if not facts:
            for base_fact in self.immutable_facts()[:2]:
                facts.append(base_fact)

        return facts[:max_facts]

    def find_anatomy_contradictions(self, text: str) -> List[str]:
        content = (text or "").lower()
        findings: List[str] = []
        seen: Set[str] = set()

        for rid, patterns, message in self._contradiction_rules:
            if rid == "sclera_black_or_colored" and self._scleral_pigment_noncontradiction_pattern.search(content):
                continue
            if any(p.search(content) for p in patterns):
                if message not in seen:
                    seen.add(message)
                    findings.append(message)

        for left, right in self._different_from:
            identity = r"(same as|is the same as|are the same as|equivalent to|identical to|equals?)"
            eq_pattern = re.compile(
                rf"\b{re.escape(left.replace('_', ' '))}\b[^.]{{0,40}}{identity}[^.]{{0,40}}\b{re.escape(right.replace('_', ' '))}\b",
                re.I,
            )
            eq_pattern_rev = re.compile(
                rf"\b{re.escape(right.replace('_', ' '))}\b[^.]{{0,40}}{identity}[^.]{{0,40}}\b{re.escape(left.replace('_', ' '))}\b",
                re.I,
            )
            if eq_pattern.search(content) or eq_pattern_rev.search(content):
                message = f"{left.replace('_', ' ').title()} is anatomically different from {right.replace('_', ' ').title()}."
                if message not in seen:
                    seen.add(message)
                    findings.append(message)

        return findings

    def extractable_structure_terms(self) -> List[str]:
        terms = []
        for node_id, node in self.nodes.items():
            if node.get("extractable", False):
                terms.append(node_id.replace("_", " "))
        return sorted(set(terms))


_GRAPH_SINGLETON: EyeAnatomyGraph | None = None


def get_eye_anatomy_graph() -> EyeAnatomyGraph:
    global _GRAPH_SINGLETON
    if _GRAPH_SINGLETON is None:
        _GRAPH_SINGLETON = EyeAnatomyGraph()
    return _GRAPH_SINGLETON
