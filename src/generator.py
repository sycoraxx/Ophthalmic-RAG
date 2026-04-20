"""
generator.py — Handles all MedGemma loading, prompting, and inference.
UPDATED: Session state support, EyeCLIP-integrated entity extraction, 
         anatomical consistency enforcement, and history summarization.
──────────────────────────────────────────────────────────────────────────────
Provides methods for query refinement, answer generation, and grounding
verification. Supports MULTIMODAL inputs (text + images) natively via
MedGemma's vision encoder (SigLIP).

MedGemma 1.5 4B IT is loaded as AutoModelForImageTextToText with an
AutoProcessor, enabling direct image understanding alongside text.
"""

import os
import re
import torch
import json
from typing import Optional, List, Dict, Any
from transformers import AutoProcessor, AutoModelForImageTextToText
from langchain_core.documents import Document
from PIL import Image as PILImage

# Import session state components
from src.state.clinical_session_state import ClinicalSessionState, ClinicalEntity, EntityType
from src.state.clinical_entity_extractor import ClinicalEntityExtractor
from src.anatomy import get_eye_anatomy_graph

MEDGEMMA_MODEL = "./models/checkpoints/medgemma-1.5-4b-it"

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

TOKEN_CANONICAL_MAP: Dict[str, str] = {
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


class MedGemmaGenerator:
    """Manages MedGemma 1.5 4B for refinement, generation, and verification.
    
    Now uses AutoModelForImageTextToText + AutoProcessor to support
    multimodal (image + text) inputs natively.
    
    NEW FEATURES:
    - ClinicalEntityExtractor with EyeCLIP integration
    - Session state support for context-aware generation
    - History summarization for small context windows
    - Anatomical consistency enforcement
    """

    def __init__(self):
        print(f"Loading {MEDGEMMA_MODEL}...")
        self.processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL)
        self.medgemma = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.medgemma.eval()

        # Shared deterministic anatomy graph for parsing/retrieval/generation guardrails.
        self.anatomy_graph = get_eye_anatomy_graph()
        
        # Initialize entity extractor with EyeCLIP integration
        self.entity_extractor = ClinicalEntityExtractor(self)
        
        print("MedGemma + Entity Extractor loaded.")

    # ── Internal: Unified Inference ──────────────────────────────────────────
    def _generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.4,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        skip_thought: bool = False,
    ) -> str:
        """
        Run MedGemma inference on a list of chat messages.
        Messages can contain text-only or multimodal (text + image) content.
        """
        # Normalize: AutoProcessor requires content to be a list of typed dicts
        normalized = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            
            # Inject a fast-path command if requested to skip the CoT wait time
            if skip_thought and msg["role"] == "system":
                # Ensure the text element exists
                if len(content) > 0 and content[0].get("type") == "text":
                    content[0]["text"] += "\n\nCRITICAL: DO NOT use <thought> or 'Constraint Checklist'. Write the direct answer IMMEDIATELY."
            
            normalized.append({"role": msg["role"], "content": content})
        messages = normalized

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.medgemma.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Build generation kwargs to avoid HF warnings when do_sample=False
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = self.medgemma.generate(
                **inputs,
                **gen_kwargs
            )

        generation = outputs[0][input_len:]
        out = self.processor.decode(generation, skip_special_tokens=True).strip()
        
        raw_out = out # Keep for fallback
        
        # ─── Cleaning Logic ──────────────────────────────────────────────────
        # 1. Regex for when special tokens are present or literal
        out = re.sub(r'(?s)<unused94>thought.*?(?:<unused95>|\Z)', '', out).strip()
        
        # 2. Heuristic for skip_special_tokens=True which drops <unused94>
        # Check for both "thought" and "thought:"
        if out.lower().startswith("thought"):
            lines = out.split('\n')
            checklist_idx = -1
            for i, line in enumerate(lines):
                if 'constraint checklist' in line.lower():
                    checklist_idx = i
                    break
            
            if checklist_idx != -1:
                end_idx = len(lines)
                for i in range(checklist_idx + 1, len(lines)):
                    line = lines[i]
                    # The actual answer usually starts completely unindented 
                    # Checklist items are usually indented or start with a bullet
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        # Check if it's a checklist item like "- item: Yes" or "* item: Yes"
                        if re.match(r'^[-*]\s+.*?:\s*(?:Yes|No)', line):
                            continue
                        end_idx = i
                        break
                out = '\n'.join(lines[end_idx:]).strip()
            else:
                # If no constraint checklist, just strip up to the first double newline after thought
                parts = out.split('\n\n')
                if len(parts) > 1:
                    out = '\n\n'.join(parts[1:]).strip()

        # 3. Last resort fallback: Check if the end of the string is literally just the checklist
        out = re.sub(r'(?i)Constraint Checklist.*?$', '', out, flags=re.DOTALL)
        
        # Strip trailing newlines or extra asterisks/bullets left by the checklist
        out = re.sub(r'^\s*[-*]\s*.*?:\s*Yes\s*$', '', out, flags=re.MULTILINE)
        out = out.strip()
        
        # 4. Emergency Fallback: If stripping made it empty, the model only output
        # the thought/checklist (no closing <unused95>). Attempt a last-resort strip
        # of the raw output by looking for the first blank line after the thought
        # header, then returning whatever follows.
        if not out:
            # Try to recover the answer portion from raw_out by finding where the
            # thought section ends. The actual answer usually begins after the first
            # double-newline following the thought opening.
            stripped = re.sub(r'(?si)^.*?<unused94>.*?\n\n', '', raw_out, count=1).strip()
            if not stripped:
                # If still empty or raw_out had no double-newline, try after first \n
                stripped = re.sub(r'(?si)^.*?<unused94>[^\n]*\n', '', raw_out, count=1).strip()
            if not stripped:
                # Absolute fallback: strip known thought prefixes
                stripped = re.sub(r'(?i)^thought:?\s*', '', raw_out).strip()
            out = stripped or ""
                    
        return out

    def _get_anatomy_graph(self):
        graph = getattr(self, "anatomy_graph", None)
        if graph is None:
            graph = get_eye_anatomy_graph()
            self.anatomy_graph = graph
        return graph

    # ── Anatomy Detection Helper ─────────────────────────────────────────────
    ANATOMY_KEYWORDS = {
        "retina": ["retina", "retinal", "macula", "fovea", "RPE", "photoreceptor", 
                   "OCT", "drusen", "AMD", "macular", "subretinal"],
        "lens": ["lens", "cataract", "crystalline", "accommodation", "IOL", 
                 "phakic", "pseudophakic", "capsule"],
        "cornea": ["cornea", "corneal", "kerat", "epithelium", "endothelium", 
                   "stroma", "keratoconus", "ulcer"],
        "optic_nerve": ["optic nerve", "optic disc", "glaucoma", "cup", "rim", 
                        "papilledema", "ONH"],
        "choroid": ["choroid", "choroidal", "choriocapillaris", "CNV", "polyp"],
        "anterior_chamber": ["anterior chamber", "angle", "trabecular", "IOP", 
                             "gonio", "hyphema"],
        "conjunctiva": ["conjunctiva", "conjunctival", "pterygium", "pinguecula", 
                        "subconjunctival"],
        "vitreous": ["vitreous", "vitreoretinal", "floaters", "PVD", "hemorrhage"],
    }

    def _detect_anatomy(self, text: str) -> set:
        """Detect which anatomical structures are mentioned in text."""
        text_lower = text.lower()
        detected = set()

        graph = self._get_anatomy_graph()
        graph_detected = graph.detect_structures(text_lower)
        mapping = {
            "optic disc": "optic_nerve",
            "optic nerve": "optic_nerve",
            "anterior chamber": "anterior_chamber",
        }
        for structure in graph_detected:
            detected.add(mapping.get(structure.replace("_", " "), structure))

        for anatomy, keywords in self.ANATOMY_KEYWORDS.items():
            if any(kw.lower() in text_lower for kw in keywords):
                detected.add(anatomy)
        return detected

    # ── Query Refinement (History-Free for Precision) ────────────────────────
    def refine_query(
        self,
        raw_query: str,
        recent_history: Optional[list] = None,  # Kept for backward compat but NOT used
        image_path: Optional[str] = None,
        visual_findings: Optional[str] = None,
    ) -> str:
        """Translate layperson queries to clinical terminology.
        
        Focused on current query ONLY for precise retrieval.
        Conversation history is used later in generation, not retrieval.
        """
        # Safety-first bypass: if query language suggests a potentially sight-threatening
        # corneal surface presentation, skip free-form LLM rewriting to avoid semantic drift.
        mapping = self._surface_sign_profile(raw_query)
        if mapping["high_risk_surface"]:
            seeded = (
                "cornea corneal infiltrate corneal ulcer infectious keratitis microbial keratitis "
                "red eye tearing pain photophobia urgent ophthalmology"
            )
            return self.apply_symptom_sign_mapping_to_query(raw_query, seeded, max_terms=15)

        # Non-inflammatory surface spot (e.g. "white spot on black part of eye" without
        # redness/pain) → bias to anterior-segment differentials (leukocoria, cataract,
        # corneal opacity) and suppress posterior-segment distractors.
        if mapping.get("has_surface_spot") and not mapping["high_risk_surface"]:
            seeded = (
                "leukocoria white pupillary reflex cataract corneal opacity "
                "corneal scar anterior segment pupil abnormality"
            )
            return self.apply_symptom_sign_mapping_to_query(raw_query, seeded, max_terms=15)

        system_prompt = (
            "You are a clinical search query generator for ophthalmology.\n"
            "Convert the patient's question into a SHORT search query using "
            "clinical ophthalmological terminology.\n\n"
            "STRICT OUTPUT RULES:\n"
            "- Output ONLY the search keywords. Nothing else.\n"
            "- Maximum 13 words.\n"
            "- No sentences, no explanations, no descriptions.\n"
            "- No JSON, no code, no function calls, no formatting.\n"
            "- Do NOT describe what you see in the image.\n"
            "- Do NOT explain your reasoning.\n"
            "- If an image is attached, identify the condition and put it in the query.\n"
            "- If prior topics exist, maintain clinical context.\n"
            "EXAMPLES:\n"
            "Patient: my eye hurts when I look at bright lights\n"
            "photophobia etiology differential diagnosis\n\n"
            "Patient: I get headaches when reading up close\n"
            "asthenopia convergence insufficiency presbyopia\n\n"
            "Patient: [image of red eye] what is wrong with my eye\n"
            "conjunctival injection anterior uveitis keratitis differential\n\n"
            "Patient: my kid's eyes are crossing\n"
            "pediatric esotropia strabismus childhood onset"
        )

        # Build EyeCLIP hint from visual findings
        eyeclip_hint = ""
        if visual_findings:
            parsed = self._parse_eyeclip_findings(visual_findings)
            if parsed and parsed.get("conditions"):
                # Filter out anything below 10%
                valid_conds = [c for c in parsed["conditions"] if c["score"] >= 4.0]
                conds = [f"{c['name']} ({c['score']}%)" for c in valid_conds]
                cond_str = ", ".join(conds) if conds else "Normal/Unremarkable"
                modality = parsed.get("modality") or "Unknown"
                eyeclip_hint = f" [Image Findings: Modality={modality}, Findings={cond_str}]"
            else:
                short_findings = visual_findings.split("----------------------------------------")[0].strip()
                eyeclip_hint = f" [Image Findings: {short_findings}]"

        # Build Anatomy Graph Hint
        anatomy_hint = ""
        graph = self._get_anatomy_graph()
        anatomy_profile = graph.infer_query_profile(raw_query)
        lay_mentions = anatomy_profile.get("lay_mentions", {})
        
        if lay_mentions:
            hint_parts = []
            for phrase, targets in lay_mentions.items():
                hint_parts.append(f"'{phrase}' -> {', '.join(targets)}")
            anatomy_hint += f" [Anatomy Translation: {'; '.join(hint_parts)}]"
            
        facts = graph.grounding_facts_for_query(raw_query, max_facts=2)
        if facts:
            anatomy_hint += f" [Anatomy Facts: {' '.join(facts)}]"

        user_text = f"Patient:{eyeclip_hint}{anatomy_hint} {raw_query}"
        user_content = self._build_multimodal_content(user_text, image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        refined = self._generate(
            messages,
            max_new_tokens=80,
            temperature=0.1,
            top_p=0.15,
            do_sample=True,
            repetition_penalty=1.1,
            skip_thought=True,
        )

        # Post-process aggressively to remove repetition and prompt leakage.
        refined = self.normalize_retrieval_query(refined, max_terms=13)

        # Fallback to original if refinement is too short, then apply symptom-sign mapping.
        base_query = refined if len(refined) >= 5 else raw_query
        return self.apply_symptom_sign_mapping_to_query(raw_query, base_query, max_terms=13)

    def _parse_eyeclip_findings(self, visual_findings: str) -> Optional[dict]:
        """Parse EyeCLIP output into structured format."""
        if not visual_findings:
            return None
        
        result = {"modality": None, "conditions": []}
        
        # Parse modality
        mod_match = re.search(r'Detected Image Type:\s*(\w+)', visual_findings, re.I)
        if mod_match:
            result["modality"] = mod_match.group(1).upper()
        
        # Parse conditions: "● Probable: drusen (69.0%)" or "○ Possible: scar (5.0%)"
        pattern = r'[●○]\s*(Probable|Possible|Detected):\s*([^(]+?)\s*\(([\d.]+)%\)'
        for match in re.finditer(pattern, visual_findings, re.I):
            conf_label = match.group(1).lower()
            name = match.group(2).strip()
            score = float(match.group(3))
            result["conditions"].append({
                "name": name,
                "confidence": conf_label,
                "score": score,
            })
        
        return result if result["conditions"] else None

    def _clean_query_output(self, text: str) -> str:
        """Clean query output from JSON/formatting artifacts."""
        text = re.sub(r'```\w*\s*', '', text)
        text = re.sub(r'[{}\[\]"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Handle JSON wrapper if present
        json_match = re.search(r'"(?:refined_query|query|text)":\s*"([^"]+)"', text)
        if json_match:
            text = json_match.group(1)
        return text

    def normalize_retrieval_query(
        self,
        text: str,
        max_terms: int = 14,
        banned_tokens: Optional[set[str]] = None,
    ) -> str:
        """Normalize query text into compact, deduplicated retrieval keywords."""
        if not text:
            return ""

        cleaned = self._clean_query_output(text)

        # Keep the trailing payload if prompt examples leaked into output.
        lowered = cleaned.lower()
        for marker in ("output:", "query:", "patient:", "context:", "prior questions:"):
            idx = lowered.rfind(marker)
            if idx != -1:
                cleaned = cleaned[idx + len(marker):].strip()
                lowered = cleaned.lower()

        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", cleaned)

        deduped: List[str] = []
        seen = set()
        for raw_tok in tokens:
            tok = raw_tok.lower().strip("'\".,;:!?()[]{}")
            if not tok:
                continue
            canonical = TOKEN_CANONICAL_MAP.get(tok)
            if canonical:
                tok = canonical
            if len(tok) <= 2 and tok not in MEDICAL_ACRONYMS:
                continue
            if tok in QUERY_NOISE_TOKENS and tok not in MEDICAL_ACRONYMS:
                continue
            if banned_tokens and tok in banned_tokens and tok not in MEDICAL_ACRONYMS:
                continue
            if tok in seen:
                continue

            seen.add(tok)
            deduped.append(tok)
            if len(deduped) >= max_terms:
                break

        return " ".join(deduped)

    def _surface_sign_profile(self, text: str) -> Dict[str, bool]:
        """Classify whether language suggests a visible anterior-segment emergency."""
        q = (text or "").lower()
        anatomy_profile = self._get_anatomy_graph().infer_query_profile(q)

        has_spot = any(re.search(p, q) for p in SURFACE_SPOT_PATTERNS)
        has_surface_location = (
            any(re.search(p, q) for p in SURFACE_LOCATION_PATTERNS)
            or anatomy_profile.get("has_surface_location", False)
        )
        inflammatory_hits = sum(bool(re.search(p, q)) for p in INFLAMMATORY_FEATURE_PATTERNS)
        has_inflammation = inflammatory_hits >= 1
        has_fundus_context = (
            any(re.search(p, q) for p in FUNDUS_CONTEXT_PATTERNS)
            or anatomy_profile.get("has_fundus_context", False)
        )
        has_eye_reference = (
            bool(re.search(r"\b(eye|cornea|iris|conjunctiva|sclera|pupil)\b", q))
            or anatomy_profile.get("has_eye_reference", False)
        )

        contact_lens_surface_risk = bool(re.search(r"contact\s+lens|lens\s+wearer", q)) and has_inflammation

        high_risk_surface = (
            (
                has_spot
                and has_inflammation
                and (has_surface_location or has_eye_reference)
            )
            or (contact_lens_surface_risk and has_spot)
        ) and not has_fundus_context

        # NEW: Surface spot without inflammation still indicates anterior-segment
        # pathology (leukocoria, corneal opacity, cataract). Must NOT be mapped
        # to posterior-segment findings like cherry-red spot.
        has_surface_spot = (
            has_spot
            and (has_surface_location or has_eye_reference)
            and not has_fundus_context
        )

        return {
            "has_spot": has_spot,
            "has_surface_location": has_surface_location,
            "has_inflammation": has_inflammation,
            "has_fundus_context": has_fundus_context,
            "has_eye_reference": has_eye_reference,
            "surface_targets": anatomy_profile.get("surface_targets", []),
            "high_risk_surface": high_risk_surface,
            "has_surface_spot": has_surface_spot,
        }

    def apply_symptom_sign_mapping_to_query(
        self,
        raw_query: str,
        candidate_query: str,
        max_terms: int = 18,
    ) -> str:
        """
        Enforce clinically safe symptom-sign mapping for retrieval.

        If user describes a visible white spot on the eye surface with inflammatory
        symptoms, bias retrieval to corneal surface emergencies and suppress
        posterior-segment distractors unless fundus context is explicit.
        """
        mapping = self._surface_sign_profile(raw_query)
        merged = f"{candidate_query} {raw_query}".strip()

        if mapping["high_risk_surface"]:
            seeded = (
                "cornea corneal infiltrate corneal ulcer infectious keratitis microbial keratitis "
                "red eye tearing pain photophobia surface lesion urgent ophthalmology"
            )
            merged = f"{seeded} {merged}"
        elif mapping.get("has_surface_spot"):
            # Non-inflammatory surface spot → anterior-segment differentials
            seeded = (
                "leukocoria white pupillary reflex cataract corneal opacity "
                "corneal scar anterior segment pupil abnormality"
            )
            merged = f"{seeded} {merged}"

        banned_tokens = None
        if (mapping["high_risk_surface"] or mapping.get("has_surface_spot")) and not mapping["has_fundus_context"]:
            banned_tokens = POSTERIOR_SEGMENT_MISLEADING_TOKENS

        normalized = self.normalize_retrieval_query(
            merged,
            max_terms=max_terms,
            banned_tokens=banned_tokens,
        )
        return normalized if normalized else self.normalize_retrieval_query(candidate_query, max_terms=max_terms)

    # ── Context-Aware Query Rewriting for Follow-ups ─────────────────────────
    def rewrite_query_for_retrieval(
        self,
        current_query: str,
        session_state: Optional[ClinicalSessionState] = None,
        visual_findings: Optional[str] = None,
        image_path: Optional[str] = None,
        recent_history: Optional[list] = None,
    ) -> str:
        """
        Rewrite a follow-up query into a standalone, retrieval-optimized query
        using pinned clinical context from session state and prior message history.
        
        Example:
          Input: "What to do now?" + {condition: "drusen", anatomy: "retina"} + recent_history="I also have floaters"
          Output: "drusen AMD management monitoring treatment options"
        """
        # Build context injection from session state
        state_context = session_state.to_query_context(include_provisional=True) if session_state else ""
        
        # Build EyeCLIP hint
        eyeclip_hint = ""
        if visual_findings:
            parsed = self._parse_eyeclip_findings(visual_findings)
            if parsed:
                probable = [c["name"] for c in parsed["conditions"] if c["confidence"] == "probable"]
                if probable:
                    eyeclip_hint = f" [EyeCLIP: {', '.join(probable)}]"
        
        # Build history context
        history_str = ""
        if recent_history:
            # Drop the current query from history if it's there
            hist_to_use = [q for q in recent_history if q != current_query]
            if hist_to_use:
                history_str = f"\nPrior Questions: {' → '.join(hist_to_use)}"

        # Build Anatomy Graph Hint
        anatomy_hint = ""
        graph = self._get_anatomy_graph()
        anatomy_profile = graph.infer_query_profile(current_query)
        lay_mentions = anatomy_profile.get("lay_mentions", {})
        
        if lay_mentions:
            hint_parts = []
            for phrase, targets in lay_mentions.items():
                hint_parts.append(f"'{phrase}' -> {', '.join(targets)}")
            anatomy_hint += f"\nAnatomy Translation: {'; '.join(hint_parts)}"
            
        facts = graph.grounding_facts_for_query(current_query, max_facts=2)
        if facts:
            anatomy_hint += f"\nAnatomy Facts: {' '.join(facts)}"
        
        system_prompt = (
            "You are a clinical query rewriter for ophthalmology RAG.\n"
            "Rewrite the patient's follow-up question into a STANDALONE search query "
            "that includes necessary context from prior conversation.\n\n"
            "RULES:\n"
            "- Output ONLY 5-10 clinical keywords. Nothing else.\n"
            "- If the query is ambiguous (e.g., 'What to do now?'), use the provided "
            "  clinical context and Prior Questions to make it specific.\n"
            "- Include anatomy and condition terms from the context injection.\n"
            "- If the user mentions 'it/this/that', resolve it using the Context and Prior Questions.\n"
            "- If EyeCLIP findings are provided, include detected conditions.\n"
            "- Output in ENGLISH only.\n\n"
            "EXAMPLES:\n"
            "Context: [anatomy:retina | condition:age-related macular degeneration]\n"
            "Prior Questions: What is AMD?\n"
            "Query: What to do now?\n"
            "Output: AMD management monitoring AREDS treatment progression\n\n"
            "Context: [anatomy:retina | condition:diabetic retinopathy]\n"
            "Prior Questions: I have blurry vision\n"
            "Query: Is this serious?\n"
            "Output: diabetic retinopathy severity staging progression risk\n\n"
            "Context: [anatomy:lens | condition:cataract]\n"
            "Prior Questions: My doctor said I have a cataract\n"
            "Query: What are my options?\n"
            "Output: cataract surgery options IOL timing recovery"
        )
        
        user_text = f"Context:{state_context}{eyeclip_hint}{anatomy_hint}{history_str}\nPatient: {current_query}"
        user_content = self._build_multimodal_content(user_text, image_path)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        rewritten = self._generate(
            messages,
            max_new_tokens=80,
            temperature=0.1,
            top_p=0.15,
            repetition_penalty=1.1,
            skip_thought=True,
        )
        
        # Post-process: keep rewritten query compact and retrieval-safe.
        rewritten = self.normalize_retrieval_query(rewritten, max_terms=12)

        # Fallback to current query if rewriting yielded nothing or too little.
        base_query = rewritten if len(rewritten) >= 5 else current_query
        return self.apply_symptom_sign_mapping_to_query(current_query, base_query, max_terms=12)

    # ── Context Formatting ───────────────────────────────────────────────────
    def build_context_block(self, context_docs: list[Document]) -> str:
        """Format retrieved documents into a context block.
        
        Uses 2000 chars per source (safe within MedGemma's 8K context).
        With k=3 sources: ~6000 chars ≈ 1500 tokens context, well under the limit.
        """
        parts = []
        for i, doc in enumerate(context_docs, 1):
            src = doc.metadata.get("source", "Unknown")
            path = doc.metadata.get("section_path", "General")
            # Use 2000 chars to preserve treatment/mechanism details that
            # were being lost at 800 chars, causing confabulation.
            content = doc.page_content[:2000].strip()
            parts.append(f"[Source {i}: {src} — {path}]\n{content}")
        return "\n\n---\n\n".join(parts)

    def _build_patient_context(self, patient_profile: Optional[dict]) -> str:
        if not patient_profile:
            return ""
        parts = []
        for key, value in patient_profile.items():
            if value:
                label = key.replace("_", " ").title()
                parts.append(f"- {label}: {value}")
        if not parts:
            return ""
        return "PATIENT PROFILE (use this to personalize your advice):\n" + "\n".join(parts) + "\n\n"

    def _build_history_block(self, recent_history: Optional[list]) -> str:
        """Build history block with summarization for small context windows."""
        if not recent_history:
            return ""
        
        # If short, use verbatim
        if len(recent_history) <= 2:
            return "\n".join([f"Patient: {q}" for q in recent_history]) + "\n\n"
        
        # Otherwise: summarize older, keep recent verbatim
        summary = self._summarize_history_for_generation(recent_history[:-2])
        recent = "\n".join([f"Patient: {q}" for q in recent_history[-2:]])
        return f"{summary}{recent}\n\n"

    def _summarize_history_for_generation(
        self,
        recent_history: list,
        max_summary_tokens: int = 150,
    ) -> str:
        """Compress conversation history into brief clinical summary."""
        if not recent_history:
            return ""
        
        summary_prompt = (
            "Summarize this ophthalmology conversation into 2-3 bullet points.\n"
            "Focus on: symptoms, anatomical structures, conditions identified.\n"
            "Output format:\n- [bullet 1]\n- [bullet 2]\n- [bullet 3]\n\n"
            f"History:\n" + "\n".join([f"Q: {h}" for h in recent_history[-5:]])
        )
        
        messages = [
            {"role": "system", "content": "You are a clinical note summarizer. Be concise."},
            {"role": "user", "content": summary_prompt},
        ]
        
        summary = self._generate(
            messages,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
        )
        
        summary = summary.strip().lstrip("- ").replace("\n\n", "\n")
        return f"CONVERSATION SUMMARY:\n{summary}\n\n" if summary else ""

    # ── Multimodal Content Builder ───────────────────────────────────────────
    def _build_multimodal_content(
        self, text: str, image_path: Optional[str] = None
    ) -> list[dict] | str:
        """
        Build message content for MedGemma.
        If image_path exists, returns list of content parts; otherwise plain text.
        """
        if image_path and os.path.exists(image_path):
            try:
                pil_image = PILImage.open(image_path).convert("RGB")
                return [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": text},
                ]
            except Exception as e:
                print(f"[MedGemma] ⚠ Failed to load image {image_path}: {e}")
        return text

    def _apply_anatomy_guardrails_to_answer(
        self,
        *,
        raw_query: str,
        draft_answer: str,
        context_block: str,
        anatomy_facts: List[str],
    ) -> str:
        """Rewrite answer once if deterministic anatomy checks detect contradictions."""
        graph = self._get_anatomy_graph()
        contradictions = graph.find_anatomy_contradictions(draft_answer)
        if not contradictions:
            return draft_answer

        facts_block = "\n".join(f"- {fact}" for fact in anatomy_facts[:8])
        contradiction_block = "\n".join(f"- {issue}" for issue in contradictions)

        system_prompt = (
            "You are a clinical safety editor for ophthalmology answers.\n"
            "Revise the draft answer to remove anatomy contradictions while preserving tone and useful advice.\n"
            "Do not invent new diagnoses and do not contradict the verified anatomy facts.\n"
            "Output ONLY the corrected final answer."
        )
        user_prompt = (
            f"PATIENT QUESTION:\n{raw_query}\n\n"
            f"VERIFIED ANATOMY FACTS:\n{facts_block}\n\n"
            f"DETECTED CONTRADICTIONS:\n{contradiction_block}\n\n"
            f"SOURCE CONTEXT:\n{context_block}\n\n"
            f"DRAFT ANSWER:\n{draft_answer}"
        )

        corrected = self._generate(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.05,
            skip_thought=True,
        ).strip()

        if not corrected:
            return draft_answer

        if graph.find_anatomy_contradictions(corrected):
            # Conservative fail-safe: emit minimal guidance.
            # We avoid dumping raw facts here as it can look like a technical error to the patient.
            return (
                "I detected a potential clinical inconsistency in my generated response regarding eye anatomy. "
                "To ensure your safety, please seek an in-person eye examination for an accurate diagnosis, "
                "especially if you are experiencing pain, persistent redness, or any sudden changes in your vision."
            )

        return corrected

    # ── Answer Generation (Multimodal + Session State) ───────────────────────
    def generate_answer(
        self,
        raw_query: str,
        context_docs: list[Document],
        session_state: Optional[ClinicalSessionState] = None,  # NEW
        correction_context: Optional[str] = None,
        patient_profile: Optional[dict] = None,
        recent_history: Optional[list] = None,
        visual_findings: Optional[str] = None,
        image_path: Optional[str] = None,
        target_language: str = "English",
    ) -> str:
        """
        Generate a patient-friendly answer from retrieved context documents.
        
        NEW: session_state parameter for context-aware personalization.
        """
        context_block = self.build_context_block(context_docs)
        patient_block = self._build_patient_context(patient_profile)
        history_block = self._build_history_block(recent_history)
        
        # Build session state context for generation
        state_block = ""
        if session_state:
            state_context = session_state.to_generation_context(include_provisional=True)
            if state_context:
                state_block = f"\nCLINICAL CONTEXT FROM PRIOR CONVERSATION:\n{state_context}\n\n"

        anatomy_graph = self._get_anatomy_graph()
        anatomy_facts = anatomy_graph.grounding_facts_for_query(raw_query, max_facts=4)
        anatomy_facts_block = ""
        if anatomy_facts:
            anatomy_facts_block = "ANATOMICAL CONTEXT HINTS:\n" + "\n".join(
                f"- {fact}" for fact in anatomy_facts
            ) + "\n\n"
        
        # Check anatomical consistency
        query_anatomy = self._detect_anatomy(raw_query)
        context_anatomy = set()
        for doc in context_docs:
            doc_anat = doc.metadata.get("anatomy", set())
            if isinstance(doc_anat, str):
                doc_anat = {doc_anat}
            context_anatomy.update(doc_anat)
        
        anatomy_mismatch = query_anatomy and context_anatomy and not query_anatomy.intersection(context_anatomy)

        mapping = self._surface_sign_profile(raw_query)
        high_risk_corneal_pattern = mapping["high_risk_surface"]
        has_surface_visible_pattern = mapping["has_spot"] and mapping["has_surface_location"]
        has_fundus_context = mapping["has_fundus_context"]
        
        base_rules = (
            "RULES:\n"
            "1. Use simple, everyday language. Explain medical terms clearly.\n"
            "2. Be warm and reassuring but honest. Do not make firm diagnoses.\n"
            "3. Structure: direct answer → possible causes → safety advice and when to seek care. "
            "Include home care ONLY when presentation appears mild and non-urgent.\n"
            "4. Cite sources as [Source N] ONLY if fact appears in that source. NEVER invent citations.\n"
            "5. CRITICAL GROUNDING RULE: Every specific medical claim (diagnosis, treatment, "
            "mechanism of action, statistic, anatomical fact) MUST be directly traceable to the "
            "MEDICAL REFERENCE TEXTS provided below. If the references do NOT contain the specific "
            "information needed to answer the question, you MUST say: "
            "'The available medical references don't specifically address this point. "
            "Please consult your ophthalmologist for a definitive answer.' "
            "Do NOT fill in from general medical knowledge. Do NOT guess.\n"
            "6. Keep answer concise (150-250 words).\n"
            "7. If CLINICAL CONTEXT is provided, personalize advice accordingly.\n"
            "8. Maintain continuity with prior conversation topics.\n"
            "9. If VISUAL FINDINGS are provided, use them as supporting evidence "
            "alongside the MEDICAL REFERENCE TEXTS. Do not treat them as standalone diagnosis.\n"
            "10. If an IMAGE is attached, your interpretation must be grounded in the "
            "MEDICAL REFERENCE TEXTS. Do not invent findings beyond what EyeCLIP reported "
            "and what the sources support.\n"
            "11. Strict anatomy constraint: If the patient reports a visible spot/patch on the eye "
            "that can be seen externally (including mirror-visible language), treat it primarily as "
            "an anterior/surface finding (cornea, conjunctiva, iris) unless explicit posterior-segment "
            "evidence is provided.\n"
        )

        if not has_fundus_context:
            base_rules += (
                "12. Negative constraint: Do NOT mention Roth spots or retinal vascular changes "
                "unless the user explicitly mentions dilated fundus exam, retinal photo, fundus image, "
                "or equivalent posterior-segment imaging.\n"
            )

        if has_surface_visible_pattern and not has_fundus_context:
            base_rules += (
                "13. For visible surface-spot presentations, avoid posterior-segment anchoring "
                "(retina/fundus vascular causes) unless supported by explicit retinal examination data.\n"
            )

        if high_risk_corneal_pattern:
            base_rules += (
                "14. HIGH-RISK PATTERN: White corneal/front-eye spot with inflammatory symptoms. "
                "Prioritize possible infectious keratitis/corneal ulcer in differential and explicitly recommend "
                "urgent in-person ophthalmic evaluation within 24 hours to reduce risk of permanent vision loss.\n"
                "15. For high-risk pattern, avoid suggesting home care as sufficient treatment. "
                "You may include temporary precautions, but urgent in-person care must be the main recommendation.\n"
            )
        
        if anatomy_mismatch:
            base_rules += (
                "16. ANATOMICAL MISMATCH DETECTED: The retrieved sources discuss a different eye structure "
                "than the patient's question. Acknowledge this limitation and suggest consulting a specialist.\n"
            )
        
        if target_language != "English":
            base_rules += (
                f"17. TRANSLATION REQUIRED: Write ENTIRE response in {target_language}.\n"
            )
        
        if correction_context:
            system_prompt = (
                "You are a friendly eye health assistant.\n\n"
                f"{patient_block}{anatomy_facts_block}{base_rules}\n"
                "CRITICAL CORRECTION: Previous answer contained unsupported claims:\n"
                f"{correction_context}\n"
                "DO NOT repeat these. ONLY state facts from source texts.\n"
            )
        else:
            system_prompt = (
                "You are a friendly eye health assistant.\n\n"
                f"{patient_block}{anatomy_facts_block}{base_rules}"
            )
        
        vision_block = ""
        if visual_findings and image_path:
            # For final generation, provide the FULL EyeCLIP findings (including low confidences)
            # This helps MedGemma frame differential diagnoses and uncertainty properly.
            vision_block = f"\n[IMAGE ANALYSIS RESULTS (Include in differential if relevant)]\n{visual_findings}\n"
        
        text_content = (
            f"{history_block}"
            f"{state_block}"
            f"PATIENT'S QUESTION:\n{raw_query}"
            f"{vision_block}"
            f"MEDICAL REFERENCE TEXTS:\n\n{context_block}"
        )
        
        user_content = self._build_multimodal_content(text_content, image_path)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        answer = self._generate(
            messages, 
            max_new_tokens=768,
            temperature=0.2,          # Keep reasoning tight and grounded
            #top_p=0.85,               # Clamp vocabulary to highly probable clinical tokens
            do_sample=False,           # Greedy decoding for deterministic, grounded answers
            repetition_penalty=1.15,  # Prevent looping in thinking trace
        )
        
        # Guard: if _generate returned empty (thought-only model output with no
        # recoverable answer text), ask the user to rephrase rather than silently
        # returning a misleading "outside knowledge base" rejection.
        if not answer or not answer.strip():
            return (
                "I wasn't able to generate a clear response for this question. "
                "Could you please rephrase or add more detail about your symptoms?"
            )

        answer = self._apply_anatomy_guardrails_to_answer(
            raw_query=raw_query,
            draft_answer=answer,
            context_block=context_block,
            anatomy_facts=anatomy_facts,
        )
        
        return answer

    def apply_abstention_disclaimer(self, answer: str, grounding_result: dict) -> str:
        """Prepend a safety disclaimer if the answer is not fully grounded.
        
        Called after grounding verification. For PARTIAL or FAIL verdicts,
        warns the user that the answer may not be fully supported by sources.
        This is safer than confidently delivering potentially wrong advice.
        """
        verdict = grounding_result.get("verdict", "PASS")
        
        if verdict == "PASS":
            return answer
        
        contradicted = grounding_result.get("contradicted_count", 0)
        unverified = grounding_result.get("unverified_count", 0)
        
        if verdict == "FAIL" and contradicted > 0:
            disclaimer = (
                "⚠️ **Important Notice:** Some statements in this response may conflict with "
                "the medical reference texts. Please treat this as preliminary guidance only "
                "and **consult your ophthalmologist** for accurate medical advice.\n\n"
            )
        elif verdict == "FAIL":
            disclaimer = (
                "⚠️ **Notice:** I could not fully verify this response against my medical "
                "reference texts. Please treat this as general guidance and **confirm with "
                "your eye care professional**.\n\n"
            )
        else:  # PARTIAL
            disclaimer = (
                "ℹ️ *Note: Some parts of this response could not be fully verified against "
                "the medical reference texts. Please confirm with your eye doctor.*\n\n"
            )
        
        return disclaimer + answer

    # ── Entity Extraction with EyeCLIP Integration ───────────────────────────
    def extract_entities_from_answer(
        self,
        answer: str,
        visual_findings: Optional[str] = None,
        turn_id: int = 0,
    ) -> List[ClinicalEntity]:
        """Extract clinical entities from generated answer + EyeCLIP findings."""
        return self.entity_extractor.extract_entities(
            text=answer,
            visual_findings=visual_findings,
            turn_id=turn_id,
        )

    def extract_entities_from_turn(
        self,
        query_text: str,
        answer_text: str,
        visual_findings: Optional[str] = None,
        turn_id: int = 0,
    ) -> List[ClinicalEntity]:
        """
        Extract entities for session updates with source-aware safeguards.

        Symptoms/findings/imaging/conditions are treated as patient-side signals
        and are therefore sourced only from the raw user query and EyeCLIP
        inferences.
        """
        patient_entities = self.entity_extractor.extract_entities(
            text=query_text,
            visual_findings=visual_findings,
            turn_id=turn_id,
            source="user_query",
        )
        answer_entities = self.entity_extractor.extract_entities(
            text=answer_text,
            visual_findings=None,
            turn_id=turn_id,
            source="answer",
        )

        patient_owned_types = {
            EntityType.SYMPTOM,
            EntityType.FINDING,
            EntityType.IMAGING,
            EntityType.CONDITION,
        }
        answer_entities = [
            entity for entity in answer_entities
            if entity.entity_type not in patient_owned_types
        ]

        merged: Dict[tuple, ClinicalEntity] = {}
        for entity in patient_entities + answer_entities:
            key = ((entity.normalized or entity.text.lower()), entity.entity_type)
            prev = merged.get(key)
            if prev is None or entity.confidence > prev.confidence:
                merged[key] = entity
            elif prev is not None:
                prev.confidence = max(prev.confidence, entity.confidence)

        return list(merged.values())

    # ── Grounding Verification with Anatomy Check ────────────────────────────
    def verify_grounding(
        self, 
        answer: str, 
        context_block: str, 
        query_anatomy: Optional[set] = None,
        verbose: bool = True,
        method: str = "nli"
    ) -> dict:
        """Verify factual claims in the answer against context using the specified method."""
        if method == "generative":
            return self._verify_grounding_generative(answer, context_block, query_anatomy, verbose)
        return self._verify_grounding_nli(answer, context_block, query_anatomy, verbose)

    def _verify_grounding_nli(
        self, 
        answer: str, 
        context_block: str, 
        query_anatomy: Optional[set] = None,
        verbose: bool = True
    ) -> dict:
        import re
        from src.evaluator import get_evaluator
        
        # Fast exit: if the model is just refusing to answer, it needs no grounding
        refusal_patterns = [
            "outside my ophthalmology knowledge",
            "outside my knowledge base",
            "cannot answer this",
            "i am sorry",
            "i'm sorry",
        ]
        
        # If it's a short refusal, skip grounding logic and just PASS
        if len(answer) < 150 and any(p in answer.lower() for p in refusal_patterns):
            if verbose:
                print("[Grounding NLI] Answer is a standard refusal. Bypassing grounding.")
            return {
                "verdict": "PASS",
                "flagged_claims": [],
                "anatomy_mismatch": "Unknown",
                "reasoning": "Standard refusal/apology bypass."
            }
        
        # Fast sentence splitting to extract claims
        # Ignore short introductory/hedging phrases or common boilerplate
        raw_claims = []
        for line in answer.split('\n'):
            line = line.strip()
            if not line or line.endswith(':'):
                continue  # Skip empty lines and headings like "Possible Causes:"
            raw_claims.append(line)
            
        potential_claims = [c.strip() for c in raw_claims if len(c.strip()) > 15]
        
        # Only skip purely advisory/referral sentences and automated screening disclaimers.
        boilerplate_skip_keywords = {
            "consult your doctor", "see a specialist", "seek professional",
            "healthcare provider", "seek medical attention",
            "please visit", "go to the emergency",
            "automated screening findings", "preliminary findings",
            "only a professional eye", "evaluation by a qualified",
            "perform a comprehensive eye exam", "healthy lifestyle choices",
        }
        
        claims = []
        for c in potential_claims:
            low_c = c.lower()
            
            # Skip advisory sentences and standard VLM disclaimers (no length limit for VLM disclaimers)
            if any(kw in low_c for kw in boilerplate_skip_keywords):
                continue
                
            # Strip common introductory phrases that confuse the strict NLI model
            prefixes_to_strip = [
                "based on the provided medical reference texts, ",
                "based on the medical reference texts, ",
                "according to the texts, ",
                "the oct images show "
            ]
            clean_c = low_c
            for p in prefixes_to_strip:
                if clean_c.startswith(p):
                    clean_c = clean_c[len(p):].strip()
                    c = c[len(p):].strip()
            
            claims.append(c)
        
        # NOTE: Anatomy meta-claims are intentionally NOT injected into the NLI
        # claim list. They are system assertions, not answer claims. Injecting them
        # causes false NLI failures that cascade into the self-correction prompt,
        # resulting in instruction leakage ("DO NOT repeat these...") appearing
        # verbatim in the patient-facing output. Anatomy consistency is enforced
        # separately via _apply_anatomy_guardrails_to_answer().

        try:
            evaluator = get_evaluator()
            result = evaluator.verify_grounding(claims, context_block)
            
            verdict = result["verdict"]
            unsupported = result.get("unsupported_claims", [])
            flagged = [c["claim"] for c in unsupported]
            
            if verbose:
                print(f"[Grounding NLI] Verdict: {verdict}")
                for claim in flagged:
                    print(f"  ⚠️  Flagged: {claim}")
            
            return {
                "verdict": verdict,
                "flagged_claims": flagged,
                "anatomy_mismatch": "Unknown",
                "reasoning": result["reasoning"]
            }
        except Exception as e:
            if verbose:
                print(f"[Grounding NLI] Error: {e}")
            return {
                "verdict": "ERROR",
                "flagged_claims": [],
                "anatomy_mismatch": "Unknown",
                "reasoning": str(e)
            }

    def _verify_grounding_generative(
        self, 
        answer: str, 
        context_block: str, 
        query_anatomy: Optional[set] = None,
        verbose: bool = True
    ) -> dict:
        """Verify factual claims in the answer against context + check anatomy using MedGemma."""
        system_prompt = (
            "You are a medical fact-checker with anatomical expertise. You will receive:\n"
            "  1. A PATIENT ANSWER\n"
            "  2. The SOURCE TEXTS used\n"
            "  3. The QUERY ANATOMY (which eye structure the question is about)\n\n"
            "Your job:\n"
            "1. Verify EVERY factual claim is supported by source texts.\n"
            "2. Verify the answer discusses the same anatomy as the query.\n"
            "3. Flag any claims that:\n"
            "   - Are not in the sources\n"
            "   - Discuss a different anatomical structure than the query\n"
            "   - Invent source citations not present in context\n\n"
            "OUTPUT FORMAT (strictly follow):\n"
            "VERDICT: PASS or FAIL\n"
            "FLAGGED CLAIMS:\n"
            "- [claim 1]\n"
            "ANATOMY MISMATCH: [Yes/No + brief explanation]\n"
            "REASONING: [1-2 sentences]"
        )

        anatomy_note = f"\n\nQUERY ANATOMY: {query_anatomy}" if query_anatomy else ""
        user_message = f"PATIENT ANSWER:\n{answer}\n\nSOURCE TEXTS:\n{context_block}{anatomy_note}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self._generate(
            messages,
            max_new_tokens=256,
            temperature=0.1,
            repetition_penalty=1.3,
        )

        return self._parse_grounding_response(response, verbose)

    def _parse_grounding_response(self, response: str, verbose: bool) -> dict:
        verdict = "PASS"
        flagged_claims = []
        anatomy_mismatch = "No"
        reasoning = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                verdict = "FAIL" if "FAIL" in v else "PASS"
            elif line.startswith("- ") and verdict == "FAIL":
                claim = line[2:].strip()
                if claim.lower() != "none" and len(claim) > 5:
                    flagged_claims.append(claim)
            elif line.upper().startswith("ANATOMY MISMATCH:"):
                anatomy_mismatch = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        if flagged_claims and verdict == "PASS":
            verdict = "FAIL"

        if verbose:
            print(f"[Grounding Gen] Verdict: {verdict}")
            for claim in flagged_claims:
                print(f"  ⚠️  Flagged: {claim}")
            print(f"[Grounding Gen] Anatomy mismatch: {anatomy_mismatch}")
            print(f"[Grounding Gen] Reasoning: {reasoning}")

        return {
            "verdict": verdict,
            "flagged_claims": flagged_claims,
            "anatomy_mismatch": anatomy_mismatch,
            "reasoning": reasoning,
        }

    # ── Modality Detection (unchanged from original) ─────────────────────────
    VALID_MODALITIES = {
        "OCT", "CFP", "FFA", "ICGA", "FAF", "OUS", "RetCam",
        "slit lamp", "specular", "corneal photography", "external",
    }

    def detect_modality_vlm(self, image_path: str, use_vlm_fallback: bool = True) -> str:
        """
        Detect ophthalmic imaging modality from an image.

        Strategy:
          1. **Primary**: Fast pixel-heuristic based on RGB statistics
             (saturation, brightness, color dominance). Reliable for clearly
             differentiated modalities (OCT, CFP, FFA, slit lamp).
          2. **Fallback** (optional): If the heuristic lands in an ambiguous
             zone, queries MedGemma VLM with a tightly constrained prompt.
             Controlled by `use_vlm_fallback` (default True).

        Returns a modality prefix matching EyeCLIP label prefixes:
          'OCT', 'CFP', 'FFA', 'ICGA', 'FAF', 'OUS', 'RetCam',
          'slit lamp', 'specular', 'corneal photography', or 'external'
        Falls back to 'CFP' on failure.
        """
        import os
        if not image_path or not os.path.exists(image_path):
            return "CFP"

        import numpy as np
        from PIL import Image

        try:
            pil_image = Image.open(image_path).convert("RGB")
            img_rgb = np.array(pil_image.resize((64, 64))).astype(float) / 255.0
            r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

            cmax = img_rgb.max(axis=2)
            cmin = img_rgb.min(axis=2)
            sat = np.where(cmax > 0.05, (cmax - cmin) / (cmax + 1e-6), 0)
            mean_sat = sat.mean()
            mean_bright = img_rgb.mean()
            bright_frac = (img_rgb.mean(axis=2) > 0.7).mean()
            dark_frac   = (img_rgb.mean(axis=2) < 0.1).mean()

            ambiguous = False  # Flag to trigger VLM fallback

            # 1. Grayscale? (OCT / FAF / OUS)
            if mean_sat < 0.15:
                if mean_bright > 0.35:
                    modality = "OCT"
                elif mean_bright < 0.15:
                    modality = "FAF"
                elif dark_frac > 0.30:
                    modality = "OCT"
                else:
                    modality = "OUS"
            # 2. External / Slit Lamp (broad color, high saturation)
            elif (g > 0.25).mean() > 0.35 and mean_sat > 0.25:
                modality = "slit lamp"
            # 3. FFA (B&W high contrast)
            elif mean_sat < 0.20 and bright_frac > 0.05:
                modality = "FFA"
            # 4. CFP (orange/red dominant, moderate saturation)
            elif sat[sat > 0.15].shape[0] > 50 and ((r[sat > 0.15] > g[sat > 0.15] * 1.1) & (r[sat > 0.15] > b[sat > 0.15] * 1.2)).mean() > 0.45:
                modality = "CFP"
            else:
                modality = "CFP"
                ambiguous = True  # Heuristic wasn't confident — try VLM

            print(f"[Modality] Pixel heuristic → {modality}" + (" (ambiguous)" if ambiguous else ""))

            # ── Optional VLM fallback for ambiguous cases ─────────────────────
            if ambiguous and use_vlm_fallback:
                vlm_modality = self._detect_modality_with_vlm(image_path)
                if vlm_modality:
                    print(f"[Modality] VLM override → {vlm_modality}")
                    modality = vlm_modality

            return modality

        except Exception as e:
            print(f"[Modality] ⚠ Detection failed: {e}. Defaulting to CFP.")
            return "CFP"

    def _detect_modality_with_vlm(self, image_path: str) -> str | None:
        """
        Use MedGemma VLM to classify the ophthalmic imaging modality.
        Tightly constrainted single-token output to minimize hallucination.
        Returns a valid modality string, or None on failure.
        """
        try:
            from PIL import Image as PILImage
            pil_image = PILImage.open(image_path).convert("RGB")

            messages = [
                {"role": "system", "content": (
                    "You are an ophthalmic imaging modality classifier.\n"
                    "Given an eye image, identify EXACTLY which imaging modality produced it.\n\n"
                    "VALID MODALITIES (output ONLY one of these, exactly as written):\n"
                    "OCT\nCFP\nFFA\nICGA\nFAF\nOUS\nRetCam\n"
                    "slit lamp\nspecular\ncorneal photography\nexternal\n\n"
                    "OUTPUT RULES:\n"
                    "- Output ONLY the modality name. Nothing else.\n"
                    "- No explanations, no sentences, no punctuation.\n"
                    "- If uncertain, output: CFP"
                )},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "What imaging modality is this?"},
                ]},
            ]

            result = self._generate(
                messages,
                max_new_tokens=10,
                temperature=0.05,
                do_sample=False,
                repetition_penalty=1.0,
            ).strip()

            # Validate: result must be an exact match to a valid modality
            for valid in self.VALID_MODALITIES:
                if result.lower() == valid.lower():
                    return valid

            # Partial match (e.g., "slit lamp photography" → "slit lamp")
            for valid in self.VALID_MODALITIES:
                if valid.lower() in result.lower():
                    return valid

            print(f"[Modality] VLM returned unrecognized modality: '{result}'")
            return None

        except Exception as e:
            print(f"[Modality] ⚠ VLM fallback failed: {e}")
            return None
