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

MEDGEMMA_MODEL = "./models/checkpoints/medgemma-1.5-4b-it"


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
        
        # 4. Emergency Fallback: If stripping made it empty, the model might have 
        # only output the thought/checklist. Return at least something or the original
        if not out:
            # If it's empty, maybe the model didn't generate any answer yet. 
            # We'll return the raw_out but without the thought prefix if possible.
            out = re.sub(r'(?i)^thought:?\s*', '', raw_out).strip()
            if not out:
                return "I'm sorry, I couldn't generate a clear response for this query. Could you please rephrase?"
                    
        return out

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
                valid_conds = [c for c in parsed["conditions"] if c["score"] >= 10.0]
                conds = [f"{c['name']} ({c['score']}%)" for c in valid_conds]
                cond_str = ", ".join(conds) if conds else "Normal/Unremarkable"
                modality = parsed.get("modality") or "Unknown"
                eyeclip_hint = f" [Image Findings: Modality={modality}, Findings={cond_str}]"
            else:
                short_findings = visual_findings.split("----------------------------------------")[0].strip()
                eyeclip_hint = f" [Image Findings: {short_findings}]"

        user_text = f"Patient:{eyeclip_hint} {raw_query}"
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

        # The strict prompt already enforces keyword-only output. 
        # Just do a safe basic strip to avoid mangling valid medical hyphenations.
        refined = refined.replace('"', '').replace("'", "").strip()
        
        # Fallback to original if refinement is too short
        return refined if len(refined) >= 5 else raw_query

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

    # ── Context-Aware Query Rewriting for Follow-ups ─────────────────────────
    def rewrite_query_for_retrieval(
        self,
        current_query: str,
        session_state: Optional[ClinicalSessionState] = None,
        visual_findings: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """
        Rewrite a follow-up query into a standalone, retrieval-optimized query
        using pinned clinical context from session state.
        
        Example:
          Input: "What to do now?" + {condition: "drusen", anatomy: "retina"}
          Output: "drusen AMD management monitoring treatment options"
        """
        # Build context injection from session state
        state_context = session_state.to_query_context() if session_state else ""
        
        # Build EyeCLIP hint
        eyeclip_hint = ""
        if visual_findings:
            parsed = self._parse_eyeclip_findings(visual_findings)
            if parsed:
                probable = [c["name"] for c in parsed["conditions"] if c["confidence"] == "probable"]
                if probable:
                    eyeclip_hint = f" [EyeCLIP: {', '.join(probable)}]"
        
        system_prompt = (
            "You are a clinical query rewriter for ophthalmology RAG.\n"
            "Rewrite the patient's follow-up question into a STANDALONE search query "
            "that includes necessary context from prior conversation.\n\n"
            "RULES:\n"
            "- Output ONLY 5-10 clinical keywords. Nothing else.\n"
            "- If the query is ambiguous (e.g., 'What to do now?'), use the provided "
            "  clinical context to make it specific.\n"
            "- Include anatomy and condition terms from the context injection.\n"
            "- If EyeCLIP findings are provided, include detected conditions.\n"
            "- Output in ENGLISH only.\n\n"
            "EXAMPLES:\n"
            "Context: [anatomy:retina | condition:age-related macular degeneration]\n"
            "Query: What to do now?\n"
            "Output: AMD management monitoring AREDS treatment progression\n\n"
            "Context: [anatomy:retina | condition:diabetic retinopathy]\n"
            "Query: Is this serious?\n"
            "Output: diabetic retinopathy severity staging progression risk\n\n"
            "Context: [anatomy:lens | condition:cataract]\n"
            "Query: What are my options?\n"
            "Output: cataract surgery options IOL timing recovery"
        )
        
        user_text = f"Context:{state_context}{eyeclip_hint}\nPatient: {current_query}"
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
        
        # Post-process: extract clean search terms
        rewritten = self._clean_query_output(rewritten)
        
        # Fallback to current query if rewriting yielded nothing or too little
        return rewritten if len(rewritten) >= 5 else current_query

    # ── Context Formatting ───────────────────────────────────────────────────
    def build_context_block(self, context_docs: list[Document]) -> str:
        """Format retrieved documents into a context block."""
        parts = []
        for i, doc in enumerate(context_docs, 1):
            src = doc.metadata.get("source", "Unknown")
            path = doc.metadata.get("section_path", "General")
            # Trim content to save tokens
            content = doc.page_content[:800].strip()
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
            state_context = session_state.to_generation_context()
            if state_context:
                state_block = f"\nCLINICAL CONTEXT FROM PRIOR CONVERSATION:\n{state_context}\n\n"
        
        # Check anatomical consistency
        query_anatomy = self._detect_anatomy(raw_query)
        context_anatomy = set()
        for doc in context_docs:
            doc_anat = doc.metadata.get("anatomy", set())
            if isinstance(doc_anat, str):
                doc_anat = {doc_anat}
            context_anatomy.update(doc_anat)
        
        anatomy_mismatch = query_anatomy and context_anatomy and not query_anatomy.intersection(context_anatomy)
        
        base_rules = (
            "RULES:\n"
            "1. Use simple, everyday language. Explain medical terms clearly.\n"
            "2. Be warm and reassuring but honest. Do not make firm diagnoses.\n"
            "3. Structure: direct answer → possible causes → home care → when to see doctor.\n"
            "4. Cite sources as [Source N] ONLY if fact appears in that source. NEVER invent citations.\n"
            "5. If reference texts lack information, say so. DO NOT GUESS.\n"
            "6. Keep answer concise (150-250 words).\n"
            "7. If CLINICAL CONTEXT is provided, personalize advice accordingly.\n"
            "8. Maintain continuity with prior conversation topics.\n"
            "9. If VISUAL FINDINGS provided, weave into explanation if consistent with sources.\n"
            "10. If IMAGE attached, describe observable features only; do not diagnose.\n"
        )
        
        if anatomy_mismatch:
            base_rules += (
                "11. ANATOMICAL MISMATCH DETECTED: The retrieved sources discuss a different eye structure "
                "than the patient's question. Acknowledge this limitation and suggest consulting a specialist.\n"
            )
        
        if target_language != "English":
            base_rules += (
                f"12. TRANSLATION REQUIRED: Write ENTIRE response in {target_language}.\n"
            )
        
        if correction_context:
            system_prompt = (
                "You are a friendly eye health assistant.\n\n"
                f"{patient_block}{base_rules}\n"
                "CRITICAL CORRECTION: Previous answer contained unsupported claims:\n"
                f"{correction_context}\n"
                "DO NOT repeat these. ONLY state facts from source texts.\n"
            )
        else:
            system_prompt = (
                "You are a friendly eye health assistant.\n\n"
                f"{patient_block}{base_rules}"
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
            top_p=0.85,               # Clamp vocabulary to highly probable clinical tokens
            do_sample=True,           # Required for temp/top_p to work
            repetition_penalty=1.15,  # Prevent looping in thinking trace
        )
        
        if "<unused94>thought" in answer or "I cannot answer" in answer.lower():
            return "I'm sorry, but this question is outside my ophthalmology knowledge base."
        
        return answer

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

    # ── Grounding Verification with Anatomy Check ────────────────────────────
    def verify_grounding(
        self, 
        answer: str, 
        context_block: str, 
        query_anatomy: Optional[set] = None,
        verbose: bool = True
    ) -> dict:
        """Verify factual claims in the answer against context + check anatomy."""
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
            print(f"[Grounding] Verdict: {verdict}")
            for claim in flagged_claims:
                print(f"  ⚠️  Flagged: {claim}")
            print(f"[Grounding] Anatomy mismatch: {anatomy_mismatch}")
            print(f"[Grounding] Reasoning: {reasoning}")

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
