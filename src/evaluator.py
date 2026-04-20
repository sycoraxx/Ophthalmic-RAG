"""
evaluator.py — Fast, specialized evaluation models
────────────────────────────────────────────────────
Implements `LightweightEvaluator` which uses specialized, small models
instead of a bulky generative LLM for evaluation tasks:

1. NLI Grounding: uses a Cross-Encoder (e.g., DeBERTa-v3 NLI) to detect
   entailment vs contradiction (hallucination) between claims and context.
2. Zero-Shot Extraction: uses a fast zero-shot classifier (e.g., BART-Large-MNLI)
   to extract the chosen MCQ option from a verbose answer.
"""

from __future__ import annotations
import gc
from typing import List, Dict, Any, Optional

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


class LightweightEvaluator:
    def __init__(
        self,
        nli_model_id: str = "cross-encoder/nli-deberta-v3-small",
        mcq_extractor_id: str = "facebook/bart-large-mnli",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the specialized evaluation models.
        These are small enough (~1GB combined) to stay resident on GPU alongside MedGemma.
        """
        self.device = device
        self.nli_model_id = nli_model_id
        self.mcq_extractor_id = mcq_extractor_id
        
        # Pipelines
        self._nli_pipe = None
        self._zs_pipe = None

    def _load_nli(self):
        if self._nli_pipe is None:
            print(f"[LightweightEvaluator] Loading NLI model: {self.nli_model_id}")
            self._nli_pipe = pipeline(
                "text-classification",
                model=self.nli_model_id,
                device=0 if self.device == "cuda" else -1
            )
        return self._nli_pipe

    def _load_zs(self):
        if self._zs_pipe is None:
            print(f"[LightweightEvaluator] Loading Zero-Shot model: {self.mcq_extractor_id}")
            self._zs_pipe = pipeline(
                "zero-shot-classification",
                model=self.mcq_extractor_id,
                device=0 if self.device == "cuda" else -1
            )
        return self._zs_pipe

    def extract_mcq_choice(self, generated_answer: str, options: List[str]) -> Optional[int]:
        """
        Extract which MCQ option the verbose answer is selecting.
        Uses Zero-Shot classification: "This text selects which option?"
        """
        if not options or not generated_answer.strip():
            return None

        # Clean up options to just the text content for semantic matching
        # e.g., "A. Glaucoma" -> "Glaucoma"
        clean_opts = []
        for opt in options:
            if opt.startswith(("A.", "B.", "C.", "D.", "E.")):
                clean_opts.append(opt[3:].strip())
            else:
                clean_opts.append(opt.strip())

        zs_pipe = self._load_zs()
        try:
            res = zs_pipe(
                generated_answer,
                candidate_labels=clean_opts,
                hypothesis_template="The diagnosis or answer selected is {}.",
            )
            top_label = res["labels"][0]
            top_score = res["scores"][0]
            
            # If the model is confident enough, return that index
            if top_score > 0.4:
                return clean_opts.index(top_label)
            return None
        except Exception as e:
            print(f"[LightweightEvaluator] Error in MCQ extraction: {e}")
            return None

    def verify_grounding(self, claims: List[str], context: str) -> Dict[str, Any]:
        """
        Verify if a list of clinical claims are entailed by the context text.
        
        Claims are first sentence-tokenised so the NLI model receives atomic
        statements rather than full paragraphs (which it cannot entail reliably).

        Uses a sliding-window approach to handle contexts that exceed
        DeBERTa-v3-small's 512-token limit. Each claim is tested against
        all context windows; a claim is entailed if ANY window entails it.

        Verdict tiers:
          PASS    — all claims entailed by at least one context window
          PARTIAL — some claims neutral (unverified) but none contradicted
          FAIL    — any claim contradicted, OR >66% of claims unverified
        """
        if not claims:
            return {"verdict": "PASS", "reasoning": "No claims provided to verify.", "unsupported_claims": []}
        
        if not context.strip():
            return {"verdict": "FAIL", "reasoning": "Context is empty.", "unsupported_claims": claims}

        nli_pipe = self._load_nli()
        tokenizer = nli_pipe.tokenizer

        # ── Sentence-tokenise claims ────────────────────────────────────────
        # The generator produces paragraph-length lines. DeBERTa-v3-small is
        # trained on sentence-pair NLI; it cannot reliably entail full
        # paragraphs. Split each line into individual sentences first.
        import re as _re
        atomic_claims: List[str] = []
        seen_claim_text: set = set()
        for claim in claims:
            # Split on sentence boundaries: ., !, ? followed by whitespace or end
            sentences = _re.split(r'(?<=[.!?])\s+', claim.strip())
            for sent in sentences:
                sent = sent.strip()
                # Skip bullets/list markers passed as claims, very short fragments,
                # and exact duplicates.
                if len(sent) < 20:
                    continue
                if sent.lower() in seen_claim_text:
                    continue
                # Skip pure bullet items ("- item", "* item") with no verb
                if _re.match(r'^[-•*]\s*\w', sent) and len(sent.split()) <= 6:
                    continue
                seen_claim_text.add(sent.lower())
                atomic_claims.append(sent)

        if not atomic_claims:
            return {"verdict": "PASS", "reasoning": "No verifiable atomic claims extracted.", "unsupported_claims": []}
        
        # ── Chunk context into overlapping windows ──────────────────────────
        # Reserve ~256 tokens for the claim + special tokens; use rest for context
        MAX_SEQ = 512
        CLAIM_BUDGET = 128  # single sentences are short; give more room to context
        CTX_BUDGET = MAX_SEQ - CLAIM_BUDGET  # ~384 tokens for context per window
        OVERLAP = 64  # token overlap between consecutive windows
        
        ctx_ids = tokenizer.encode(context, add_special_tokens=False, verbose=False)
        
        if len(ctx_ids) <= CTX_BUDGET:
            # Context fits in a single window — fast path (no windowing needed)
            context_windows = [context]
        else:
            # Build overlapping token windows and decode back to text
            context_windows = []
            start = 0
            while start < len(ctx_ids):
                end = min(start + CTX_BUDGET, len(ctx_ids))
                window_text = tokenizer.decode(ctx_ids[start:end], skip_special_tokens=True)
                context_windows.append(window_text)
                if end >= len(ctx_ids):
                    break
                start += CTX_BUDGET - OVERLAP  # slide forward with overlap

        # ── Check each claim against all windows ─────────────────────────────
        # Three categories per claim: entailed, contradicted, neutral (unverified)
        entailed_claims = []
        contradicted_claims = []
        unverified_claims = []
        ENTAILMENT_THRESHOLD = 0.45  # minimum score to count as entailed
        
        for claim in atomic_claims:
            # Build NLI pairs for every window
            pairs = [{"text": window, "text_pair": claim} for window in context_windows]
            
            try:
                all_results = nli_pipe(pairs, top_k=None)
            except Exception as e:
                print(f"[LightweightEvaluator] NLI error for claim: {e}")
                continue
            
            claim_status = "neutral"  # default
            best_entailment_score = 0.0
            best_contradiction_score = 0.0
            
            for res_list in all_results:
                for pred in res_list:
                    label = pred["label"].lower()
                    score = pred["score"]
                    
                    if "entailment" in label and score > best_entailment_score:
                        best_entailment_score = score
                    if "contradiction" in label and score > best_contradiction_score:
                        best_contradiction_score = score
                
                top_label = res_list[0]["label"].lower()
                # Require a meaningful entailment score, not just "top label"
                if "entailment" in top_label and best_entailment_score >= ENTAILMENT_THRESHOLD:
                    claim_status = "entailed"
                    break  # One window entailing is enough
            
            # If not entailed by any window, check if any window contradicts it
            if claim_status != "entailed" and best_contradiction_score > 0.55:
                claim_status = "contradicted"
            
            claim_record = {
                "claim": claim,
                "status": claim_status,
                "entailment_score": round(best_entailment_score, 3),
                "contradiction_score": round(best_contradiction_score, 3),
            }
            
            if claim_status == "entailed":
                entailed_claims.append(claim_record)
            elif claim_status == "contradicted":
                contradicted_claims.append(claim_record)
            else:
                unverified_claims.append(claim_record)

        # ── Determine verdict ────────────────────────────────────────────────
        total_claims = len(entailed_claims) + len(contradicted_claims) + len(unverified_claims)
        
        if contradicted_claims:
            # Any contradiction → FAIL
            verdict = "FAIL"
            reasoning = (
                f"{len(contradicted_claims)} claim(s) contradicted by sources, "
                f"{len(unverified_claims)} unverified, {len(entailed_claims)} entailed "
                f"({len(context_windows)} context windows, {len(atomic_claims)} atomic claims checked)."
            )
        elif unverified_claims:
            # No contradictions, but some claims not in sources
            unverified_ratio = len(unverified_claims) / max(total_claims, 1)
            if unverified_ratio > 0.66:
                # Super-majority of claims not supported → FAIL
                verdict = "FAIL"
                reasoning = (
                    f"Super-majority of claims ({len(unverified_claims)}/{total_claims}) not supported by sources "
                    f"({len(context_windows)} context windows, {len(atomic_claims)} atomic claims checked)."
                )
            else:
                # Some unverified but most are entailed → PARTIAL
                verdict = "PARTIAL"
                reasoning = (
                    f"{len(unverified_claims)} claim(s) unverified (not in sources), "
                    f"{len(entailed_claims)} entailed "
                    f"({len(context_windows)} context windows, {len(atomic_claims)} atomic claims checked)."
                )
        else:
            # All claims entailed
            verdict = "PASS"
            reasoning = f"All {len(entailed_claims)} atomic claims entailed ({len(context_windows)} context windows checked)."

        # Build unsupported list (contradicted + unverified) for downstream use
        unsupported = contradicted_claims + unverified_claims

        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "unsupported_claims": unsupported,
            "entailed_count": len(entailed_claims),
            "contradicted_count": len(contradicted_claims),
            "unverified_count": len(unverified_claims),
        }

    def clear_memory(self):
        """Free up GPU memory if used intermittently."""
        self._nli_pipe = None
        self._zs_pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

# Global singleton so we don't reload weights repeatedly during evaluation loops
_global_evaluator = None

def get_evaluator() -> LightweightEvaluator:
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = LightweightEvaluator()
    return _global_evaluator
