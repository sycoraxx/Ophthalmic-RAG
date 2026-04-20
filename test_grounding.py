import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import RetinaRetriever
import torch
import re

class GroundingTester:
    def __init__(self):
        # We only need the retriever to get the reranker_model and reranker_tokenizer
        self.retriever = RetinaRetriever()

    def _verify_grounding_medcpt(self, answer: str, context_block: str, verbose: bool = True) -> dict:
        # Copied from src/engine.py for isolated testing without loading Gemma/Vision
        import torch

        # ── Quick exits ──────────────────────────────────────────────────────
        refusal_patterns = [
            "outside my ophthalmology knowledge",
            "outside my knowledge base",
            "cannot answer this",
            "i am sorry",
            "i'm sorry",
            "couldn't find relevant information",
        ]
        if len(answer) < 150 and any(p in answer.lower() for p in refusal_patterns):
            if verbose:
                print("[Grounding MedCPT] Standard refusal — bypass.")
            return {
                "verdict": "PASS",
                "flagged_claims": [],
                "unsupported_claims": [],
                "reasoning": "Standard refusal bypass.",
            }

        if not context_block.strip():
            return {
                "verdict": "FAIL",
                "flagged_claims": [],
                "unsupported_claims": [],
                "reasoning": "Empty context.",
            }

        # ── Sentence-tokenize the answer into atomic claims ──────────────────
        atomic_claims: list[str] = []
        seen: set[str] = set()

        boilerplate_skip = {
            "consult your doctor", "see a specialist", "seek professional",
            "healthcare provider", "seek medical attention", "please visit",
            "go to the emergency", "automated screening", "preliminary findings",
            "only a professional", "evaluation by a qualified",
            "perform a comprehensive eye exam", "healthy lifestyle",
            "i am an ai", "i'm an ai", "disclaimer",
        }

        for line in answer.split("\n"):
            line = line.strip()
            if not line or line.endswith(":"):
                continue
            low_line = line.lower()
            if any(kw in low_line for kw in boilerplate_skip):
                continue

            sentences = re.split(r'(?<=[.!?])\s+', line)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 25:
                    continue
                clean = re.sub(r'^[-•*]\s*', '', sent)
                if len(clean.split()) <= 4:
                    continue
                key = clean.lower()
                if key not in seen:
                    seen.add(key)
                    atomic_claims.append(clean)

        if not atomic_claims:
            if verbose:
                print("[Grounding MedCPT] No verifiable claims extracted → PASS")
            return {
                "verdict": "PASS",
                "flagged_claims": [],
                "unsupported_claims": [],
                "reasoning": "No verifiable claims extracted.",
            }

        # ── Chunk context into overlapping windows ───────────────────────────
        tokenizer = self.retriever.reranker_tokenizer
        MAX_CTX_TOKENS = 400
        OVERLAP = 80

        ctx_ids = tokenizer.encode(context_block, add_special_tokens=False)

        if len(ctx_ids) <= MAX_CTX_TOKENS:
            context_windows = [context_block]
        else:
            context_windows = []
            start = 0
            while start < len(ctx_ids):
                end = min(start + MAX_CTX_TOKENS, len(ctx_ids))
                window_text = tokenizer.decode(ctx_ids[start:end], skip_special_tokens=True)
                context_windows.append(window_text)
                if end >= len(ctx_ids):
                    break
                start += MAX_CTX_TOKENS - OVERLAP

        # ── Score each claim against all windows ─────────────────────────────
        SUPPORT_THRESHOLD = 0.35
        LOW_THRESHOLD = 0.10

        supported_claims: list[dict] = []
        unsupported_claims: list[dict] = []
        weak_claims: list[dict] = []

        model = self.retriever.reranker_model
        device = self.retriever.reranker_device

        for claim in atomic_claims:
            pairs = [[claim, window] for window in context_windows]

            with torch.no_grad():
                encoded = tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(device)
                logits = model(**encoded).logits

                if logits.dim() == 1:
                    relevance = logits
                elif logits.size(-1) == 1:
                    relevance = logits.squeeze(dim=-1)
                else:
                    relevance = torch.softmax(logits, dim=-1)[..., -1]

                if relevance.dim() == 0:
                    relevance = relevance.unsqueeze(0)

                best_score = relevance.max().item()

            record = {
                "claim": claim,
                "relevance_score": round(best_score, 4),
            }

            if best_score >= SUPPORT_THRESHOLD:
                record["status"] = "supported"
                supported_claims.append(record)
            elif best_score <= LOW_THRESHOLD:
                record["status"] = "unsupported"
                unsupported_claims.append(record)
            else:
                record["status"] = "weak"
                weak_claims.append(record)

        total = len(supported_claims) + len(unsupported_claims) + len(weak_claims)
        unsupported_ratio = len(unsupported_claims) / max(total, 1)

        if unsupported_ratio > 0.5:
            verdict = "FAIL"
            reasoning = f"Majority of claims unsupported: {len(unsupported_claims)}/{total} below relevance threshold."
        elif unsupported_claims or weak_claims:
            verdict = "PARTIAL"
            reasoning = f"{len(supported_claims)}/{total} claims supported, {len(weak_claims)} weakly matched, {len(unsupported_claims)} unsupported."
        else:
            verdict = "PASS"
            reasoning = f"All {len(supported_claims)} claims supported by context."

        if verbose:
            print(f"[Grounding MedCPT] Verdict: {verdict}")
            print(f"  Supported: {len(supported_claims)}, Weak: {len(weak_claims)}, Unsupported: {len(unsupported_claims)} / {total} total claims")

        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "flagged_claims": [r["claim"] for r in unsupported_claims],
            "unsupported_claims": unsupported_claims,
            "supported_claims": supported_claims,
            "weak_claims": weak_claims
        }

if __name__ == "__main__":
    print("Initializing GroundingTester...")
    tester = GroundingTester()

    context_block = """
Glaucoma is a group of eye conditions that damage the optic nerve. 
This damage is often caused by an abnormally high pressure in your eye. 
It is one of the leading causes of blindness for people over the age of 60.
Treatment options include prescription eye drops, oral medicines, laser treatment, surgery or a combination of any of these.
"""

    print("\n--- Test 1: Fully Supported Claim ---")
    supported_answer = "Glaucoma damages the optic nerve and is often linked to high eye pressure. Treatments involve eye drops, medications, or surgery."
    res1 = tester._verify_grounding_medcpt(supported_answer, context_block, verbose=True)
    print(f"Result: {res1['verdict']}")
    print(f"Supported claims: {res1['supported_claims']}")
    print(f"Unsupported claims: {res1['unsupported_claims']}")

    print("\n--- Test 2: Unsupported Claim ---")
    unsupported_answer = "Glaucoma is completely curable by eating carrots. It primarily affects teenagers."
    res2 = tester._verify_grounding_medcpt(unsupported_answer, context_block, verbose=True)
    print(f"Result: {res2['verdict']}")
    print(f"Supported claims: {res2['supported_claims']}")
    print(f"Unsupported claims: {res2['unsupported_claims']}")

    print("\n--- Test 3: Partially Supported Claim ---")
    partial_answer = "Glaucoma damages the optic nerve. Also, drinking coffee cures it instantly."
    res3 = tester._verify_grounding_medcpt(partial_answer, context_block, verbose=True)
    print(f"Result: {res3['verdict']}")
    print(f"Supported claims: {res3['supported_claims']}")
    print(f"Unsupported claims: {res3['unsupported_claims']}")
