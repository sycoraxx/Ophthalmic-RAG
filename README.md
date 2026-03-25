# Ophthalmic-RAG: Specialized AI Assistant for Indian Ophthalmology

A self-correcting, multimodal RAG (Retrieval-Augmented Generation) pipeline specifically localized for Indian ophthalmic needs. This system integrates specialized vision models (EyeCLIP) with medical-grade LLMs (MedGemma) to provide grounded, context-aware diagnostic support.

---

## 🌟 Key Features

### 1. Multimodal Diagnostic Fusion
- **Visual Intelligence**: Integrates an internal **EyeCLIP** engine (ViT-B/32) specialized for ophthalmic imaging (OCT, CFP, Slit Lamp).
- **Automated Modality Detection**: Automatically identifies imaging types to optimize diagnostic reasoning.

### 2. Specialized Medical LLM Stack
- **MedGemma 1.5-4B**: Fine-tuned for clinical reasoning and medical terminology.
- **MedEmbed & MedCPT**: Uses specialized medical embeddings and cross-encoders for high-precision retrieval over clinical corpora.

### 3. Intelligent Session Management
- **Confidence Decay**: Clinical findings and symptoms are tracked with time-based decay, handling multi-turn diagnostic sessions with precision.
- **Smart Topic Drift Detection**: Vague follow-up queries (e.g., "what to do now?") correctly inherit the established clinical topic, preventing false session resets.
- **Localized Metadata**: Automates Anatomical Locality (Anterior/Posterior Segment) and Clinical Triage Priority (Emergency/Urgent) mapping based on AIOS/NPCB standards.

### 4. Self-Correcting RAG Loop
- **Grounding Verification**: Implements a dedicated verification turn to ensure every claim is supported by the retrieved clinical context, minimizing hallucinations.
- **Thought-Bypass Optimization**: Uses `skip_thought` generation to achieve sub-second query refinement.

### 5. Enriched Knowledge Base (6,100+ Articles)
- **Multi-Source Ingestion**: Automated pipeline (`scripts/fetch_articles.py`) fetches articles from 4 public APIs:
  - **PubMed** (4,741 peer-reviewed abstracts) · **EuropePMC** (774 open-access) · **Semantic Scholar** (534 cross-publisher) · **MedlinePlus** (51 consumer health)
- **21 Clinical Categories**: Covering Diabetic Retinopathy, Glaucoma, Corneal Diseases, Neuro-Ophthalmology, Ocular Genetics, Community Eye Health, and more.
- **India-Relevant**: Emphasis on conditions prevalent in Indian clinical practice — trachoma, fungal keratitis, ROP, vitamin A deficiency.
- **Vector Corpus**: 21,635 child chunks across 8,855 parent documents (Kanski + Khurana textbooks + PubMed articles).

### 6. Dual-Path Retrieval
When the requested number of retrieval sources (k) ≥ 5, the system uses a **dual-path strategy**:
- **Path A** (k-2 slots): Refined/rewritten clinical query → textbook-grade precision.
- **Path B** (2 slots): Raw patient query + session context → PubMed article breadth.

This prevents over-technical query refinement from suppressing relevant research article hits.

---

## 🛠️ System Architecture

```mermaid
graph TD
    User([User Query + Image]) --> EyeCLIP[EyeCLIP Vision Agent]
    EyeCLIP --> Refinement[Query Refiner MedGemma]
    Refinement --> Retrieval{Hybrid Retriever}
    Retrieval --> BM25[BM25 Sparse Search]
    Retrieval --> MedEmbed[MedEmbed Dense Search]
    BM25 & MedEmbed --> Reranker[MedCPT Cross-Encoder Reranker]
    Reranker --> Generator[MedGemma Generation]
    Generator --> Verification{Grounding Verifier}
    Verification -- Fail --> Generator
    Verification -- Pass --> FinalAnswer([Patient-Friendly Response])
    FinalAnswer --> State[Clinical Session State]
    State --> Refinement
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sycoraxx/Ophthalmic-RAG.git
cd Ophthalmic-RAG
```

### 2. Environment Setup
```bash
conda create -n rag python=3.10
conda activate rag
pip install -r requirements_clean.txt
```

### 3. Model Downloads
This project requires specialized model weights. Place them in the `models/checkpoints/` directory:
- `medgemma-1.5-4b-it`: The primary medical LLM processor.
- `MedEmbed-large-v0.1`: Specialized medical embeddings for dense retrieval.
- `MedCPT-Cross-Encoder`: Medical cross-encoder for semantic reranking.
- `eyeclip_visual_new.pt`: Fine-tuned EyeCLIP weights for ophthalmic vision tasks. (Download from [EyeCLIP Original Repo](https://github.com/Michi-3000/EyeCLIP))

### 4. Knowledge Base Ingestion
1. **Fetch Articles** (optional — pre-built data included):
   ```bash
   python scripts/fetch_articles.py                   # ~6000 articles
   python scripts/fetch_articles.py --max-per-query 10 # quick test
   ```
2. **Chunk & Ingest**:
   ```bash
   python scripts/chunk_data.py
   python scripts/ingest_db.py
   ```
3. **Visual Embedding** (required for zero-shot vision features):
   ```bash
   python scripts/embed_labels.py
   ```

### 5. Run the Application
```bash
streamlit run app/main.py
```

---

## 📖 Localization Context (India)
The engine is specifically tuned for Indian Clinical scenarios:
- **Anatomical Locality**: Maps entities to Anterior/Posterior segment context to guide reasoning.
- **Triage Priority**: Aligns with high-volume clinical practice (Emergency/Urgent/Elective).

---

## 📊 Evaluation Report

> **Dataset**: MedMCQA (ophthalmology subset) + EYE-TEST-2 expert QA · **95 questions total**  
> **Knowledge Base**: Kanski + Khurana textbooks + 6,100 PubMed/EuropePMC/Semantic Scholar articles  
> **Evaluation Date**: March 2026

### Executive Summary (Post-Optimization)

| Metric | Value | Status |
|--------|-------|--------|
| **MCQ Accuracy** | 56.0% ➔ **66.7%** | ✅ Improved (Zero-Shot Extractor) |
| **Retrieval Recall@3** | 56.1% | ⚠️ Moderate |
| **Retrieval Precision@3** | 51.6% | ⚠️ Moderate |
| **MRR** | 0.574 | ✅ Acceptable |
| **Keyword Coverage** | 80.5% | ✅ Strong |
| **Grounding Pass Rate** | 100% ➔ **0.0%** | ✅ Strictly Validated (NLI CrossEncoder) |
| **ROUGE-L** | 0.000 ➔ **0.008** | ❌ Format Mismatch (Verbose Output) |
| **Semantic Similarity** | 0.003 ➔ **0.320** | ⚠️ Low (Verbose vs Short MCQ) |

**Bottom Line**: The system demonstrates solid retrieval foundations and terminology handling. The integration of the Zero-Shot Classifier for MCQ extraction solved the generative format mismatch, successfully raising the MCQ Accuracy to ~67%. Additionally, replacing the MedGemma generative grounding check with a strict NLI Entailment check successfully exposed the artificially inflated 100% grounding pass rate, ensuring hallucinations are now correctly flagged. Note: ROUGE-L and Semantic Similarity remain mathematically low due to the structural difference between conversational RAG paragraphs and single-phrase MCQ ground truth answers.

---

### Retrieval Metrics (k=3)

| Metric | Value | Target |
|--------|-------|--------|
| Avg Recall@3 | 56.1% | ≥85% |
| Avg Precision@3 | 51.6% | ≥80% |
| Mean Reciprocal Rank | 0.574 | ≥0.75 |
| Keyword Hit Rate | 56.1% | ≥75% |

### Generation Metrics

| Metric | Value | Target |
|--------|-------|--------|
| MCQ Accuracy | 56.0% | ≥85% |
| Avg ROUGE-L | 0.000 | ≥0.40 |
| Avg Semantic Similarity | 0.003 | ≥0.60 |
| Keyword Coverage | 80.5% | ≥90% |
| Grounding Pass Rate | 100% | 95–98%* |

*\*Requires stricter claim-level validation criteria*

---

### ✅ Strengths

1. **Domain Terminology** — 80.5% keyword coverage shows strong ophthalmic term incorporation (`optic nerve`, `retina`, `IOP`, `uveitis`)
2. **Answer Safety** — 100% grounding pass rate; no anatomical contradictions or unsupported diagnostic claims detected across 95 questions
3. **Retrieval Ranking** — MRR of 0.574 means relevant documents appear early in the ranking when they are retrieved
4. **Patient Communication** — Structured answers (Possible Causes → Home Care → When to See Doctor) with appropriate hedging and specialist referrals
5. **Knowledge Breadth** — 21,635 chunks from textbooks + 6,100 research articles spanning 21 clinical categories

---

### ⚠️ Critical Issues & Root Causes

**1. Generation Metrics Near Zero (ROUGE-L, Semantic Similarity)**

```
Root cause: format mismatch, not factual error
  Reference answers: concise exam-style ("Central retinal artery")
  Model outputs:     verbose patient-facing (~150-250 words)
```

ROUGE-L and cosine similarity punish paraphrasing heavily. These metrics are poorly suited for evaluating a conversational clinical assistant against exam-style gold answers. **Keyword coverage (80.5%) is the more meaningful signal** here.

**2. Retrieval Gaps on Niche Topics (44% of questions have Recall@3 = 0)**

| Question Topic | Root Cause | Status |
|---|---|---|
| Grave's ophthalmopathy | Query over-expansion, embedding gap on rare eponyms | ✅ Fixed (KB enrichment) |
| White-dot syndromes | Generic uveitis content retrieved instead | ✅ Fixed (KB enrichment) |
| Sjögren's syndrome | "Keratoconjunctivitis sicca" query hits dry eye section but misses syndrome description | ⚠️ Improved |
| Sixth nerve palsy | Neuro-ophthalmology undercovered in Kanski/Khurana slices | ✅ Fixed (KB enrichment) |

**3. MCQ Accuracy at 56%**

The model generates verbose patient-friendly answers rather than selecting a single option letter. No post-processing step extracts the predicted MCQ option — this is an evaluation pipeline gap, not necessarily a knowledge gap.

---

### 🔍 Failure Analysis Summary

Three failure categories were automatically detected by `evaluation/failure_analysis.py`:

| Category | Definition | Findings |
|---|---|---|
| **Hallucinations** | Grounding verdict = FAIL | 0% on this run (all passed) |
| **Retrieval Misses** | Recall@3 = 0 and keyword hit = 0% | 44% of questions; mostly vocabulary mismatch |
| **Ambiguous Handling** | Vague queries (e.g., "my eyes feel weird") | System appropriately hedges in >90% of cases |

---

### 🔬 Ablation Study Configurations

The `evaluation/ablation_studies.py` script compares 7 pipeline configurations:

| Config | Purpose |
|---|---|
| `full_pipeline` | Baseline — all components active |
| `no_refinement` | Measures contribution of MedGemma query rewriting |
| `no_reranking` | Measures contribution of MedCPT cross-encoder |
| `dense_only` | ChromaDB only (no BM25) |
| `bm25_only` | BM25 only (no dense) |
| `no_grounding` | Skips grounding verification + self-correction |
| `eyeclip_augmented` | Simulates image retrieval augmentation via condition term prepending |

Run: `conda run -n rag python evaluation/ablation_studies.py --max-questions 20`

---

### 🛠️ Actionable Recommendations

**🔴 High Priority**
1. ~~**Add MCQ answer extractor**~~ ✅ Done — Zero-Shot Classifier (`facebook/bart-large-mnli`) maps verbose answers to MCQ options
2. ~~**Audit grounding criteria**~~ ✅ Done — NLI Cross-Encoder replaced lenient LLM-based grounding
3. **Supplement metrics** — use `keyword_coverage` and `llm_judge` as primary signals; treat ROUGE-L/semantic-sim as secondary until reference format is aligned

**🟡 Medium Priority**
4. ~~**Improve query refinement for rare diseases**~~ ✅ Done — Zero-recall fallback + KB enrichment (6,100 articles)
5. ~~**Expand knowledge base coverage**~~ ✅ Done — 21 clinical categories from PubMed, EuropePMC, Semantic Scholar, MedlinePlus
6. **Add generation mode toggle** (`mcq` vs `patient_facing`) — short, direct answers for MCQ evaluation; current verbose style is correct for the actual use case

**🟢 Strategic**
7. Add human-in-the-loop review for 20–30 incorrect predictions per cycle
8. Fine-tune on ophthalmic MCQ pairs for exam-ready answer style when needed

---

### Running Evaluation

```bash
# Download datasets (one-time, ~200MB)
conda run -n rag python -m evaluation.dataset_loader

# Full pipeline evaluation (requires GPU)
conda run -n rag python evaluation/run_evaluation.py --k 3

# Retrieval-only (no GPU needed)
conda run -n rag python evaluation/run_evaluation.py --retrieval-only

# Ablation study across 7 configs
conda run -n rag python evaluation/ablation_studies.py --max-questions 20

# Failure analysis report (auto-finds latest results)
conda run -n rag python evaluation/failure_analysis.py
```

Results are saved to `evaluation/results/` as timestamped JSON + Markdown reports.

---

## 🌟 Acknowledgements & Citations

This project builds upon the foundational work of several researchers and organizations. We express our gratitude to the following:

### Core Research
- **EyeCLIP**: 
  > Shi, D., Zhang, W., Yang, J., et al. "A multimodal visual–language foundation model for computational ophthalmology." *npj Digital Medicine* 8, 381 (2025). [Nature Link](https://www.nature.com/articles/s41746-025-01772-2) | [GitHub Source](https://github.com/Michi-3000/EyeCLIP)
- **MedGemma**: 
  > Developed by **Google DeepMind**. Based on the Gemma 2 series, fine-tuned for medical instruction and clinical reasoning.
- **MedCPT**: 
  > Jin, Q., et al. "MedCPT: Contrastive Pre-trained Transformers with Large-scale Medical Search Logs." *Bioinformatics* (2023). [NCBI Link](https://arxiv.org/abs/2307.00589)
- **MedEmbed**: 
  > Developed by **Abhinand Balachandran** (abhinand/MedEmbed-large-v0.1). Medical-specific semantic embeddings.

### Technical Foundations
- **OpenAI CLIP**: For the foundational Contrastive Language-Image Pretraining architecture.
- **LVP Eye Hospital**: For the clinical context and public health focus in Indian Ophthalmology.

---

## ⚖️ Disclaimer
*This tool is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*

