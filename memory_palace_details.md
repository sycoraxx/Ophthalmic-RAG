# MemPalace: Longitudinal Ophthalmic Patient Memory

The MemPalace integration provides a robust, multi-layered memory system for the Ophthalmic RAG pipeline. It allows the system to remember patient-specific clinical details across multiple turns and sessions, enabling truly longitudinal care coordination.

## Architecture

MemPalace uses a dual-storage approach to ensure both semantic relevance and structural integrity:

1.  **Vector Store (ChromaDB):**
    *   **Drawers:** Individual clinical entities (symptoms, findings, conditions) are stored as "drawers".
    *   **Semantic Search:** When a patient asks a question, MemPalace performs a vector search over these drawers to find relevant past clinical context.
    *   **Spatial Indexing:** Memory is organized into "Rooms" (e.g., Anterior Segment, Posterior Segment, Adnexa) based on the anatomical structure involved.

2.  **Knowledge Graph (SQLite):**
    *   **Triples:** Entities are linked via clinical relationships (e.g., `Patient -> has_condition -> Glaucoma`, `Glaucoma -> located_in -> Anterior Segment`).
    *   **Structural Context:** This allows the system to reason about relationships that might not be captured by simple vector similarity.

## Workflow

### 1. Recording Memory (`record_turn`)
On every turn where the grounding verdict is `PASS` or `PARTIAL`:
*   Clinical entities are extracted from the user query and the assistant's answer.
*   These entities are filed into the appropriate "Room" and "Wing" (Patient ID).
*   Entities are timestamped and assigned a confidence score.

### 2. Fetching Context (`fetch_context`)
When a new query is received:
*   The system detects the anatomical structures mentioned.
*   It targets the corresponding "Rooms" in the MemPalace.
*   It retrieves the most relevant "Loci" (memory entries).
*   These are injected into the LLM prompt as `PATIENT LONGITUDINAL MEMORY`.

### 3. Clinician Summary (`build_clinician_summary`)
MemPalace can generate a structured JSON summary for physicians, including:
*   Active problem list.
*   Current symptoms and findings.
*   Recent memory loci.
*   Personalized visit notes.

## Technical Details

*   **Backend:** ChromaDB for vector search, SQLite for Knowledge Graph.
*   **Anatomy Integration:** Uses a dedicated `EyeAnatomyGraph` to map layperson terms to clinical "Rooms".
*   **Source Weighting:** User-reported symptoms are weighted higher than assistant-generated inferences to prevent hallucination feedback loops.

## Future Enhancements
*   **Temporal Decay:** Weighting more recent memories higher than historical ones (already partially implemented via `filed_at` sorting).
*   **Trend Analysis:** Detecting disease progression (e.g., worsening visual acuity) over multiple years of stored memory.
