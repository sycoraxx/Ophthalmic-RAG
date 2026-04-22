from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from src.anatomy import get_eye_anatomy_graph
from src.state.clinical_entity_extractor import ClinicalEntity, EntityType, OphthalmicRegion
from src.state.clinical_session_state import ClinicalSessionState
from src.state.entity_source_policy import source_bucket, source_weight
from src.state.patient_memory_store import MemoryContext


def _sqlite_version_tuple() -> tuple[int, int, int]:
    raw = sqlite3.sqlite_version
    parts = raw.split(".")
    padded = (parts + ["0", "0", "0"])[:3]
    try:
        return int(padded[0]), int(padded[1]), int(padded[2])
    except Exception:
        return 0, 0, 0


def _ensure_sqlite_compat(min_version: tuple[int, int, int] = (3, 35, 0)) -> bool:
    """Patch stdlib sqlite3 with pysqlite3 when runtime SQLite is too old for Chroma."""
    if _sqlite_version_tuple() >= min_version:
        return True

    try:
        import pysqlite3  # type: ignore

        sys.modules["sqlite3"] = pysqlite3
        return True
    except Exception:
        return False


def _load_mempalace_symbols():
    if not _ensure_sqlite_compat():
        raise RuntimeError("sqlite3 >= 3.35 is required for MemPalace/Chroma")

    from mempalace.knowledge_graph import KnowledgeGraph as _KnowledgeGraph
    from mempalace.palace import build_closet_lines as _build_closet_lines
    from mempalace.palace import purge_file_closets as _purge_file_closets
    from mempalace.palace import get_closets_collection as _get_closets_collection
    from mempalace.palace import get_collection as _get_collection
    from mempalace.palace import upsert_closet_lines as _upsert_closet_lines
    from mempalace.searcher import search_memories as _search_memories

    return (
        _KnowledgeGraph,
        _get_collection,
        _get_closets_collection,
        _search_memories,
        _build_closet_lines,
        _upsert_closet_lines,
        _purge_file_closets,
    )


class MemPalacePatientMemoryStore:
    """MemPalace-backed longitudinal memory adapter for patient context."""

    EXPORT_DIR = Path("./data/sessions/clinician_exports")

    def __init__(
        self,
        palace_path: str = "./data/sessions/mempalace_palace",
        enabled: bool = True,
        enable_kg: bool = True,
    ):
        self.enabled = bool(enabled)
        self.palace_path = str(palace_path)
        self.enable_kg = bool(enable_kg)
        self._graph = get_eye_anatomy_graph()
        self._drawers = None
        self._closets = None
        self._kg = None
        self._search_memories = None
        self._build_closet_lines = None
        self._upsert_closet_lines = None
        self._purge_file_closets = None

        if not self.enabled:
            return

        (
            kg_cls,
            get_collection,
            get_closets_collection,
            search_memories,
            build_closet_lines,
            upsert_closet_lines,
            purge_file_closets,
        ) = _load_mempalace_symbols()

        Path(self.palace_path).mkdir(parents=True, exist_ok=True)
        self._drawers = get_collection(self.palace_path, create=True)
        self._closets = get_closets_collection(self.palace_path, create=True)
        self._search_memories = search_memories
        self._build_closet_lines = build_closet_lines
        self._upsert_closet_lines = upsert_closet_lines
        self._purge_file_closets = purge_file_closets

        if self.enable_kg and kg_cls is not None:
            kg_path = str(Path(self.palace_path) / "knowledge_graph.sqlite3")
            self._kg = kg_cls(db_path=kg_path)

    def _wing_for_patient(self, patient_id: str) -> str:
        raw = (patient_id or "").strip().lower()
        safe = re.sub(r"[^a-z0-9._-]+", "_", raw).strip("_")
        safe = safe[:80] if safe else "anonymous"
        return f"patient_{safe}"

    def _room_from_region(self, region: Optional[OphthalmicRegion]) -> Optional[str]:
        if not region:
            return None
        mapping = {
            OphthalmicRegion.ANTERIOR: "anterior_segment",
            OphthalmicRegion.POSTERIOR: "posterior_segment",
            OphthalmicRegion.ADNEXA: "adnexa",
            OphthalmicRegion.SYSTEMIC: "systemic",
        }
        return mapping.get(region)

    def _room_from_structure(self, structure: str) -> tuple[Optional[str], Optional[str]]:
        token = (structure or "").strip().lower().replace(" ", "_")
        if not token:
            return None, None

        anterior = {
            "cornea", "iris", "pupil", "lens", "anterior_chamber", "conjunctiva", "sclera",
            "anterior_segment", "aqueous_humor", "epithelium", "stroma", "endothelium",
            "bowman_layer", "descemet_membrane", "posterior_chamber",
        }
        posterior = {
            "retina", "macula", "fovea", "optic_disc", "optic_nerve", "choroid",
            "vitreous", "vitreous_humor", "vitreous_chamber", "posterior_segment", "rpe",
            "photoreceptors",
        }
        adnexa = {"eyelids", "lacrimal_system", "extraocular_muscles", "adnexa"}

        if token in anterior:
            return "anterior_segment", token
        if token in posterior:
            return "posterior_segment", token
        if token in adnexa:
            return "adnexa", token
        return None, token

    def _extract_room_subroom(
        self,
        entity: ClinicalEntity,
        session_state: Optional[ClinicalSessionState],
    ) -> tuple[Optional[str], Optional[str]]:
        if entity.entity_type == EntityType.ANATOMY:
            room, subroom = self._room_from_structure(entity.text)
            if room:
                return room, subroom

        if entity.spatial_location:
            room, subroom = self._room_from_structure(entity.spatial_location)
            if room:
                return room, subroom

        if session_state and session_state.anatomy_of_interest:
            room, subroom = self._room_from_structure(session_state.anatomy_of_interest.value)
            if room:
                return room, subroom

        room_from_region = self._room_from_region(entity.region)
        return room_from_region, None

    def _resolve_target_rooms(
        self,
        query_text: str,
        session_state: Optional[ClinicalSessionState] = None,
    ) -> list[str]:
        rooms: list[str] = []
        seen: set[str] = set()

        for structure in self._graph.detect_structures(query_text or ""):
            room, _ = self._room_from_structure(structure)
            if room and room not in seen:
                seen.add(room)
                rooms.append(room)

        if session_state and session_state.anatomy_of_interest:
            room, _ = self._room_from_structure(session_state.anatomy_of_interest.value)
            if room and room not in seen:
                seen.add(room)
                rooms.append(room)

        return rooms

    def _make_drawer_id(
        self,
        wing: str,
        session_id: str,
        turn_id: int,
        idx: int,
        normalized: str,
        entity_type: str,
    ) -> str:
        seed = f"{wing}|{session_id}|{turn_id}|{idx}|{normalized}|{entity_type}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

    def _kg_predicate(self, entity_type: EntityType) -> str:
        mapping = {
            EntityType.ANATOMY: "has_anatomy",
            EntityType.CONDITION: "has_condition",
            EntityType.SYMPTOM: "has_symptom",
            EntityType.FINDING: "has_finding",
            EntityType.IMAGING: "has_imaging",
            EntityType.MEDICATION: "uses_medication",
            EntityType.PROCEDURE: "has_procedure",
        }
        return mapping.get(entity_type, "has_entity")

    def _closet_id_base(self, source_file: str) -> str:
        digest = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:16]
        return f"closet_{digest}"

    def _effective_distance(self, distance: float, metadata: Optional[dict[str, Any]]) -> float:
        try:
            base_distance = float(distance)
        except Exception:
            base_distance = 1.0

        source = ""
        confidence = 0.0
        if isinstance(metadata, dict):
            source = str(metadata.get("source") or "")
            try:
                confidence = float(metadata.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0

        source_penalty = {
            "user_query": 0.0,
            "eyeclip": 0.03,
            "merged": 0.08,
            "other": 0.14,
            "answer": 0.45,
        }.get(source_bucket(source), 0.14)

        confidence = max(0.05, min(1.0, confidence))
        confidence_penalty = (1.0 - confidence) * (0.1 + (1.0 - source_weight(source)) * 0.35)
        return base_distance + source_penalty + confidence_penalty

    def _metadata_from_text(self, text: str, fallback_room: str) -> dict[str, Any]:
        content = text or ""
        entity_match = re.search(r"Entity type:\s*([a-z_]+)", content, flags=re.I)
        value_match = re.search(r"Value:\s*([^\.]+)", content, flags=re.I)
        conf_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", content, flags=re.I)
        source_match = re.search(r"Source:\s*([^\.]+)", content, flags=re.I)

        value = (value_match.group(1).strip() if value_match else "").strip()
        normalized = value.lower() if value else ""
        confidence = float(conf_match.group(1)) if conf_match else 0.0
        source = (source_match.group(1).strip().lower() if source_match else "")

        return {
            "room": fallback_room or "general",
            "entity_type": (entity_match.group(1).strip().lower() if entity_match else "entity"),
            "value": value or "unknown",
            "normalized_value": normalized,
            "confidence": confidence,
            "source": source,
            "source_bucket": source_bucket(source),
            "source_weight": source_weight(source),
        }

    def record_turn(
        self,
        *,
        patient_id: str,
        session_id: str,
        turn_id: int,
        entities: Iterable[ClinicalEntity],
        session_state: Optional[ClinicalSessionState] = None,
    ):
        if not self.enabled or self._drawers is None:
            return

        patient_id = (patient_id or "").strip()
        if not patient_id:
            return

        entities = list(entities)
        if not entities:
            return

        wing = self._wing_for_patient(patient_id)
        source_file = f"patient://{wing}/session/{session_id}/turn/{turn_id}"
        filed_at = datetime.utcnow().isoformat(timespec="seconds")

        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for idx, entity in enumerate(entities):
            room, subroom = self._extract_room_subroom(entity, session_state)
            normalized = (entity.normalized or entity.text or "").strip().lower()
            if not normalized:
                continue
            source = (entity.source or "").strip().lower()

            drawer_id = self._make_drawer_id(
                wing=wing,
                session_id=session_id,
                turn_id=turn_id,
                idx=idx,
                normalized=normalized,
                entity_type=entity.entity_type.value,
            )
            doc = (
                f"Patient {patient_id} longitudinal memory. "
                f"Turn {turn_id}. "
                f"Room: {room or 'unspecified'}. "
                f"Entity type: {entity.entity_type.value}. "
                f"Value: {entity.text}. "
                f"Confidence: {entity.confidence:.2f}. "
                f"Source: {entity.source}."
            )
            meta = {
                "wing": wing,
                "room": room or "general",
                "source_file": source_file,
                "chunk_index": idx,
                "filed_at": filed_at,
                "patient_id": patient_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "entity_type": entity.entity_type.value,
                "value": entity.text,
                "normalized_value": normalized,
                "confidence": float(max(0.05, min(1.0, entity.confidence))),
                "source": source,
                "source_bucket": source_bucket(source),
                "source_weight": source_weight(source),
                "subroom": subroom,
                "priority": entity.priority.value if entity.priority else "routine",
                "region": entity.region.value if entity.region else "unspecified",
                "modality": entity.modality,
            }
            documents.append(doc)
            ids.append(drawer_id)
            metadatas.append(meta)

            if self._kg is not None:
                try:
                    self._kg.add_triple(
                        patient_id,
                        self._kg_predicate(entity.entity_type),
                        normalized,
                        valid_from=filed_at,
                        source_file=source_file,
                        source_drawer_id=drawer_id,
                        adapter_name="ophthalmic_rag_mempalace",
                    )
                    if room:
                        self._kg.add_triple(
                            normalized,
                            "located_in",
                            room,
                            valid_from=filed_at,
                            source_file=source_file,
                            source_drawer_id=drawer_id,
                            adapter_name="ophthalmic_rag_mempalace",
                        )
                except Exception:
                    pass

        if not ids:
            return

        self._drawers.upsert(documents=documents, ids=ids, metadatas=metadatas)

        if (
            self._closets is not None
            and self._build_closet_lines is not None
            and self._upsert_closet_lines is not None
            and self._purge_file_closets is not None
        ):
            try:
                primary_room = metadatas[0].get("room", "general")
                combined_content = "\n".join(documents)
                closet_lines = self._build_closet_lines(
                    source_file=source_file,
                    drawer_ids=ids,
                    content=combined_content,
                    wing=wing,
                    room=primary_room,
                )
                self._purge_file_closets(self._closets, source_file)
                self._upsert_closet_lines(
                    self._closets,
                    self._closet_id_base(source_file),
                    closet_lines,
                    {
                        "wing": wing,
                        "room": primary_room,
                        "source_file": source_file,
                        "filed_at": filed_at,
                    },
                )
            except Exception:
                pass

    def _collect_patient_loci(self, patient_id: str, limit: int = 12) -> list[dict[str, Any]]:
        if self._drawers is None:
            return []

        wing = self._wing_for_patient(patient_id)
        try:
            rows = self._drawers.get(where={"wing": wing}, include=["metadatas"], limit=max(limit * 20, 60))
            metadatas = rows.metadatas if hasattr(rows, "metadatas") else (rows.get("metadatas") or [])
        except Exception:
            metadatas = []

        metadatas = [m for m in metadatas if isinstance(m, dict)]
        metadatas.sort(key=lambda m: str(m.get("filed_at", "")), reverse=True)
        loci: list[dict[str, Any]] = []
        for m in metadatas[:limit]:
            raw_turn_id = m.get("turn_id")
            try:
                parsed_turn_id = int(raw_turn_id) if raw_turn_id is not None else None
            except Exception:
                parsed_turn_id = None
            loci.append({
                "room": m.get("room") or "unspecified",
                "subroom": m.get("subroom"),
                "entity_type": m.get("entity_type"),
                "value": m.get("value"),
                "normalized_value": m.get("normalized_value"),
                "confidence": float(m.get("confidence") or 0.0),
                "created_at": m.get("filed_at"),
                "turn_id": parsed_turn_id,
                "session_id": m.get("session_id"),
            })
        return loci

    def build_clinician_summary(
        self,
        *,
        patient_id: str,
        session_id: str,
        turn_id: int,
        entities: Iterable[ClinicalEntity],
        session_state: Optional[ClinicalSessionState] = None,
        conversation_date: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized_patient_id = (patient_id or "").strip()
        normalized_session_id = (session_id or "").strip()
        loci = self._collect_patient_loci(normalized_patient_id, limit=12) if normalized_patient_id else []

        symptoms = list(dict.fromkeys((e.text for e in entities if e.entity_type == EntityType.SYMPTOM)))
        findings = list(dict.fromkeys((e.text for e in entities if e.entity_type == EntityType.FINDING)))
        conditions = list(dict.fromkeys((e.text for e in entities if e.entity_type == EntityType.CONDITION)))
        medications = list(dict.fromkeys((e.text for e in entities if e.entity_type == EntityType.MEDICATION)))
        procedures = list(dict.fromkeys((e.text for e in entities if e.entity_type == EntityType.PROCEDURE)))
        anatomy = list(dict.fromkeys((e.text for e in entities if e.entity_type == EntityType.ANATOMY)))
        imaging = list(dict.fromkeys((e.text for e in entities if e.entity_type == EntityType.IMAGING)))

        if session_state:
            if session_state.primary_condition:
                conditions.insert(0, session_state.primary_condition.value)
            if session_state.anatomy_of_interest:
                anatomy.insert(0, session_state.anatomy_of_interest.value)
            if session_state.imaging_modality:
                imaging.insert(0, session_state.imaging_modality.value)

        dedupe = lambda items: list(dict.fromkeys(item for item in items if item))

        summary = {
            "patient_id": normalized_patient_id,
            "session_id": normalized_session_id,
            "turn_id": turn_id,
            "backend": "mempalace",
            "conversation_date": conversation_date or datetime.utcnow().date().isoformat(),
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "active_problem_list": dedupe(conditions),
            "current_symptoms": dedupe(symptoms),
            "current_findings": dedupe(findings),
            "current_anatomy": dedupe(anatomy),
            "current_imaging": dedupe(imaging),
            "current_medications": dedupe(medications),
            "current_procedures": dedupe(procedures),
            "recent_memory_loci": loci,
            "visit_note": {
                "turn_id": turn_id,
                "patient_reported": dedupe(symptoms + findings + conditions),
                "memory_terms": dedupe([entry["normalized_value"] for entry in loci if entry.get("normalized_value")]),
            },
        }
        return summary

    def export_clinician_summary(
        self,
        *,
        patient_id: str,
        session_id: str,
        turn_id: int,
        entities: Iterable[ClinicalEntity],
        session_state: Optional[ClinicalSessionState] = None,
        conversation_date: Optional[str] = None,
    ) -> Path:
        summary = self.build_clinician_summary(
            patient_id=patient_id,
            session_id=session_id,
            turn_id=turn_id,
            entities=entities,
            session_state=session_state,
            conversation_date=conversation_date,
        )
        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        safe_patient = re.sub(r"[^a-zA-Z0-9._-]+", "_", (patient_id or "unknown")).strip("_") or "unknown"
        safe_session = re.sub(r"[^a-zA-Z0-9._-]+", "_", (session_id or "session")).strip("_") or "session"
        export_path = self.EXPORT_DIR / f"{safe_patient}_{safe_session}_turn{turn_id}.json"
        export_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        return export_path

    def _query_drawers_direct(self, *, query_text: str, wing: str, room: Optional[str], n_results: int) -> list[dict]:
        if self._drawers is None:
            return []

        where = {"wing": wing}
        if room:
            where = {"$and": [{"wing": wing}, {"room": room}]}

        try:
            raw = self._drawers.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        docs_outer = raw.documents if hasattr(raw, "documents") else (raw.get("documents") or [])
        metas_outer = raw.metadatas if hasattr(raw, "metadatas") else (raw.get("metadatas") or [])
        dists_outer = raw.distances if hasattr(raw, "distances") else (raw.get("distances") or [])

        docs = docs_outer[0] if docs_outer else []
        metas = metas_outer[0] if metas_outer else []
        dists = dists_outer[0] if dists_outer else []

        hits: list[dict] = []
        for idx, doc in enumerate(docs):
            meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            dist = float(dists[idx]) if idx < len(dists) else 1.0
            if not meta.get("source"):
                parsed = self._metadata_from_text(doc or "", meta.get("room", room or "general"))
                if parsed.get("source"):
                    meta["source"] = parsed.get("source")
            meta["source_bucket"] = source_bucket(meta.get("source"))
            meta["source_weight"] = source_weight(meta.get("source"))
            effective_distance = self._effective_distance(dist, meta)
            hits.append(
                {
                    "text": doc or "",
                    "wing": meta.get("wing", wing),
                    "room": meta.get("room", room or "general"),
                    "source_file": Path(str(meta.get("source_file", "?"))).name,
                    "distance": dist,
                    "effective_distance": effective_distance,
                    "metadata": meta,
                }
            )
        return hits

    def _collect_hits(
        self,
        *,
        query_text: str,
        wing: str,
        rooms: list[str],
        max_items: int,
    ) -> list[dict]:
        search_fn = self._search_memories
        if search_fn is None:
            return []

        all_hits: list[dict] = []
        seen_keys: set[tuple[str, str, str]] = set()

        queries = rooms if rooms else [None]
        for room in queries:
            if room is None:
                payload = search_fn(
                    query=query_text,
                    palace_path=self.palace_path,
                    wing=wing,
                    n_results=max(max_items * 2, 8),
                )
            else:
                payload = search_fn(
                    query=query_text,
                    palace_path=self.palace_path,
                    wing=wing,
                    room=room,
                    n_results=max(max_items * 2, 8),
                )
            hits = payload.get("results", []) if isinstance(payload, dict) else []
            for hit in hits:
                metadata = hit.get("metadata") or {}
                if not metadata:
                    metadata = self._metadata_from_text(hit.get("text", ""), hit.get("room", room or "general"))
                elif not metadata.get("source"):
                    parsed = self._metadata_from_text(
                        hit.get("text", ""),
                        metadata.get("room") or hit.get("room", room or "general"),
                    )
                    if parsed.get("source"):
                        metadata["source"] = parsed.get("source")
                
                metadata["source_bucket"] = source_bucket(metadata.get("source"))
                metadata["source_weight"] = source_weight(metadata.get("source"))
                hit["metadata"] = metadata
                
                try:
                    dist = float(hit.get("distance", hit.get("effective_distance", 1.0)))
                except Exception:
                    dist = 1.0
                hit["distance"] = dist
                hit["effective_distance"] = self._effective_distance(dist, metadata)
                
                # Robust deduplication key: Room + EntityType + NormalizedValue
                norm_val = (metadata.get("normalized_value") or "").strip().lower()
                if not norm_val:
                    continue
                
                dedupe_key = (
                    str(metadata.get("room", "general")),
                    str(metadata.get("entity_type", "entity")),
                    norm_val
                )
                
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                all_hits.append(hit)

            # Also query the vector collection directly to catch anything the searcher missed
            direct_hits = self._query_drawers_direct(
                query_text=query_text,
                wing=wing,
                room=room,
                n_results=max(max_items * 3, 12),
            )
            for hit in direct_hits:
                metadata = hit.get("metadata") or {}
                norm_val = (metadata.get("normalized_value") or "").strip().lower()
                if not norm_val:
                    continue
                
                dedupe_key = (
                    str(metadata.get("room", "general")),
                    str(metadata.get("entity_type", "entity")),
                    norm_val
                )
                
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                all_hits.append(hit)

        # Sort by effective distance (which accounts for source confidence)
        all_hits.sort(key=lambda h: float(h.get("effective_distance", h.get("distance", 1.0))))
        return all_hits[:max_items]

    def fetch_context(
        self,
        *,
        patient_id: str,
        query_text: str,
        session_state: Optional[ClinicalSessionState] = None,
        max_items: int = 6,  # Reduced from 8 for better SNR
    ) -> MemoryContext:
        if not self.enabled:
            return MemoryContext(query_terms="", generation_block="", rooms=[], loci_count=0)

        patient_id = (patient_id or "").strip()
        if not patient_id:
            return MemoryContext(query_terms="", generation_block="", rooms=[], loci_count=0)

        wing = self._wing_for_patient(patient_id)
        rooms = self._resolve_target_rooms(query_text, session_state=session_state)
        # We query slightly more and then prune
        hits = self._collect_hits(query_text=query_text or "eye symptoms", wing=wing, rooms=rooms, max_items=max_items)

        if not hits:
            return MemoryContext(query_terms="", generation_block="", rooms=rooms, loci_count=0)

        term_budget = max(1, int(max_items))
        high_priority_terms: list[str] = []
        answer_terms: list[str] = []
        seen_terms: set[str] = set()
        generation_lines: list[str] = []
        seen_loci: set[tuple[str, str, str]] = set()

        for hit in hits:
            meta = hit.get("metadata") or {}
            room = meta.get("room") or hit.get("room") or "general"
            entity_type = meta.get("entity_type") or "entity"
            value = meta.get("value") or meta.get("normalized_value") or "unknown"
            normalized = (meta.get("normalized_value") or value or "").strip().lower()
            confidence = float(meta.get("confidence") or 0.0)

            # Deduplication already happens in _collect_hits, but double-check here
            loci_key = (str(room), str(entity_type), normalized)
            if loci_key in seen_loci:
                continue
            seen_loci.add(loci_key)

            if normalized and normalized not in seen_terms:
                seen_terms.add(normalized)
                if source_bucket(meta.get("source")) == "answer":
                    answer_terms.append(normalized)
                else:
                    high_priority_terms.append(normalized)

            # More compact generation line
            generation_lines.append(
                f"- {entity_type}: {value} ({room}, conf {confidence:.2f})"
            )

        query_terms = high_priority_terms[:term_budget]
        if len(query_terms) < term_budget:
            query_terms.extend(answer_terms[: term_budget - len(query_terms)])

        # Concise header
        generation_block = "PAST CLINICAL RECORDS (MemPalace):\n" + "\n".join(generation_lines)
        return MemoryContext(
            query_terms=" ".join(query_terms),
            generation_block=generation_block,
            rooms=rooms,
            loci_count=len(hits),
        )

    def get_patient_summary(self, patient_id: str, max_items: int = 6) -> dict:
        patient_id = (patient_id or "").strip()
        if not patient_id or not self.enabled or self._drawers is None:
            return {
                "patient_id": patient_id,
                "loci_count": 0,
                "rooms": [],
                "query_terms": "",
                "generation_block": "",
            }

        wing = self._wing_for_patient(patient_id)
        try:
            rows = self._drawers.get(where={"wing": wing}, include=["metadatas"], limit=max(max_items * 20, 60))
            metadatas = rows.metadatas if hasattr(rows, "metadatas") else (rows.get("metadatas") or [])
        except Exception:
            metadatas = []

        metadatas = [m for m in metadatas if isinstance(m, dict)]
        metadatas.sort(key=lambda m: str(m.get("filed_at", "")), reverse=True)
        top = metadatas[:max_items]

        rooms = list(dict.fromkeys((m.get("room") or "general") for m in top))
        terms = []
        seen = set()
        lines = []
        for m in top:
            term = (m.get("normalized_value") or m.get("value") or "").strip().lower()
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
            lines.append(
                f"- [{m.get('room', 'general')}] {m.get('entity_type', 'entity')}: "
                f"{m.get('value', 'unknown')} (confidence {float(m.get('confidence') or 0.0):.2f})"
            )

        block = "PATIENT LONGITUDINAL MEMORY (MemPalace):\n" + "\n".join(lines) if lines else ""
        return {
            "patient_id": patient_id,
            "loci_count": len(metadatas),
            "rooms": rooms,
            "query_terms": " ".join(terms),
            "generation_block": block,
        }
