from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from src.anatomy import get_eye_anatomy_graph
from src.state.clinical_entity_extractor import ClinicalEntity, EntityType, OphthalmicRegion
from src.state.clinical_session_state import ClinicalSessionState


@dataclass
class MemoryContext:
    query_terms: str
    generation_block: str
    rooms: list[str]
    loci_count: int


class PatientMemoryStore:
    """SQLite-backed longitudinal patient memory for anatomy-grounded loci."""

    EXPORT_DIR = Path("./data/sessions/clinician_exports")

    def __init__(self, db_path: str = "./data/sessions/patient_memory.sqlite", enabled: bool = True):
        self.enabled = enabled
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph = get_eye_anatomy_graph()
        if self.enabled:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_loci (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    session_id TEXT,
                    turn_id INTEGER,
                    created_at TEXT NOT NULL,
                    created_ts INTEGER NOT NULL,
                    anatomy_room TEXT,
                    subroom TEXT,
                    entity_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    source TEXT,
                    modality TEXT,
                    contradiction_status TEXT NOT NULL DEFAULT 'active',
                    supersedes_locus_id INTEGER,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_patient_loci_patient_time
                ON patient_loci(patient_id, created_ts DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_patient_loci_patient_room
                ON patient_loci(patient_id, anatomy_room)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_patient_loci_status
                ON patient_loci(patient_id, contradiction_status)
                """
            )

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

    def _upsert_locus(
        self,
        conn: sqlite3.Connection,
        *,
        patient_id: str,
        session_id: str,
        turn_id: int,
        room: Optional[str],
        subroom: Optional[str],
        entity: ClinicalEntity,
        created_at: str,
        created_ts: int,
    ):
        normalized = (entity.normalized or entity.text or "").strip().lower()
        if not normalized:
            return

        existing = conn.execute(
            """
            SELECT id, confidence, evidence_count
            FROM patient_loci
            WHERE patient_id = ?
              AND contradiction_status = 'active'
              AND entity_type = ?
              AND normalized_value = ?
              AND COALESCE(anatomy_room, '') = COALESCE(?, '')
            ORDER BY created_ts DESC
            LIMIT 1
            """,
            (patient_id, entity.entity_type.value, normalized, room),
        ).fetchone()

        metadata = {
            "region": entity.region.value if entity.region else None,
            "priority": entity.priority.value if entity.priority else None,
            "spatial_location": entity.spatial_location,
        }

        if existing:
            prior_conf = float(existing["confidence"])
            prior_evidence = int(existing["evidence_count"])
            alpha = 0.35 if prior_evidence < 3 else 0.2
            merged_conf = max(0.05, min(1.0, (1 - alpha) * prior_conf + alpha * entity.confidence))
            merged_evidence = min(prior_evidence + 1, 12)
            conn.execute(
                """
                UPDATE patient_loci
                SET session_id = ?,
                    turn_id = ?,
                    created_at = ?,
                    created_ts = ?,
                    confidence = ?,
                    evidence_count = ?,
                    source = ?,
                    modality = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    session_id,
                    turn_id,
                    created_at,
                    created_ts,
                    merged_conf,
                    merged_evidence,
                    entity.source,
                    entity.modality,
                    json.dumps(metadata, ensure_ascii=True),
                    int(existing["id"]),
                ),
            )
            new_row_id = int(existing["id"])
        else:
            cur = conn.execute(
                """
                INSERT INTO patient_loci (
                    patient_id, session_id, turn_id, created_at, created_ts,
                    anatomy_room, subroom, entity_type, value, normalized_value,
                    confidence, evidence_count, source, modality,
                    contradiction_status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    patient_id,
                    session_id,
                    turn_id,
                    created_at,
                    created_ts,
                    room,
                    subroom,
                    entity.entity_type.value,
                    entity.text,
                    normalized,
                    max(0.05, min(1.0, entity.confidence)),
                    1,
                    entity.source,
                    entity.modality,
                    json.dumps(metadata, ensure_ascii=True),
                ),
            )
            row_id = cur.lastrowid
            if row_id is None:
                return
            new_row_id = int(row_id)

        # Contradiction-aware supersession for mutually-exclusive primary buckets.
        if room and entity.entity_type in {EntityType.ANATOMY, EntityType.CONDITION, EntityType.IMAGING}:
            conn.execute(
                """
                UPDATE patient_loci
                SET contradiction_status = 'superseded',
                    supersedes_locus_id = ?
                WHERE patient_id = ?
                  AND contradiction_status = 'active'
                  AND anatomy_room = ?
                  AND entity_type = ?
                  AND normalized_value != ?
                  AND id != ?
                """,
                (new_row_id, patient_id, room, entity.entity_type.value, normalized, new_row_id),
            )

    def record_turn(
        self,
        *,
        patient_id: str,
        session_id: str,
        turn_id: int,
        entities: Iterable[ClinicalEntity],
        session_state: Optional[ClinicalSessionState] = None,
    ):
        if not self.enabled:
            return
        patient_id = (patient_id or "").strip()
        if not patient_id:
            return

        entities = list(entities)
        if not entities:
            return

        created_at = datetime.utcnow().isoformat(timespec="seconds")
        created_ts = int(time.time())

        with self._connect() as conn:
            for entity in entities:
                room, subroom = self._extract_room_subroom(entity, session_state)
                self._upsert_locus(
                    conn,
                    patient_id=patient_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    room=room,
                    subroom=subroom,
                    entity=entity,
                    created_at=created_at,
                    created_ts=created_ts,
                )

    def _collect_patient_loci(self, patient_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM patient_loci
                WHERE patient_id = ?
                  AND contradiction_status = 'active'
                ORDER BY created_ts DESC
                LIMIT ?
                """,
                (patient_id, limit),
            ).fetchall()

        loci: list[dict[str, Any]] = []
        for row in rows:
            loci.append({
                "room": row["anatomy_room"] or "unspecified",
                "subroom": row["subroom"],
                "entity_type": row["entity_type"],
                "value": row["value"],
                "normalized_value": row["normalized_value"],
                "confidence": float(row["confidence"]),
                "created_at": row["created_at"],
                "turn_id": int(row["turn_id"]) if row["turn_id"] is not None else None,
                "session_id": row["session_id"],
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
            "backend": "sqlite",
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

    def _score_row(self, row: sqlite3.Row, now_ts: int) -> float:
        age_hours = max(0.0, (now_ts - int(row["created_ts"])) / 3600.0)
        recency = 1.0 / (1.0 + age_hours / 72.0)
        confidence = float(row["confidence"])
        evidence = min(int(row["evidence_count"]) / 5.0, 1.0)
        return 0.55 * confidence + 0.25 * recency + 0.2 * evidence

    def fetch_context(
        self,
        *,
        patient_id: str,
        query_text: str,
        session_state: Optional[ClinicalSessionState] = None,
        max_items: int = 8,
    ) -> MemoryContext:
        if not self.enabled:
            return MemoryContext(query_terms="", generation_block="", rooms=[], loci_count=0)

        patient_id = (patient_id or "").strip()
        if not patient_id:
            return MemoryContext(query_terms="", generation_block="", rooms=[], loci_count=0)

        target_rooms = self._resolve_target_rooms(query_text, session_state=session_state)
        now_ts = int(time.time())

        with self._connect() as conn:
            if target_rooms:
                placeholders = ", ".join("?" for _ in target_rooms)
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM patient_loci
                    WHERE patient_id = ?
                      AND contradiction_status = 'active'
                      AND anatomy_room IN ({placeholders})
                    ORDER BY created_ts DESC
                    LIMIT 64
                    """,
                    (patient_id, *target_rooms),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM patient_loci
                    WHERE patient_id = ?
                      AND contradiction_status = 'active'
                    ORDER BY created_ts DESC
                    LIMIT 64
                    """,
                    (patient_id,),
                ).fetchall()

        if not rows:
            return MemoryContext(query_terms="", generation_block="", rooms=target_rooms, loci_count=0)

        scored = sorted(rows, key=lambda r: self._score_row(r, now_ts), reverse=True)[:max_items]

        query_terms = []
        generation_lines = []
        seen_terms = set()
        for row in scored:
            term = (row["normalized_value"] or row["value"] or "").strip().lower()
            if term and term not in seen_terms:
                query_terms.append(term)
                seen_terms.add(term)

            room = row["anatomy_room"] or "unspecified"
            kind = row["entity_type"]
            value = row["value"]
            conf = float(row["confidence"])
            generation_lines.append(f"- [{room}] {kind}: {value} (confidence {conf:.2f})")

        generation_block = ""
        if generation_lines:
            generation_block = "PATIENT LONGITUDINAL MEMORY (anatomy-grounded):\n" + "\n".join(generation_lines)

        return MemoryContext(
            query_terms=" ".join(query_terms),
            generation_block=generation_block,
            rooms=target_rooms,
            loci_count=len(scored),
        )

    def get_patient_summary(self, patient_id: str, max_items: int = 6) -> dict[str, Any]:
        ctx = self.fetch_context(patient_id=patient_id, query_text="", session_state=None, max_items=max_items)
        return {
            "patient_id": patient_id,
            "loci_count": ctx.loci_count,
            "rooms": ctx.rooms,
            "query_terms": ctx.query_terms,
            "generation_block": ctx.generation_block,
        }
