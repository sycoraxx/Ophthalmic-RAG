from __future__ import annotations

from typing import Optional


def source_bucket(source: Optional[str]) -> str:
    token = (source or "").strip().lower()
    if token.startswith("user_query"):
        return "user_query"
    if token.startswith("eyeclip"):
        return "eyeclip"
    if token.startswith("merged"):
        return "merged"
    if token.startswith("answer"):
        return "answer"
    return "other"


def source_weight(source: Optional[str]) -> float:
    bucket = source_bucket(source)
    if bucket == "user_query":
        return 1.0
    if bucket == "eyeclip":
        return 0.9
    if bucket == "merged":
        return 0.75
    if bucket == "answer":
        return 0.5
    return 0.6


def source_rank(source: Optional[str]) -> int:
    ranking = {
        "user_query": 5,
        "eyeclip": 4,
        "merged": 3,
        "other": 2,
        "answer": 1,
    }
    return ranking.get(source_bucket(source), 2)


def clamp_confidence(value: float) -> float:
    return max(0.05, min(1.0, float(value)))
