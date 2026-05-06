from typing import Optional


def make_cache_key(
    content: str,
    top_k: int,
    entity_type: Optional[str] = None,
) -> str:
    normalized = " ".join(content.lower().strip().split())

    if entity_type:
        return f"{normalized}:{entity_type}:{top_k}"

    return f"{normalized}:{top_k}"

def build_text(payload: dict) -> str:
    return (
        f"Должность: {payload.get('title', '')};\n"
        f"Описание: {payload.get('description', '')};\n"
        f"Навыки: {payload.get('skills', '')}"
    )


def build_match_output(points) -> list[dict]:
    return [
        {
            "id": str(p.id),
            "title": p.payload.get("title", ""),
            "skills": p.payload.get("skills", ""),
            "description": p.payload.get("description", ""),
            "vacancy_ids": p.payload.get("vacancy_ids", []),
            "score": getattr(p, "score", None),
        }
        for p in points
    ]