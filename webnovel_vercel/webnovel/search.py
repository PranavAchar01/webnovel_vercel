"""Simple search utilities for the webnovel service.

This module implements a naïve search across the SQLite database when Whoosh
is not available.  It scans all novels, scores them based on the presence
of query terms in different fields, and returns a list of novel IDs ordered
by relevance.  It also filters by audio availability if requested.
"""

from typing import List, Optional

from . import db


def ensure_index() -> None:
    """No-op: we don't maintain a search index without Whoosh."""
    pass


def update_index(novel: dict) -> None:
    """No-op: nothing to update when not using Whoosh."""
    pass


def search_novels(query: str, audio_available: Optional[bool] = None) -> List[str]:
    """Search novels by simple keyword matching.

    Args:
        query: Space-separated keywords to search for.
        audio_available: When True, return only novels with audio generated for
            all chapters; when False, return only novels without full audio.
            If None (default), don't filter on audio availability.

    Returns:
        A list of novel IDs sorted by descending relevance score.
    """
    query = query.strip().lower()
    if not query:
        # Return all novels (optionally filtered) when no query is provided.
        novels = db.get_novels()
        ids = [n["id"] for n in novels]
        if audio_available is None:
            return ids
        filtered = []
        for nid in ids:
            chapters = db.get_chapters(nid)
            has_audio = bool(chapters and all(ch.get("audio_path") for ch in chapters))
            if audio_available == has_audio:
                filtered.append(nid)
        return filtered

    keywords = query.split()
    results: List[tuple[str, float]] = []
    novels = db.get_novels()

    for novel in novels:
        # Compute a simple score based on keyword occurrences.
        score = 0.0
        title = (novel.get("title") or "").lower()
        author = (novel.get("author") or "").lower()
        desc = (novel.get("description") or "").lower()
        tags = (novel.get("tags") or "").lower()

        for kw in keywords:
            if kw in title:
                score += 4.0  # title has highest weight
            if kw in author:
                score += 2.0
            if kw in desc:
                score += 1.0
            if kw in tags.split(","):
                score += 1.0

        # Skip novels with zero score.
        if score == 0.0:
            continue

        # Filter by audio availability if requested.
        if audio_available is not None:
            chapters = db.get_chapters(novel["id"])
            has_audio = bool(chapters and all(ch.get("audio_path") for ch in chapters))
            if audio_available != has_audio:
                continue

        results.append((novel["id"], score))

    # Sort by score descending.
    results.sort(key=lambda item: item[1], reverse=True)
    return [nid for nid, _ in results]
