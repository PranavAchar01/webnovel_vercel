"""Search index integration for the web novel service.

The service is designed to index metadata about novels for quick
keyword search. When the Whoosh library is available it is used to
build a proper inverted index with boosted fields and ranking. If
Whoosh is not available (because external packages cannot be
installed) the module falls back to a naive search implementation
based on scanning the SQLite database.

The primary entry points are ``ensure_index()``, ``update_index()`` and
``search_novels()``. These functions abstract away whether a real
indexing backend is being used.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import db


try:
    from whoosh import index as whoosh_index  # type: ignore
    from whoosh.fields import Schema, TEXT, KEYWORD, ID, NUMERIC, BOOLEAN, DATETIME  # type: ignore
    from whoosh.qparser import MultifieldParser  # type: ignore

    _WHOOSH_AVAILABLE = True
except Exception:
    _WHOOSH_AVAILABLE = False

INDEX_DIR = Path(os.environ.get("WEBNOVEL_INDEX_DIR", "/mnt/data/search_index"))


def ensure_index() -> None:
    """Create the search index directory and schema if necessary.

    If Whoosh is unavailable the function is a no-op. When Whoosh is
    available it creates or opens the index directory and ensures the
    schema matches the specification: fields for id, title, author,
    description, tags, chapters (numeric count), audio_available and
    updated_at. The title field is given a higher boost than author
    or description so matches on the title rank higher in search
    results. Field boosting is documented in the Whoosh
    documentation【697090982858053†L255-L269】.
    """
    if not _WHOOSH_AVAILABLE:
        return
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not whoosh_index.exists_in(INDEX_DIR):
        schema = Schema(
            id=ID(unique=True, stored=True),
            title=TEXT(stored=True, field_boost=4.0),
            author=TEXT(stored=True, field_boost=2.0),
            description=TEXT(stored=True),
            tags=KEYWORD(commas=True, stored=True),
            chapters=NUMERIC(stored=True),
            audio_available=BOOLEAN(stored=True),
            updated_at=DATETIME(stored=True),
        )
        whoosh_index.create_in(INDEX_DIR, schema)


def _update_whoosh_index(novel: Dict[str, any]) -> None:
    """Add or update a novel's metadata in the Whoosh index."""
    if not _WHOOSH_AVAILABLE:
        return
    ensure_index()
    idx = whoosh_index.open_dir(INDEX_DIR)
    writer = idx.writer()
    writer.update_document(
        id=novel["id"],
        title=novel.get("title", ""),
        author=novel.get("author", ""),
        description=novel.get("description", ""),
        tags=novel.get("tags", ""),
        chapters=novel.get("num_chapters", 0),
        audio_available=novel.get("audio_available", False),
        updated_at=novel.get("updated_at"),
    )
    writer.commit()


def update_index(novel: Dict[str, any]) -> None:
    """Update the search index with a novel's metadata.

    If Whoosh is available the novel's metadata is indexed for full
    text search. Otherwise this function is a no-op because naive
    search reads directly from the database.
    """
    _update_whoosh_index(novel)


def _search_whoosh(query_str: str, audio_available: Optional[bool] = None) -> List[str]:
    """Run a search query against the Whoosh index and return matching novel IDs."""
    ensure_index()
    idx = whoosh_index.open_dir(INDEX_DIR)
    search_fields = ["title", "author", "description", "tags"]
    parser = MultifieldParser(search_fields, schema=idx.schema)
    q = parser.parse(query_str)
    ids: List[str] = []
    with idx.searcher() as searcher:
        results = searcher.search(q, limit=20)
        for hit in results:
            if audio_available is not None:
                if hit["audio_available"] != audio_available:
                    continue
            ids.append(hit["id"])
    return ids


def _search_naive(query_str: str, audio_available: Optional[bool] = None) -> List[str]:
    """Search novels in SQLite when Whoosh is unavailable.

    The query string is split on whitespace into keywords. Each novel
    receives a simple score based on the number of keyword matches in
    its title, author, description and tags. Title matches are
    weighted highest, followed by author, then description and tags.
    """
    keywords = [kw.lower() for kw in re.split(r"\W+", query_str) if kw.strip()]
    if not keywords:
        # Return all novel IDs
        return [n["id"] for n in db.list_novels()]
    novels = db.list_novels()
    scores: List[Tuple[str, float]] = []
    for novel in novels:
        # Optional audio filter
        if audio_available is not None:
            # Determine if novel has audio by checking any chapter has audio_path
            chapters = db.get_chapters(novel["id"])
            has_audio = all(ch.get("audio_path") for ch in chapters) if chapters else False
            if has_audio != audio_available:
                continue
        score = 0.0
        text_fields = {
            "title": novel.get("title", ""),
            "author": novel.get("author", ""),
            "description": novel.get("description", ""),
            "tags": novel.get("tags", ""),
        }
        weights = {
            "title": 4.0,
            "author": 2.0,
            "description": 1.0,
            "tags": 1.0,
        }
        for field_name, field_value in text_fields.items():
            text = field_value.lower() if field_value else ""
            weight = weights[field_name]
            for kw in keywords:
                if kw in text:
                    score += weight
        if score > 0:
            scores.append((novel["id"], score))
    # Sort by score descending
    scores.sort(key=lambda t: (-t[1], t[0]))
    return [t[0] for t in scores[:20]]


def search_novels(query_str: str, audio_available: Optional[bool] = None) -> List[str]:
    """Search for novels matching the query string and return their IDs.

    If Whoosh is available and an index exists then the query is
    delegated to Whoosh. Otherwise a naive search is performed.
    """
    if _WHOOSH_AVAILABLE and (INDEX_DIR / "MAIN_WRITELOCK").exists() or whoosh_index.exists_in(INDEX_DIR):
        try:
            return _search_whoosh(query_str, audio_available=audio_available)
        except Exception:
            # Fall back to naive if anything goes wrong
            pass
    return _search_naive(query_str, audio_available=audio_available)
