"""Main FastAPI application for the web novel service.

This module defines the HTTP API and web UI routes for the service.
Endpoints are organised according to the specification. Where
appropriate long running tasks (extraction and TTS synthesis) are
offloaded to background workers via FastAPI's ``BackgroundTasks``.

The application initialises its database and search index on startup.
Templates are rendered using Jinja2; TailwindCSS is pulled from a CDN
to provide a lightweight but modern appearance. The JavaScript
embedded in ``templates/novel.html`` implements a simple audio player
with next/previous controls and playback persistence using
``localStorage``.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from . import db, extractor, tts, search


app = FastAPI(title="Web Novel Extractor & Audiobook Service")

BASE_DIR = Path("/mnt/data/novels")
BASE_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# In‑memory cache of providers. In a full implementation this could
# dynamically load adapters from configuration.
PROVIDERS = {
    "silent": tts.DummyTTSProvider(),
}

# Dynamically register OpenAI TTS provider if an API key is available.
openai_key = os.environ.get("OPENAI_API_KEY")
openai_voice = os.environ.get("OPENAI_TTS_VOICE", "alloy")
if openai_key:
    try:
        PROVIDERS["openai"] = tts.OpenAITTSProvider(api_key=openai_key, voice=openai_voice)
    except Exception:
        # If provider initialisation fails (e.g. missing httpx), ignore silently.
        pass


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise database and search index on startup."""
    db.init_db()
    search.ensure_index()


def _format_novel_for_search(novel: Dict[str, Any]) -> Dict[str, Any]:
    """Augment a novel dict with additional fields required for indexing."""
    chapters = db.get_chapters(novel["id"])
    # Determine audio availability: all chapters must have an audio_path
    audio_available = bool(chapters and all(ch.get("audio_path") for ch in chapters))
    novel = novel.copy()
    novel["audio_available"] = audio_available
    return novel


async def background_extract_novel(novel_id: str, url: str, job_id: str) -> None:
    """Background task to extract a full novel (all chapters) from a URL."""
    try:
        db.update_job(job_id, status="running", progress=0)
        # Determine chapter URLs
        chapter_urls = await extractor.discover_chapter_urls(url)
        if not chapter_urls:
            chapter_urls = [url]
        total = len(chapter_urls)
        chapters_data: List[Dict[str, Any]] = []
        for idx, chapter_url in enumerate(chapter_urls, start=1):
            chap = await extractor.extract_chapter(chapter_url)
            if not chap:
                # Mark job error and stop
                db.update_job(job_id, status="error", error=f"Failed to extract chapter {idx}")
                db.update_novel_status(novel_id, "error")
                return
            # Save text to file
            novel_dir = BASE_DIR / novel_id
            novel_dir.mkdir(parents=True, exist_ok=True)
            chap_id = str(uuid.uuid4())
            text_filename = novel_dir / f"chapter-{idx}.txt"
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(chap["text"])
            # Insert chapter record
            chapter_record = {
                "id": chap_id,
                "novel_id": novel_id,
                "idx": idx,
                "title": chap.get("title", f"Chapter {idx}"),
                "source_url": chapter_url,
                "text_path": str(text_filename),
                "audio_path": None,
                "duration_seconds": None,
                "tts_provider": None,
                "status": "ready",  # ready with text extracted
            }
            db.insert_chapter(chapter_record)
            chapters_data.append({"id": chap_id, "title": chapter_record["title"], "text": chap["text"]})
            # Update progress (50% for extraction)
            progress = int((idx / total) * 50)
            db.update_job(job_id, progress=progress)
        # Update novel metadata after extraction
        novel = db.get_novel(novel_id) or {}
        db.insert_novel({
            "id": novel_id,
            "title": novel.get("title", ""),
            "author": novel.get("author", ""),
            "source_url": url,
            "description": novel.get("description", ""),
            "tags": novel.get("tags", ""),
            "num_chapters": total,
            "cover_path": novel.get("cover_path"),
            "status": "ready",
        })
        # Package TXT and EPUB
        txt_path = extractor.package_txt(novel_id, chapters_data, str(BASE_DIR / novel_id))
        epub_path = extractor.package_epub({"id": novel_id, "title": novel.get("title", ""),
                                            "author": novel.get("author", ""),
                                            "description": novel.get("description", "")},
                                           chapters_data, str(BASE_DIR / novel_id))
        # Mark job as complete
        db.update_job(job_id, status="done", progress=100)
        # Update search index
        search.update_index(_format_novel_for_search(db.get_novel(novel_id)))
    except Exception as e:
        db.update_job(job_id, status="error", error=str(e))
        db.update_novel_status(novel_id, "error")


async def background_tts_chapter(chapter_id: str, job_id: str, provider_name: str = "silent") -> None:
    """Background task to generate audio for a single chapter."""
    try:
        provider = PROVIDERS.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown TTS provider {provider_name}")
        db.update_job(job_id, status="running", progress=0)
        chapter = db.get_chapter(chapter_id)
        if not chapter:
            raise ValueError("Chapter not found")
        if not chapter.get("text_path") or not os.path.exists(chapter["text_path"]):
            raise ValueError("Chapter text file missing")
        with open(chapter["text_path"], "r", encoding="utf-8") as f:
            text = f.read()
        # Split into chunks
        chunks = tts.chunk_text(text)
        total_chunks = len(chunks)
        audio_bytes = b""
        total_duration = 0
        for idx, chunk in enumerate(chunks, start=1):
            chunk_audio, duration = await provider.synthesize(chunk)
            audio_bytes += chunk_audio
            total_duration += duration
            # Progress: 0–90% through TTS generation (reserve 10% for saving)
            progress = int((idx / total_chunks) * 90)
            db.update_job(job_id, progress=progress)
        # Write audio file
        novel_id = chapter["novel_id"]
        novel_dir = BASE_DIR / novel_id
        novel_dir.mkdir(parents=True, exist_ok=True)
        audio_path = novel_dir / f"chapter-{chapter['idx']}.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        # Update chapter info
        db.update_chapter_audio_info(chapter_id, str(audio_path), total_duration, provider.name)
        # Update job status
        db.update_job(job_id, status="done", progress=100)
        # Update novel index
        novel = db.get_novel(novel_id)
        if novel:
            search.update_index(_format_novel_for_search(novel))
    except Exception as e:
        db.update_job(job_id, status="error", error=str(e))


async def background_tts_novel(novel_id: str, job_id: str, provider_name: str = "silent") -> None:
    """Background task to generate audio for all chapters of a novel."""
    try:
        provider = PROVIDERS.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown TTS provider {provider_name}")
        db.update_job(job_id, status="running", progress=0)
        chapters = db.get_chapters(novel_id)
        if not chapters:
            raise ValueError("Novel has no chapters")
        total = len(chapters)
        for idx, chapter in enumerate(chapters, start=1):
            if chapter.get("audio_path"):
                continue
            # Use same logic as single chapter
            chapter_id = chapter["id"]
            chap_text_path = chapter.get("text_path")
            if not chap_text_path or not os.path.exists(chap_text_path):
                raise ValueError(f"Missing text for chapter {chapter['idx']}")
            with open(chap_text_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = tts.chunk_text(text)
            audio_bytes = b""
            total_duration = 0
            for chunk in chunks:
                chunk_audio, duration = await provider.synthesize(chunk)
                audio_bytes += chunk_audio
                total_duration += duration
            audio_path = BASE_DIR / novel_id / f"chapter-{chapter['idx']}.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            db.update_chapter_audio_info(chapter_id, str(audio_path), total_duration, provider.name)
            # Progress across chapters: allocate 90% for TTS, 10% for final
            progress = int(((idx) / total) * 90)
            db.update_job(job_id, progress=progress)
        # Update job status and novel index
        db.update_job(job_id, status="done", progress=100)
        novel = db.get_novel(novel_id)
        if novel:
            search.update_index(_format_novel_for_search(novel))
    except Exception as e:
        db.update_job(job_id, status="error", error=str(e))


@app.post("/extract")
async def extract_endpoint(request: Request) -> Response:
    """Extract a single chapter from a URL and return its content as JSON."""
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body")
    chapter = await extractor.extract_chapter(url)
    if not chapter:
        raise HTTPException(status_code=500, detail="Failed to extract chapter")
    return JSONResponse({"title": chapter.get("title", ""), "text": chapter.get("text", "")})


@app.post("/novel")
async def novel_endpoint(request: Request, background_tasks: BackgroundTasks) -> Response:
    """Extract a full novel given a table of contents URL.

    The payload must contain at least ``url``. Optional fields such as
    ``title``, ``author``, ``description`` or ``tags`` may be provided to
    prefill metadata before extraction completes. A job identifier is
    returned immediately; clients can poll ``/jobs/{id}`` for progress.
    """
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body")
    novel_id = str(uuid.uuid4())
    # Create novel record with minimal information; status extracting
    db.insert_novel({
        "id": novel_id,
        "title": data.get("title", ""),
        "author": data.get("author", ""),
        "source_url": url,
        "description": data.get("description", ""),
        "tags": data.get("tags", ""),
        "num_chapters": None,
        "cover_path": None,
        "status": "extracting",
    })
    job_id = str(uuid.uuid4())
    db.insert_job({
        "id": job_id,
        "type": "extract",
        "novel_id": novel_id,
        "payload": {"url": url},
        "status": "queued",
        "progress": 0,
        "error": None,
    })
    background_tasks.add_task(background_extract_novel, novel_id, url, job_id)
    return JSONResponse({"job_id": job_id, "novel_id": novel_id})


@app.post("/tts/chapter/{chapter_id}")
async def tts_chapter_endpoint(chapter_id: str, request: Request, background_tasks: BackgroundTasks) -> Response:
    """Generate audio for a single chapter.

    Payload may include a ``provider`` key. Unknown providers will
    result in an error. A job id is returned immediately.
    """
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass
    provider_name = data.get("provider", "silent")
    if provider_name not in PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown TTS provider")
    job_id = str(uuid.uuid4())
    db.insert_job({
        "id": job_id,
        "type": "tts_chapter",
        "novel_id": None,
        "payload": {"chapter_id": chapter_id, "provider": provider_name},
        "status": "queued",
        "progress": 0,
        "error": None,
    })
    background_tasks.add_task(background_tts_chapter, chapter_id, job_id, provider_name)
    return JSONResponse({"job_id": job_id})


@app.post("/tts/novel/{novel_id}")
async def tts_novel_endpoint(novel_id: str, request: Request, background_tasks: BackgroundTasks) -> Response:
    """Generate audio for all chapters in a novel."""
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass
    provider_name = data.get("provider", "silent")
    if provider_name not in PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown TTS provider")
    job_id = str(uuid.uuid4())
    db.insert_job({
        "id": job_id,
        "type": "tts_novel",
        "novel_id": novel_id,
        "payload": {"provider": provider_name},
        "status": "queued",
        "progress": 0,
        "error": None,
    })
    background_tasks.add_task(background_tts_novel, novel_id, job_id, provider_name)
    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> Response:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "id": job["id"],
        "type": job.get("type"),
        "status": job.get("status"),
        "progress": job.get("progress"),
        "error": job.get("error"),
        "novel_id": job.get("novel_id"),
    })


@app.get("/catalog/{novel_id}")
async def get_catalog(novel_id: str) -> Response:
    novel = db.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    chapters = db.get_chapters(novel_id)
    return JSONResponse({"novel": novel, "chapters": chapters})


@app.get("/search")
async def search_endpoint(request: Request, q: Optional[str] = None, audio: Optional[bool] = None) -> Response:
    """Search novels via the API. Use ``q`` parameter for keywords and
    ``audio`` to filter by audio availability."""
    query = q or ""
    result_ids = search.search_novels(query, audio_available=audio)
    novels = [db.get_novel(nid) for nid in result_ids if db.get_novel(nid)]
    return JSONResponse({"results": novels})


def iter_file(path: str, start: int, end: int, chunk_size: int = 8192):
    """Generator to read bytes from ``path`` between ``start`` and ``end``."""
    with open(path, "rb") as f:
        f.seek(start)
        bytes_left = end - start + 1
        while bytes_left > 0:
            read_size = min(chunk_size, bytes_left)
            data = f.read(read_size)
            if not data:
                break
            bytes_left -= len(data)
            yield data


@app.get("/media/audio/{chapter_id}")
async def stream_audio(chapter_id: str, request: Request) -> Response:
    """Stream chapter audio with HTTP range support."""
    chapter = db.get_chapter(chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    audio_path = chapter.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not generated for chapter")
    file_size = os.path.getsize(audio_path)
    range_header = request.headers.get("range")
    if range_header:
        # Parse Range header, e.g. "bytes=0-1023"
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if match:
            start_str, end_str = match.groups()
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            if end >= file_size:
                end = file_size - 1
            if start > end:
                start, end = 0, file_size - 1
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
                "Content-Type": "audio/mpeg",
            }
            return StreamingResponse(iter_file(audio_path, start, end), status_code=206, headers=headers)
    # No range: return full file
    return StreamingResponse(iter_file(audio_path, 0, file_size - 1), media_type="audio/mpeg",
                             headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"})


@app.get("/download/{novel_id}.{ext}")
async def download_endpoint(novel_id: str, ext: str) -> Response:
    """Download packaged TXT or EPUB for a novel."""
    novel_dir = BASE_DIR / novel_id
    if ext not in {"txt", "epub"}:
        raise HTTPException(status_code=400, detail="Unsupported format")
    novel = db.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    # Determine filename
    target_path = novel_dir / f"{novel_id}.{ext}"
    if not target_path.exists():
        # Generate on demand
        chapters = db.get_chapters(novel_id)
        chapters_data = []
        for chapter in chapters:
            # Load text from file
            text_path = chapter.get("text_path")
            if not text_path or not os.path.exists(text_path):
                continue
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
            chapters_data.append({"id": chapter["id"], "title": chapter.get("title", ""), "text": text})
        if ext == "txt":
            extractor.package_txt(novel_id, chapters_data, str(novel_dir))
        else:
            metadata = {"id": novel_id, "title": novel.get("title", ""), "author": novel.get("author", ""),
                        "description": novel.get("description", "")}
            extractor.package_epub(metadata, chapters_data, str(novel_dir))
    return FileResponse(str(target_path), filename=target_path.name)


# UI ROUTES

@app.get("/ui/search")
async def ui_search(request: Request, q: Optional[str] = None, audio: Optional[str] = None) -> HTMLResponse:
    """Render the search page."""
    query = q or ""
    audio_filter = None
    if audio == "true":
        audio_filter = True
    elif audio == "false":
        audio_filter = False
    result_ids = search.search_novels(query, audio_available=audio_filter)
    novels = [db.get_novel(nid) for nid in result_ids if db.get_novel(nid)]
    return templates.TemplateResponse("search.html", {"request": request, "novels": novels, "query": query})


@app.get("/ui/novel/{novel_id}")
async def ui_novel(request: Request, novel_id: str) -> HTMLResponse:
    novel = db.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    chapters = db.get_chapters(novel_id)
    # Determine if audio available for each chapter
    for ch in chapters:
        ch["has_audio"] = bool(ch.get("audio_path"))
    return templates.TemplateResponse("novel.html", {"request": request, "novel": novel, "chapters": chapters})
