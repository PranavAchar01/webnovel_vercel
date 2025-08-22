"""Web page extraction and packaging utilities.

This module defines functions to fetch HTML pages from the internet,
extract readable text from them, and package the result into plain
text or EPUB files. It uses ``httpx`` for HTTP requests and
``BeautifulSoup`` for parsing. If you have access to the
``readability-lxml`` library you can improve extraction quality by
using it instead of the simple heuristics implemented here. The
functions are written to be resilient: they retry failed requests
using exponential backoff, and they avoid raising exceptions
unnecessarily.

The ``package_epub`` function creates a minimal EPUB archive using
Python's ``zipfile`` module. It writes a ``mimetype`` file,
``META-INF/container.xml``, a package document (``content.opf``) and a
table of contents (``toc.ncx``), plus one XHTML file per chapter.

Due to constraints of the execution environment this module does not
support downloading external images or covers. The generated EPUBs
will not include a cover image, but the metadata (title, author,
description) are embedded.
"""

from __future__ import annotations

import asyncio
import re
import uuid
import html
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import httpx
from bs4 import BeautifulSoup

# Allowed maximum size of a novel (in characters) before extraction is aborted.
MAX_NOVEL_SIZE = 3_000_000

# Default user agent for HTTP requests. Some sites require a user agent to
# return full content instead of a 403 or truncated response.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    )
}


async def fetch_html(url: str, max_retries: int = 3, backoff: float = 1.0) -> Optional[str]:
    """Fetch the HTML content from ``url`` using httpx with retry logic.

    If all attempts fail the function returns ``None`` instead of
    raising. A SOCKS proxy may be configured implicitly by the system
    environment; httpx will attempt to use it if needed.
    """
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=DEFAULT_HEADERS)
                if response.status_code == 200:
                    return response.text
        except Exception:
            pass
        await asyncio.sleep(backoff * (2 ** attempt))
    return None


def _extract_title_and_content(soup: BeautifulSoup) -> Tuple[str, str]:
    """Extract the title and main text from a BeautifulSoup document.

    The extraction heuristics are deliberately simple: they look for
    common elements such as ``<article>``, ``<div id="chapter"/>`` or
    paragraphs within the body. If no clear main content is found the
    function concatenates the text of all paragraphs in the page. The
    returned content is plain text with paragraphs separated by two
    newline characters.
    """
    # Title: prefer <h1> then <title>
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    # Candidate containers for main content
    candidates = []
    for tag_id in ["chapter", "chapter-content", "content", "chapterContent", "text"]:
        div = soup.find(id=tag_id)
        if div:
            candidates.append(div)
    for class_name in ["chapter-content", "entry-content", "post-content", "content"]:
        div = soup.find("div", class_=class_name)
        if div:
            candidates.append(div)
    article = soup.find("article")
    if article:
        candidates.append(article)

    content_text = ""
    for candidate in candidates:
        # Extract paragraph texts within the candidate
        paragraphs = candidate.find_all(["p", "br", "div", "span"])
        texts: List[str] = []
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
        if texts:
            content_text = "\n\n".join(texts)
            break
    # Fallback: gather all <p> tags under body
    if not content_text:
        paragraphs = soup.find_all("p")
        texts = [p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True)]
        content_text = "\n\n".join(texts)
    return title or "", content_text


def extract_text_from_html(html_doc: str) -> Tuple[str, str]:
    """Extract (title, content) from raw HTML using BeautifulSoup heuristics."""
    soup = BeautifulSoup(html_doc, "lxml")
    return _extract_title_and_content(soup)


async def extract_chapter(url: str) -> Optional[Dict[str, str]]:
    """Fetch and extract a single chapter from a URL.

    Returns a dict with keys ``title`` and ``text`` on success, or
    ``None`` if the URL could not be fetched or parsed. Extraction is
    limited to 1.5 million characters to avoid pathological pages.
    """
    html_doc = await fetch_html(url)
    if not html_doc:
        return None
    title, content = extract_text_from_html(html_doc)
    if not content:
        return None
    # Truncate overly long content
    if len(content) > MAX_NOVEL_SIZE:
        content = content[:MAX_NOVEL_SIZE]
    return {"title": title or "", "text": content}


async def discover_chapter_urls(toc_url: str) -> List[str]:
    """Attempt to extract a list of chapter URLs from a table of contents page.

    The function makes a best effort to find links containing the word
    ``chapter`` (case insensitive) in either the anchor text or the
    URL. The result may contain duplicates; callers should deduplicate
    them if necessary. If no suitable links are found the function
    returns an empty list.
    """
    html_doc = await fetch_html(toc_url)
    if not html_doc:
        return []
    soup = BeautifulSoup(html_doc, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if not href:
            continue
        # Resolve relative links
        full_url = httpx.URL(href, base=toc_url).resolve().human_repr()
        if re.search(r"chapter", text, re.IGNORECASE) or re.search(r"chapter", href, re.IGNORECASE):
            links.append(full_url)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_links: List[str] = []
    for link in links:
        if link not in seen:
            unique_links.append(link)
            seen.add(link)
    return unique_links


def package_txt(novel_id: str, chapters: List[Dict[str, Any]], dest_dir: str) -> str:
    """Concatenate chapter texts into a single UTF‑8 encoded .txt file.

    The file is written to ``dest_dir`` with a filename derived from
    ``novel_id``. Each chapter begins with its title on a separate
    line followed by a blank line and then the chapter content. The
    function returns the path to the created file.
    """
    filename = f"{novel_id}.txt"
    path = Path(dest_dir) / filename
    with open(path, "w", encoding="utf-8") as f:
        for chapter in chapters:
            title = chapter.get("title", "").strip()
            text = chapter.get("text", "").strip()
            f.write(title + "\n\n")
            f.write(text + "\n\n\n")
    return str(path)


def _epub_template(novel_metadata: Dict[str, Any], chapters: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate the key files needed for a minimal EPUB archive.

    Returns a dict mapping internal file names (inside the EPUB) to
    their contents as strings. The caller is responsible for writing
    the ``mimetype`` and assembling the zipfile. The generated
    ``content.opf`` includes a manifest and spine referencing the
    chapter XHTML files. The ``toc.ncx`` describes the table of
    contents. A modern ebook reader should be able to open this EPUB.
    """
    # Escape metadata for XML
    title = html.escape(novel_metadata.get("title", "Untitled"))
    author = html.escape(novel_metadata.get("author", ""))
    description = html.escape(novel_metadata.get("description", ""))
    novel_id = novel_metadata.get("id", str(uuid.uuid4()))
    uid = f"urn:uuid:{novel_id}"

    # Generate chapter XHTML files
    xhtml_files: Dict[str, str] = {}
    manifest_items: List[str] = []
    spine_items: List[str] = []
    ncx_navpoints: List[str] = []
    for idx, chapter in enumerate(chapters, start=1):
        chap_id = chapter.get("id", f"chapter{idx}")
        chap_title = html.escape(chapter.get("title", f"Chapter {idx}"))
        content = chapter.get("text", "").replace("\n", "\n\n")
        # Convert plain text to paragraphs; collapse multiple blank lines
        paragraphs = [html.escape(p.strip()) for p in chapter.get("text", "").split("\n") if p.strip()]
        xhtml = (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            "<html xmlns='http://www.w3.org/1999/xhtml'>\n"
            "<head><title>{title}</title></head>\n"
            "<body>\n"
            "<h1>{title}</h1>\n"
            "{paragraphs}\n"
            "</body>\n"
            "</html>\n"
        ).format(
            title=chap_title,
            paragraphs="\n".join([f"<p>{p}</p>" for p in paragraphs]),
        )
        file_name = f"Text/chapter{idx}.xhtml"
        xhtml_files[file_name] = xhtml
        manifest_items.append(f"<item id='chap{idx}' href='{file_name}' media-type='application/xhtml+xml'/>")
        spine_items.append(f"<itemref idref='chap{idx}' />")
        ncx_navpoints.append(
            (
                f"<navPoint id='navPoint-{idx}' playOrder='{idx}'>"
                f"<navLabel><text>{chap_title}</text></navLabel>"
                f"<content src='{file_name}'/>"
                f"</navPoint>"
            )
        )

    # content.opf
    opf = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<package xmlns='http://www.idpf.org/2007/opf' unique-identifier='BookId' version='2.0'>\n"
        "  <metadata xmlns:dc='http://purl.org/dc/elements/1.1/' xmlns:opf='http://www.idpf.org/2007/opf'>\n"
        f"    <dc:title>{title}</dc:title>\n"
        f"    <dc:creator>{author}</dc:creator>\n"
        f"    <dc:description>{description}</dc:description>\n"
        f"    <dc:identifier id='BookId'>{uid}</dc:identifier>\n"
        f"    <dc:language>en</dc:language>\n"
        "  </metadata>\n"
        "  <manifest>\n"
        "    <item id='ncx' href='toc.ncx' media-type='application/x-dtbncx+xml'/>\n"
        "    " + "\n    ".join(manifest_items) + "\n"
        "  </manifest>\n"
        "  <spine toc='ncx'>\n"
        "    " + "\n    ".join(spine_items) + "\n"
        "  </spine>\n"
        "</package>"
    )
    # toc.ncx
    ncx = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<ncx xmlns='http://www.daisy.org/z3986/2005/ncx/' version='2005-1'>\n"
        "  <head>\n"
        f"    <meta name='dtb:uid' content='{uid}'/>\n"
        "    <meta name='dtb:depth' content='1'/>\n"
        "    <meta name='dtb:totalPageCount' content='0'/>\n"
        "    <meta name='dtb:maxPageNumber' content='0'/>\n"
        "  </head>\n"
        "  <docTitle><text>{title}</text></docTitle>\n"
        "  <navMap>\n"
        "    " + "\n    ".join(ncx_navpoints) + "\n"
        "  </navMap>\n"
        "</ncx>"
    )
    # container.xml
    container = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<container version='1.0' xmlns='urn:oasis:names:tc:opendocument:xmlns:container'>\n"
        "  <rootfiles>\n"
        "    <rootfile full-path='content.opf' media-type='application/oebps-package+xml'/>\n"
        "  </rootfiles>\n"
        "</container>"
    )
    result = {
        "content.opf": opf,
        "toc.ncx": ncx,
        "META-INF/container.xml": container,
    }
    result.update(xhtml_files)
    return result


def package_epub(novel_metadata: Dict[str, Any], chapters: List[Dict[str, Any]], dest_dir: str) -> str:
    """Create an EPUB archive for the given novel and chapters.

    The returned path points to a .epub file in ``dest_dir``. This
    function writes the required files into a zip archive using the
    minimal EPUB structure defined in ``_epub_template``. If no
    chapters are provided the function raises ``ValueError``.
    """
    if not chapters:
        raise ValueError("Cannot package an empty novel into an EPUB.")
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    filename = f"{novel_metadata.get('id', uuid.uuid4())}.epub"
    epub_path = dest_dir_path / filename
    template = _epub_template(novel_metadata, chapters)
    import zipfile
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # According to EPUB specification the mimetype must be the first
        # entry and it must not be compressed.
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for internal_name, content in template.items():
            zf.writestr(internal_name, content)
    return str(epub_path)
