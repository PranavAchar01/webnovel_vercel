"""Web novel extractor and audiobook service.

This package contains modules that implement a FastAPI based service for
extracting web novels, packaging them into plain text and EPUB formats,
generating audio using a simple text‑to‑speech adapter, indexing the
content with Whoosh for search, and serving a minimal web UI for
discovering and listening to novels.

The modules in this package are:

* ``db.py`` – Functions for creating and interacting with the SQLite
  database. This includes helper functions for inserting and updating
  novels, chapters and jobs.

* ``extractor.py`` – Functions responsible for downloading web pages,
  extracting readable text from them and creating plain text and
  EPUB files. The extraction routines use the built‑in ``bs4``
  library together with simple heuristics to identify chapter content.

* ``tts.py`` – A simple text‑to‑speech adapter. In the absence of
  external services or libraries this module includes a silent MP3
  payload encoded as base64. For each chunk of text a copy of this
  silent MP3 is written; concatenating several copies produces
  multi‑second audio files. In a production setting you would
  implement adapters for services such as Google Cloud TTS or
  ElevenLabs.

* ``search.py`` – Functions for creating and updating a Whoosh search
  index and executing queries. The search index stores metadata
  about novels and supports boosted fields for better ranking.

* ``main.py`` – The FastAPI application itself. It defines the
  endpoints described in the user specification, wires together the
  extraction, TTS, search and database layers, and renders Jinja2
  templates to serve a minimal web UI.

The service is designed to run on platforms such as Replit where
external dependencies may be unavailable. As such the code avoids
depending on packages that are not pre‑installed. This means that
certain features (such as AI driven TTS) are implemented as simple
placeholders. You can replace these components with real
implementations when deploying to an environment with the necessary
libraries and credentials.
"""
