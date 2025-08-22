"""
Entry point for Vercel.

This module exposes the FastAPI application instance defined in the
`webnovel.main` module.  Vercel's Python runtime will import this file
and look for an object called `app`, which it uses to handle incoming
HTTP requests.  By simply re‑exporting the application here, we avoid
having to run a separate Uvicorn server; Vercel mounts the ASGI
application directly.

Usage:
    The `vercel.json` configuration references this file as the
    destination for all routes.  When deploying with Vercel, ensure
    that your repository contains this `main.py` file at the project
    root and a matching `vercel.json` file.
"""

from webnovel.main import app as app  # noqa: F401  re‑export FastAPI instance

# The FastAPI app is defined in webnovel/main.py.  We import it and
# assign it to `app` so that the Vercel Python runtime can discover
# and serve it.  No additional code is needed here.