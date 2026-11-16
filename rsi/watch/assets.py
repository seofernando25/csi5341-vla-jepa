"""The watch page's static files: an allowlist of root files plus `css/` and `js/` modules."""

from __future__ import annotations

from pathlib import Path

from rsi.watch.files import PART

STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
STATIC_DIRS = ("css", "js")
STATIC_FILES = ("index.html", "fixture.json", "fixture_events.json", "fixture_story.json")
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _static_root() -> Path:
    """The directory the page is served from; tests patch STATIC_INDEX to a temp copy."""
    return STATIC_INDEX.parent.resolve()


def static_path(url_path: str) -> Path | None:
    """Resolve one allowlisted static path, refusing traversal, dotfiles and symlinks.

    The page is a set of ES modules under `css/` and `js/` plus three root files. Anything
    else is not served, even if it exists: the static tree is not a general file server.
    """
    relative = url_path.lstrip("/")
    if not relative or relative.endswith("/"):
        return None
    parts = relative.split("/")
    if len(parts) > 3 or any(
        not PART.match(p) or p in (".", "..") or p.startswith(".") for p in parts
    ):
        return None
    if len(parts) == 1 and parts[0] not in STATIC_FILES:
        return None
    if len(parts) > 1 and parts[0] not in STATIC_DIRS:
        return None
    root = _static_root()
    candidate = root / relative
    try:
        real = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not real.is_relative_to(root) or not real.is_file() or candidate.is_symlink():
        return None
    return real


def static_response(url_path: str) -> tuple[int, bytes, str] | None:
    """(status, body, content type) for one static request, or None when it is not served."""
    if url_path in ("/", "/index.html"):
        if STATIC_INDEX.is_file():
            return 200, STATIC_INDEX.read_bytes(), STATIC_TYPES[".html"]
        hint = "<p>Watch page not built: rsi/watch/static/index.html is missing.</p>"
        return 404, f"<!doctype html><title>watch</title>{hint}".encode(), STATIC_TYPES[".html"]
    path = static_path(url_path)
    if path is None:
        return None
    ctype = STATIC_TYPES.get(path.suffix)
    if ctype is None:
        return None
    try:
        return 200, path.read_bytes(), ctype
    except OSError:
        return None
