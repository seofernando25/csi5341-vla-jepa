"""Read-only live watch of a Dream-RSI study.

`state` builds the page's state payload (with `nodes` and `policy`), `source` resolves one node
or policy step to its artifacts and transcript, `transcript` turns agent logs into events,
`context` and `dream` serve the side panels, and `server` (with `stream` and `assets`) serves
it all on loopback. The page itself lives in `static/`.
"""

from rsi.watch.assets import static_path
from rsi.watch.context import context_payload
from rsi.watch.dream import dream_payload
from rsi.watch.server import make_server, run_watch
from rsi.watch.source import Source, SourceError, events_payload, parse_source, read_artifact
from rsi.watch.state import build_state
from rsi.watch.transcript import read_events

__all__ = [
    "Source",
    "SourceError",
    "build_state",
    "context_payload",
    "dream_payload",
    "events_payload",
    "make_server",
    "parse_source",
    "read_artifact",
    "read_events",
    "run_watch",
    "static_path",
]
