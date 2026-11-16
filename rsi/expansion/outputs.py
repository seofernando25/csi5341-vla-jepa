"""Checks on what a discovery agent returned, and host-path redaction of every committed record."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rsi.agents import AgentRunResult
from rsi.paths import REPO_ROOT

# The Mistral openai-completions path can coerce a structured content block to the JS string
# `[object Object]`; a run whose output is mostly that token did not return usable text.
PLACEHOLDER_TOKEN = "[object Object]"
PLACEHOLDER_MIN_TOKENS = 4
PLACEHOLDER_SHARE_LIMIT = 0.5
# A final answer shorter than this cannot describe a proposal; such a fragment is not usable
# text, so the stdout check still runs (the observed failure left a lone "[" behind).
USABLE_TEXT_MIN_CHARS = 16
# Base paths that must never appear in a committed record, with the marker that replaces them.
HOST_PATH_MARKERS: tuple[tuple[Path, str], ...] = (
    (REPO_ROOT, "<repo>"),
    (Path.home(), "<home>"),
    (Path(tempfile.gettempdir()), "<tmp>"),
)


class AgentOutputInvalid(RuntimeError):
    """Raised when every agent in a batch returned placeholder-only output; stops the cycle."""


def placeholder_output_reason(*texts: str | None) -> str | None:
    """Reason when an agent's text is placeholder-only or mostly `[object Object]`, else None."""
    for text in texts:
        if not text:
            continue
        tokens = text.count(PLACEHOLDER_TOKEN)
        if tokens < PLACEHOLDER_MIN_TOKENS:
            continue
        placeholder_chars = tokens * len(PLACEHOLDER_TOKEN)
        rest = sum(1 for char in text.replace(PLACEHOLDER_TOKEN, "") if not char.isspace())
        share = placeholder_chars / (placeholder_chars + rest)
        if share >= PLACEHOLDER_SHARE_LIMIT:
            return (
                f"agent output is {share:.0%} `{PLACEHOLDER_TOKEN}` placeholders "
                f"({tokens} tokens): the backend returned no usable text"
            )
    return None


def _usable_text(text: str) -> bool:
    """True when an agent's final text carries more than a placeholder fragment."""
    rest = text.replace(PLACEHOLDER_TOKEN, "")
    return sum(1 for char in rest if not char.isspace()) >= USABLE_TEXT_MIN_CHARS


def agent_output_reason(res: AgentRunResult | None) -> str | None:
    """Placeholder reason for one run: the final text decides, the stream is only a fallback.

    Placeholder-heavy thinking or tool output does not invalidate a usable final answer. The
    stdout tail is consulted only when the final text is missing, a fragment, or placeholders.
    """
    if res is None:
        return None
    reason = placeholder_output_reason(res.text)
    if reason is not None:
        return reason
    if _usable_text(res.text):
        return None
    return placeholder_output_reason(res.stdout)


def redact_host_paths(text: str, *bases: tuple[Path, str]) -> str:
    """Replace each absolute base path in `text` with its marker, longest path first."""
    out = text
    for path, marker in sorted(bases, key=lambda item: -len(str(item[0]))):
        if str(path) != "/":
            out = out.replace(str(path), marker)
    return out


def redact_node_paths(text: str, node_dir: Path) -> str:
    """Replace absolute node, cycle, study and host paths in a message with markers.

    The node directory is `<study>/cycles/<cycle>/nodes/<cell>`, so the enclosing paths are
    derived from it rather than passed in.
    """
    node = Path(node_dir).resolve()
    bases = [(node, "<node>"), *HOST_PATH_MARKERS]
    cycle = node.parent.parent
    if cycle.parent.name == "cycles":
        bases.extend([(cycle, "<cycle>"), (cycle.parent.parent, "<study>")])
    return redact_host_paths(text, *bases)
