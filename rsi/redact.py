"""Secret redaction for anything an agent wrote: provider keys, bearer tokens, JWTs, private keys."""

from __future__ import annotations

import re
from typing import Any

_KEY_VALUE = re.compile(
    r"""(?<![A-Za-z0-9_])((?:"|')?(?:accessToken|refreshToken|access_token|refresh_token"""
    r"""|api_key|apiKey|x-api-key|x_api_key|authorization|password|passwd|secret|token"""
    r"""|client_secret|private_key|session_token|aws_secret_access_key|anthropic_api_key"""
    r"""|openai_api_key|mistral_api_key|hf_token|github_token|gh_token)(?:"|')?\s*[:=]\s*)"""
    r"""(?:Bearer \[redacted\]|(?:Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]+"""
    r"""|"[^"]*"|'[^']*'|[^\s,}\]"']+)""",
    re.IGNORECASE,
)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Key/value rules run first so a token rule cannot leave a half-matched value behind.
    (_KEY_VALUE, r'\1"[redacted]"'),
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]+"), "sk-ant-[redacted]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[redacted: aws key]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[redacted: github token]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[redacted: hf token]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"), "[redacted: jwt]"),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
            re.DOTALL,
        ),
        "[redacted: private key]",
    ),
    (re.compile(r"(?<=://)[^\s/:@]+:[^\s/@]+(?=@)"), "[redacted]"),
)


def redact(text: str) -> str:
    """Mask provider keys, bearer tokens, JWTs, private keys and credential-style key values."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Apply `redact` to every string inside a JSON-like object."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj
