"""Proposal step for Dream-RSI: LLM prompt/transport, strict validation, and journal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.action_space import ACTION_SPACE, build_recipe_changes, validate_proposal
from rsi.digest import (
    DREAM_DIR,
    MEASUREMENTS_DIR,
    NUMBERS_PATH,
    collect_history,
    history_digest,
)
from rsi.qwen_search import Stage2Candidate

PROPOSALS_DIR = DREAM_DIR / "proposals"
REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = REPO_ROOT / ".env"

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class TransportResponseError(RuntimeError):
    """Raised when the LLM transport returns an unusable response (e.g. empty or truncated)."""


def load_dotenv(path: Path = DOTENV_PATH) -> dict[str, str]:
    """Load `KEY=VALUE` lines from a dotenv file; real env vars take precedence."""
    if not path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val
            loaded[key] = val
    return loaded


def _scrub_secret(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def build_prompt(
    round_id: str,
    n: int,
    dream_dir: Path | str = DREAM_DIR,
    measurements_dir: Path | str = MEASUREMENTS_DIR,
    numbers_path: Path | str = NUMBERS_PATH,
) -> str:
    """Build the proposer prompt from instructions, action space, and history digest."""
    digest_str = history_digest(dream_dir, measurements_dir, numbers_path)
    space_str = json.dumps(ACTION_SPACE, indent=2)
    min_full = (n + 1) // 2
    return (
        f"You are proposing {n} compression candidates for Dream-RSI round '{round_id}' "
        f"(VLA-JEPA Qwen3-VL LIBERO-Spatial policy).\n\n"
        "Instructions:\n"
        f"1. Propose {n} distinct candidates that trade resident memory against LIBERO success and latency.\n"
        f"2. At least {min_full} of the {n} candidates (at least half) MUST use a full-vocabulary base "
        "('C1' or 'C2') so results are comparable with the C2 baseline.\n"
        "3. State one concrete, falsifiable one-sentence hypothesis per candidate.\n"
        "4. Do NOT reuse any probed cell id or duplicate any already-probed (base, knobs) configuration.\n"
        "5. Output ONLY a JSON array `[...]` where each element has exact keys:\n"
        '   {"id": "<slug>", "base": "T2"|"C1"|"C2", "knobs": {...}, '
        '"hypothesis": "<one sentence>", "expected_resident_mb": <float>}\n\n'
        "Bounded Action Space (JSON):\n"
        f"{space_str}\n\n"
        "Probed History Digest (JSON):\n"
        f"{digest_str}\n\n"
        f"Return ONLY the JSON array of {n} candidates (no prose outside JSON)."
    )


def _scan_json_list(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    fallback: Any = None
    pos = 0
    while True:
        idx = text.find("[", pos)
        if idx == -1:
            return fallback
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj):
                return obj
            if fallback is None:
                fallback = obj
        pos = idx + 1


def _extract_json_array(raw_reply: str) -> Any:
    text = raw_reply.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    for match in FENCE_RE.finditer(text):
        found = _scan_json_list(match.group(1).strip())
        if found is not None:
            return found
    found = _scan_json_list(text)
    if found is not None:
        return found
    raise ValueError("No JSON array found in proposer reply")


def _resolve_cmd_argv(cmd: str) -> list[str]:
    argv = shlex.split(cmd)
    if not argv:
        raise ValueError("proposer_cmd must not be empty")
    for idx, tok in enumerate(argv):
        if tok.startswith("-") or Path(tok).is_absolute():
            continue
        candidate = REPO_ROOT / tok
        if (idx == 0 and ("/" in tok or os.sep in tok or candidate.exists())) or (
            idx > 0 and ("/" in tok or os.sep in tok) and candidate.exists()
        ):
            argv[idx] = str(candidate.resolve())
    return argv


def _call_command(cmd: str, prompt: str, timeout: float) -> str:
    argv = _resolve_cmd_argv(cmd)
    proc = subprocess.run(
        argv,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()[-400:]
        raise RuntimeError(f"proposer command exited with {proc.returncode}: {err}")
    return proc.stdout


def _call_openai_http(
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout: float,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model, "temperature": float(temperature), "messages": messages}
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except OSError:
            detail = str(exc.reason)
        raise RuntimeError(
            f"HTTP {exc.code} from {endpoint}: {_scrub_secret(detail, api_key)}"
        ) from None
    except urllib.error.URLError as exc:
        reason = _scrub_secret(str(exc.reason), api_key)
        raise RuntimeError(f"HTTP request failed for {endpoint}: {reason}") from None

    payload = json.loads(raw)
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise TransportResponseError("OpenAI-compatible response missing 'choices[0]'")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise TransportResponseError(
            "OpenAI-compatible response truncated (finish_reason='length')"
        )
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise TransportResponseError(
            f"OpenAI-compatible response returned null or empty content (finish_reason={finish_reason!r})"
        )
    meta = {k: payload[k] for k in ("model", "usage") if payload.get(k) is not None}
    return content, meta


def _resolve_transport(
    proposer_cmd: str | None,
    base_url: str | None,
    model: str | None,
) -> tuple[str, dict[str, str]]:
    cmd = (proposer_cmd if proposer_cmd is not None else os.environ.get("RSI_PROPOSER_CMD")) or ""
    url = (base_url if base_url is not None else os.environ.get("RSI_PROPOSER_BASE_URL")) or ""
    mdl = (model if model is not None else os.environ.get("RSI_PROPOSER_MODEL")) or ""
    cmd, url, mdl = cmd.strip(), url.strip(), mdl.strip()

    has_cmd = bool(cmd)
    explicit_http = (base_url is not None and bool(base_url.strip())) or (
        model is not None and bool(model.strip())
    )
    if has_cmd and (bool(url) or explicit_http):
        raise ValueError(
            "Specify exactly one transport: either --proposer-cmd (RSI_PROPOSER_CMD) "
            "or --base-url + --model (RSI_PROPOSER_BASE_URL + RSI_PROPOSER_MODEL), not both."
        )
    if has_cmd:
        return "command", {"proposer_cmd": cmd}
    if url or mdl:
        if not (url and mdl):
            raise ValueError(
                "OpenAI-compatible transport requires both --base-url (RSI_PROPOSER_BASE_URL) "
                "and --model (RSI_PROPOSER_MODEL)."
            )
        return "openai_compatible", {"base_url": url, "model": mdl}
    raise ValueError(
        "No proposer transport configured: pass --proposer-cmd (or RSI_PROPOSER_CMD) "
        "or --base-url + --model (or RSI_PROPOSER_BASE_URL + RSI_PROPOSER_MODEL)."
    )


def propose(
    round_id: str,
    n: int = 6,
    proposer_cmd: str | None = None,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str = "RSI_PROPOSER_API_KEY",
    temperature: float = 0.0,
    timeout: float = 120.0,
    force: bool = False,
    proposals_dir: Path | str = PROPOSALS_DIR,
    dream_dir: Path | str = DREAM_DIR,
    measurements_dir: Path | str = MEASUREMENTS_DIR,
    numbers_path: Path | str = NUMBERS_PATH,
) -> dict[str, Any]:
    """Run one external proposer (command or OpenAI-compatible HTTP), validate with 1 repair retry, and journal."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    transport, tinfo = _resolve_transport(proposer_cmd, base_url, model)

    out_path = Path(proposals_dir) / f"{round_id}.json"
    if out_path.exists() and not force:
        raise FileExistsError(
            f"Proposal journal already exists: {out_path} (pass --force to overwrite)"
        )

    _, probed_ids, probed_sigs = collect_history(dream_dir, measurements_dir, numbers_path)
    prompt = build_prompt(round_id, n, dream_dir, measurements_dir, numbers_path)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    api_key = os.environ.get(api_key_env, "") if transport == "openai_compatible" else ""

    def invoke(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        if transport == "command":
            return _call_command(tinfo["proposer_cmd"], messages[-1]["content"], timeout), {}
        return _call_openai_http(
            tinfo["base_url"], tinfo["model"], api_key, temperature, timeout, messages
        )

    validation_log: list[dict[str, Any]] = []
    raw_reply = ""
    try:
        raw_reply, meta = invoke([{"role": "user", "content": prompt}])
        parsed = _extract_json_array(raw_reply)
        validated = validate_proposal(parsed, n, probed_ids, probed_sigs)
        validation_log.append({"attempt": 1, "valid": True, "errors": []})
    except (TransportResponseError, ValueError, TypeError, json.JSONDecodeError) as first_err:
        validation_log.append(
            {"attempt": 1, "valid": False, "errors": [str(first_err)], "raw_reply": raw_reply}
        )
        repair_note = (
            f"Your previous reply failed:\n{first_err}\n\n"
            f"Fix all errors and return ONLY a valid JSON array of up to {n} candidates."
        )
        if transport == "command":
            prev = f"\n\n### Previous invalid reply\n{raw_reply}" if raw_reply else ""
            repair_msgs = [
                {
                    "role": "user",
                    "content": f"{prompt}{prev}\n\n### Validation error\n{repair_note}",
                }
            ]
        elif raw_reply:
            repair_msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw_reply},
                {"role": "user", "content": repair_note},
            ]
        else:
            repair_msgs = [
                {"role": "user", "content": prompt},
                {"role": "user", "content": repair_note},
            ]
        try:
            raw_reply, meta = invoke(repair_msgs)
            parsed2 = _extract_json_array(raw_reply)
            validated = validate_proposal(parsed2, n, probed_ids, probed_sigs)
        except TransportResponseError as second_err:
            raise TransportResponseError(
                f"Proposal transport failed after 1 repair retry: {second_err}"
            ) from second_err
        except (ValueError, TypeError, json.JSONDecodeError) as second_err:
            raise ValueError(
                f"Proposal validation failed after 1 repair retry: {second_err}"
            ) from second_err
        validation_log.append({"attempt": 2, "valid": True, "errors": []})

    journal: dict[str, Any] = {
        "protocol": "dream-rsi-proposal",
        "round_id": round_id,
        "requested_n": n,
        "transport": transport,
        "prompt_sha256": prompt_sha256,
        "prompt": prompt,
        **(
            {"proposer_cmd": tinfo["proposer_cmd"]}
            if transport == "command"
            else {
                "base_url": tinfo["base_url"],
                "model": tinfo["model"],
                "temperature": float(temperature),
                **({"response_model": meta["model"]} if "model" in meta else {}),
                **({"usage": meta["usage"]} if "usage" in meta else {}),
            }
        ),
        "raw_reply": raw_reply,
        "candidates": validated,
        "validation_log": validation_log,
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(journal, indent=2) + "\n")
    return journal


def candidates_from_journal(
    round_id: str, proposals_dir: Path | str = PROPOSALS_DIR
) -> list[Stage2Candidate]:
    """Convert a validated proposal journal into `Stage2Candidate` objects for `vla-rsi-round`."""
    path = Path(proposals_dir) / f"{round_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No proposal journal for round {round_id!r} at {path}")
    data = json.loads(path.read_text())
    candidates: list[Stage2Candidate] = []
    for idx, cand in enumerate(data["candidates"]):
        cid = cand["id"]
        base = cand["base"]
        knobs = cand["knobs"]
        family = f"{round_id}_{base.lower()}"
        changes = build_recipe_changes(
            base=base,
            text_mlp_prune=float(knobs["text_mlp_prune"]),
            vision_mlp_prune=float(knobs["vision_mlp_prune"]),
            group_size=knobs.get("group_size"),
            vision_bits=int(knobs["vision_bits"]),
            action_precision=str(knobs["action_precision"]),
            solver_steps=int(knobs["solver_steps"]),
        )
        candidates.append(
            Stage2Candidate(
                cell_id=cid,
                branch=idx,
                attempt=0,
                parent_id="root",
                seq=idx,
                family=family,
                recipe={"id": cid, "family": family, "changes": changes},
            )
        )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vla-rsi-propose",
        description="Propose Dream-RSI compression candidates from history digest and action space.",
    )
    parser.add_argument("--round", required=True, help="Target round id (e.g. r7)")
    parser.add_argument("--n", type=int, default=6, help="Number of candidates to propose")
    parser.add_argument(
        "--proposer-cmd", default=None, help="External command reading stdin prompt"
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=None, help="Model name for OpenAI-compatible API")
    parser.add_argument(
        "--api-key-env",
        default="RSI_PROPOSER_API_KEY",
        help="Environment variable holding the API key (empty allowed for local servers)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=120.0, help="Transport timeout (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt and exit")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing journal")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(build_prompt(args.round, args.n))
        return 0

    if not args.proposer_cmd and not os.environ.get("RSI_PROPOSER_CMD"):
        load_dotenv(DOTENV_PATH)

    journal = propose(
        args.round,
        args.n,
        args.proposer_cmd,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        timeout=args.timeout,
        force=args.force,
        proposals_dir=PROPOSALS_DIR,
    )
    print(
        json.dumps(
            {
                "round_id": journal["round_id"],
                "transport": journal["transport"],
                "candidates": [c["id"] for c in journal["candidates"]],
                "journal": str(PROPOSALS_DIR / f"{journal['round_id']}.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
