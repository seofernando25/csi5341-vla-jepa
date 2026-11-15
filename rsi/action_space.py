"""Declarative action space, shared recipe builder, and strict proposal validator."""

from __future__ import annotations

import re
from typing import Any

CANDIDATE_KEYS = frozenset({"id", "base", "knobs", "hypothesis", "expected_resident_mb"})
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
MULTI_SENTENCE_RE = re.compile(r"(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\d)[.!?]\s+[A-Z]")

ACTION_SPACE: dict[str, Any] = {
    "bases": {
        "T2": {
            "vocab": "restricted",
            "embedding": "compact_embedding",
            "vision_quant": "hqq",
            "vision_bits": [2, 3],
            "group_size": [32, 64],
            "action_precision": ["hqq2_nf4", "bf16_attn_hqq2_mlp", "bf16_full"],
            "solver_steps": [1, 4],
            "description": "Restricted LIBERO-Spatial vocabulary (compact_embedding) + NF4 text/merger.",
        },
        "C1": {
            "vocab": "full",
            "embedding": "int8_embedding",
            "vision_quant": "hqq",
            "vision_bits": [2, 3],
            "group_size": [32, 64],
            "action_precision": ["hqq2_nf4", "bf16_attn_hqq2_mlp", "bf16_full"],
            "solver_steps": [1, 4],
            "description": "Full vocabulary (int8_embedding) + NF4 text/merger + HQQ vision.",
        },
        "C2": {
            "vocab": "full",
            "embedding": "int8_embedding",
            "vision_quant": "bnb_nf4",
            "vision_bits": [4],
            "group_size": None,
            "action_precision": ["nf4"],
            "solver_steps": [1],
            "description": "Full vocabulary NF4 everywhere (text/vision/merger/action) + 1-step solver (no graph engine).",
        },
    },
    "knobs": {
        "text_mlp_prune": {
            "type": "float",
            "min": 0.0,
            "max": 0.30,
            "description": "Fraction of text MLP hidden channels pruned (0.0 omits pruning).",
        },
        "vision_mlp_prune": {
            "type": "float",
            "min": 0.0,
            "max": 0.60,
            "description": "Fraction of vision MLP hidden channels pruned (0.0 omits pruning).",
        },
        "group_size": {
            "type": "int|null",
            "allowed": [32, 64],
            "description": "HQQ group size (32 or 64 for T2/C1; omit or null for C2 NF4).",
        },
        "vision_bits": {
            "type": "int",
            "allowed": [2, 3, 4],
            "description": "Vision block bits: 2 (HQQ axis=0) or 3 (HQQ axis=1, g64) on T2/C1; 4 (NF4) on C2.",
        },
        "action_precision": {
            "type": "str",
            "allowed": ["hqq2_nf4", "bf16_attn_hqq2_mlp", "bf16_full", "nf4"],
            "description": (
                "hqq2_nf4 (HQQ2 MLP + NF4 attn/interfaces), "
                "bf16_attn_hqq2_mlp (BF16 attn/interfaces + HQQ2 MLP), "
                "bf16_full (entire action head BF16), or nf4 (C2 full-action NF4)."
            ),
        },
        "solver_steps": {
            "type": "int",
            "allowed": [1, 4],
            "description": "Flow-matching solver steps (4 default, 1 for C2 one-step; C2 requires 1).",
        },
    },
    "constraints": [
        "Full-vocabulary bases are C1 and C2; at least half of proposed candidates must use C1 or C2.",
        "Base C2 requires vision_bits=4, action_precision='nf4', solver_steps=1, and group_size=null (or omitted).",
        "Bases T2 and C1 require vision_bits in {2, 3}, group_size in {32, 64}, and action_precision in {'hqq2_nf4', 'bf16_attn_hqq2_mlp', 'bf16_full'}.",
        "When vision_bits=3 on T2/C1, vision_blocks uses HQQ axis=1, group_size=64; group_size still sets action_mlp HQQ2.",
    ],
    "excluded_knobs": [
        "layers (CKA block dropping), heads (attention head pruning), visual_tokens (token compaction): inline literals in r3/r5, not parameterized by rsi/live.py helpers.",
        "action_horizon, replanning, conditioning_tokens, compile_linears, engine='graph', wanda/magnitude/sparsegpt/gptq/awq_diagonal/torchao: no builder helper in rsi/live.py.",
    ],
}


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _is_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def build_recipe_changes(
    *,
    base: str,
    text_mlp_prune: float = 0.0,
    vision_mlp_prune: float = 0.0,
    group_size: int | None = 64,
    vision_bits: int = 2,
    action_precision: str = "hqq2_nf4",
    solver_steps: int = 4,
    action_group_size: int | None = None,
) -> list[dict[str, Any]]:
    """Build compression recipe `changes` matching `rsi/live.py` helpers."""
    if base not in ACTION_SPACE["bases"]:
        raise ValueError(f"Unsupported base: {base!r}")
    changes: list[dict[str, Any]] = []
    if text_mlp_prune > 0.0:
        changes.append(
            {"method": "channels", "scope": "text_mlp", "fraction": float(text_mlp_prune)}
        )
    if vision_mlp_prune > 0.0:
        changes.append(
            {"method": "channels", "scope": "vision_mlp", "fraction": float(vision_mlp_prune)}
        )
    changes.append({"method": "world_remove"})
    changes.append({"method": "compact_embedding" if base == "T2" else "int8_embedding"})

    if base == "C2":
        changes.append({"method": "solver_steps", "count": 1})
        for scope in ("text", "vision_blocks", "merger", "action"):
            changes.append({"method": "bnb", "scope": scope, "bits": 4})
        return changes

    if group_size is None:
        raise ValueError(f"Base {base!r} requires group_size in {{32, 64}}")
    changes.append({"method": "bnb", "scope": "text", "bits": 4})
    if vision_bits == 2:
        changes.append(
            {
                "method": "hqq",
                "scope": "vision_blocks",
                "bits": 2,
                "axis": 0,
                "group_size": int(group_size),
                "backend": "pytorch",
            }
        )
    elif vision_bits == 3:
        changes.append(
            {
                "method": "hqq",
                "scope": "vision_blocks",
                "bits": 3,
                "axis": 1,
                "group_size": 64,
                "backend": "pytorch",
            }
        )
    else:
        raise ValueError(f"Base {base!r} requires vision_bits in {{2, 3}}, got {vision_bits!r}")

    act_gs = int(action_group_size if action_group_size is not None else group_size)
    if action_precision in {"hqq2_nf4", "bf16_attn_hqq2_mlp"}:
        changes.append(
            {
                "method": "hqq",
                "scope": "action_mlp",
                "bits": 2,
                "axis": 0,
                "group_size": act_gs,
                "backend": "pytorch",
            }
        )
    if action_precision == "hqq2_nf4":
        changes.append({"method": "bnb", "scope": "action_attention", "bits": 4})
        changes.append({"method": "bnb", "scope": "action_interfaces", "bits": 4})
    elif action_precision not in {"bf16_attn_hqq2_mlp", "bf16_full"}:
        raise ValueError(f"Unsupported action_precision for {base!r}: {action_precision!r}")

    changes.append({"method": "bnb", "scope": "merger", "bits": 4})
    if solver_steps == 1:
        changes.append({"method": "solver_steps", "count": 1})
    elif solver_steps != 4:
        raise ValueError(f"solver_steps must be 1 or 4, got {solver_steps!r}")
    return changes


def extract_knobs_from_changes(changes: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Summarize a recipe's `changes` list into `(base, knobs_dict)` for digests."""
    methods = {c.get("method") for c in changes}
    scopes_bnb = {c.get("scope") for c in changes if c.get("method") == "bnb"}
    has_hqq = "hqq" in methods
    if "compact_embedding" in methods:
        base = "T2" if has_hqq else "T1/T2"
    elif "int8_embedding" in methods:
        base = "C1" if has_hqq else "C2"
    else:
        base = "other"

    text_prune = 0.0
    vision_prune = 0.0
    solver = 4
    vision_bits: int | None = None
    vision_gs: int | None = None
    action_mlp_hqq = False
    action_gs: int | None = None
    extras: list[str] = []

    for c in changes:
        m = c.get("method")
        s = c.get("scope")
        if m == "channels" and s == "text_mlp":
            text_prune = round(float(c.get("fraction", 0.0)), 6)
        elif m == "channels" and s == "vision_mlp":
            vision_prune = round(float(c.get("fraction", 0.0)), 6)
        elif m == "hqq" and s == "vision_blocks":
            vision_bits = int(c["bits"])
            vision_gs = int(c.get("group_size", 64))
        elif m == "bnb" and s == "vision_blocks":
            vision_bits = int(c["bits"])
            if base == "T2":
                base = "T1/T2-hybrid"
        elif m == "hqq" and s == "action_mlp":
            action_mlp_hqq = True
            action_gs = int(c.get("group_size", 64))
        elif m == "solver_steps":
            solver = int(c["count"])
        elif m == "layers":
            extras.append(f"layers:{s}={c.get('count')}({c.get('criterion', '')})")
        elif m == "heads":
            extras.append(f"heads:{s}={c.get('fraction')}")
        elif m == "visual_tokens":
            extras.append(f"visual_tokens={c.get('fraction')}({c.get('criterion', '')})")
        elif m not in {"world_remove", "compact_embedding", "int8_embedding", "bnb"}:
            extras.append(str(c))

    if "action" in scopes_bnb:
        action_precision = "nf4"
    elif action_mlp_hqq and "action_attention" in scopes_bnb:
        action_precision = "hqq2_nf4"
    elif action_mlp_hqq:
        action_precision = "bf16_attn_hqq2_mlp"
    else:
        action_precision = "bf16_full"

    knobs: dict[str, Any] = {
        "text_mlp_prune": text_prune,
        "vision_mlp_prune": vision_prune,
        "vision_bits": vision_bits,
        "group_size": action_gs if action_gs is not None else vision_gs,
        "action_precision": action_precision,
        "solver_steps": solver,
    }
    if vision_gs is not None and action_gs is not None and vision_gs != action_gs:
        knobs["vision_group_size"] = vision_gs
    if extras:
        knobs["extras"] = extras
    return base, knobs


def canonical_signature_from_changes(changes: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Order-independent canonical signature of a recipe's `changes` for dedup."""
    base, knobs = extract_knobs_from_changes(changes)
    return (
        base,
        round(float(knobs["text_mlp_prune"]), 6),
        round(float(knobs["vision_mlp_prune"]), 6),
        knobs["vision_bits"],
        knobs["group_size"],
        knobs.get("vision_group_size", knobs["group_size"]),
        knobs["action_precision"],
        knobs["solver_steps"],
        tuple(knobs.get("extras", ())),
    )


def canonical_signature_from_candidate(base: str, knobs: dict[str, Any]) -> tuple[Any, ...]:
    """Compute canonical signature of a validated `{base, knobs}` candidate."""
    changes = build_recipe_changes(
        base=base,
        text_mlp_prune=float(knobs["text_mlp_prune"]),
        vision_mlp_prune=float(knobs["vision_mlp_prune"]),
        group_size=knobs.get("group_size"),
        vision_bits=int(knobs["vision_bits"]),
        action_precision=str(knobs["action_precision"]),
        solver_steps=int(knobs["solver_steps"]),
    )
    return canonical_signature_from_changes(changes)


def validate_candidate(cand: Any, index: int) -> dict[str, Any]:
    """Validate one candidate dict strictly; return normalized copy or raise ValueError."""
    prefix = f"candidate[{index}]"
    if not isinstance(cand, dict):
        raise TypeError(f"{prefix}: expected object, got {type(cand).__name__}")
    unknown = sorted(set(cand) - CANDIDATE_KEYS)
    if unknown:
        raise ValueError(f"{prefix}: unknown keys {unknown}")
    missing = sorted(CANDIDATE_KEYS - set(cand))
    if missing:
        raise ValueError(f"{prefix}: missing required keys {missing}")

    cid = cand["id"]
    if not isinstance(cid, str) or not ID_RE.match(cid):
        raise ValueError(
            f"{prefix}: invalid id {cid!r}; must match {ID_RE.pattern} (alphanumeric, '-' or '_')"
        )
    prefix = f"candidate[{index}] ({cid})"

    base = cand["base"]
    if not isinstance(base, str) or base not in ACTION_SPACE["bases"]:
        raise ValueError(f"{prefix}: invalid base {base!r}; expected one of ['T2', 'C1', 'C2']")

    hyp = cand["hypothesis"]
    if not isinstance(hyp, str) or not hyp.strip():
        raise ValueError(f"{prefix}: hypothesis must be a non-empty string")
    hyp_clean = hyp.strip()
    if "\n" in hyp_clean or len(hyp_clean) > 400 or MULTI_SENTENCE_RE.search(hyp_clean):
        raise ValueError(f"{prefix}: hypothesis must be a single concise sentence (<= 400 chars)")

    mb = cand["expected_resident_mb"]
    if not _is_number(mb) or not (100.0 <= float(mb) <= 6000.0):
        raise ValueError(f"{prefix}: expected_resident_mb {mb!r} out of range [100.0, 6000.0]")

    knobs = cand["knobs"]
    if not isinstance(knobs, dict):
        raise TypeError(f"{prefix}: knobs must be an object")

    allowed_knob_keys = set(ACTION_SPACE["knobs"])
    unknown_knobs = sorted(set(knobs) - allowed_knob_keys)
    if unknown_knobs:
        raise ValueError(f"{prefix}: unknown knob keys {unknown_knobs}")

    required_knobs = {
        "text_mlp_prune",
        "vision_mlp_prune",
        "vision_bits",
        "action_precision",
        "solver_steps",
    }
    if base in {"T2", "C1"}:
        required_knobs.add("group_size")
    missing_knobs = sorted(required_knobs - set(knobs))
    if missing_knobs:
        raise ValueError(f"{prefix}: missing knob keys {missing_knobs} for base {base!r}")

    tp = knobs["text_mlp_prune"]
    if not _is_number(tp) or not (0.0 <= float(tp) <= 0.30):
        raise ValueError(f"{prefix}: text_mlp_prune {tp!r} out of range [0.0, 0.30]")

    vp = knobs["vision_mlp_prune"]
    if not _is_number(vp) or not (0.0 <= float(vp) <= 0.60):
        raise ValueError(f"{prefix}: vision_mlp_prune {vp!r} out of range [0.0, 0.60]")

    base_spec = ACTION_SPACE["bases"][base]
    vb = knobs["vision_bits"]
    if not _is_int(vb) or vb not in base_spec["vision_bits"]:
        raise ValueError(
            f"{prefix}: vision_bits {vb!r} not allowed for base {base!r} (allowed: {base_spec['vision_bits']})"
        )

    ap = knobs["action_precision"]
    if not isinstance(ap, str) or ap not in base_spec["action_precision"]:
        raise ValueError(
            f"{prefix}: action_precision {ap!r} not allowed for base {base!r} (allowed: {base_spec['action_precision']})"
        )

    ss = knobs["solver_steps"]
    if not _is_int(ss) or ss not in base_spec["solver_steps"]:
        raise ValueError(
            f"{prefix}: solver_steps {ss!r} not allowed for base {base!r} (allowed: {base_spec['solver_steps']})"
        )

    gs = knobs.get("group_size")
    if base == "C2":
        if gs is not None:
            raise ValueError(
                f"{prefix}: group_size must be null or omitted for base 'C2', got {gs!r}"
            )
    else:
        if not _is_int(gs) or gs not in base_spec["group_size"]:
            raise ValueError(
                f"{prefix}: group_size {gs!r} not allowed for base {base!r} (allowed: {base_spec['group_size']})"
            )

    norm_knobs: dict[str, Any] = {
        "text_mlp_prune": round(float(tp), 6),
        "vision_mlp_prune": round(float(vp), 6),
        "group_size": int(gs) if gs is not None else None,
        "vision_bits": int(vb),
        "action_precision": ap,
        "solver_steps": int(ss),
    }
    return {
        "id": cid,
        "base": base,
        "knobs": norm_knobs,
        "hypothesis": hyp_clean,
        "expected_resident_mb": round(float(mb), 1),
    }


def validate_proposal(
    raw_candidates: Any,
    max_n: int,
    probed_ids: set[str] | None = None,
    probed_signatures: dict[tuple[Any, ...], str] | None = None,
    require_half_full_vocab: bool = True,
) -> list[dict[str, Any]]:
    """Validate a full proposal list strictly; return normalized candidates or raise ValueError."""
    if not isinstance(raw_candidates, list):
        raise TypeError(
            f"proposal must be a JSON list of candidates, got {type(raw_candidates).__name__}"
        )
    if not raw_candidates:
        raise ValueError("proposal must contain at least 1 candidate")
    if len(raw_candidates) > max_n:
        raise ValueError(
            f"proposal has {len(raw_candidates)} candidates, exceeding requested count {max_n}"
        )

    probed_ids = probed_ids or set()
    probed_signatures = probed_signatures or {}
    seen_ids: set[str] = set()
    seen_sigs: dict[tuple[Any, ...], str] = {}
    normalized: list[dict[str, Any]] = []

    for i, raw in enumerate(raw_candidates):
        cand = validate_candidate(raw, i)
        cid = cand["id"]
        if cid in probed_ids:
            raise ValueError(f"candidate[{i}] ({cid}): id duplicates already-probed cell '{cid}'")
        if cid in seen_ids:
            raise ValueError(f"candidate[{i}] ({cid}): duplicate id within proposal")
        seen_ids.add(cid)

        sig = canonical_signature_from_candidate(cand["base"], cand["knobs"])
        if sig in probed_signatures:
            raise ValueError(
                f"candidate[{i}] ({cid}): configuration duplicates already-probed cell '{probed_signatures[sig]}'"
            )
        if sig in seen_sigs:
            raise ValueError(
                f"candidate[{i}] ({cid}): configuration duplicates earlier proposal candidate '{seen_sigs[sig]}'"
            )
        seen_sigs[sig] = cid
        normalized.append(cand)

    if require_half_full_vocab:
        full_vocab = sum(
            1 for c in normalized if ACTION_SPACE["bases"][c["base"]]["vocab"] == "full"
        )
        if full_vocab * 2 < len(normalized):
            raise ValueError(
                f"at least half of candidates must use a full-vocabulary base (C1 or C2); "
                f"got {full_vocab}/{len(normalized)}"
            )
    return normalized
