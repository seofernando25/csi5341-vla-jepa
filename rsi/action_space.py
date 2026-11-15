"""Declarative action space, shared recipe builder, and strict proposal validator."""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Free-form Dream-RSI v2 recipe space & proper physical application order
# ---------------------------------------------------------------------------

RECIPE_TOP_KEYS = frozenset({"id", "family", "changes", "engine", "engine_options"})
CHANNEL_SCOPES = frozenset({"text_mlp", "vision_mlp", "merger", "action_mlp"})
HEAD_SCOPES = frozenset({"text_attention", "vision_attention", "action_attention"})
LAYER_SCOPES = frozenset({"text", "vision", "action"})
LINEAR_SCOPES = frozenset(
    {
        "all",
        "text",
        "vision",
        "action",
        "text_mlp",
        "text_attention",
        "vision_blocks",
        "merger",
        "vision_mlp",
        "vision_attention",
        "action_mlp",
        "action_attention",
        "action_interfaces",
    }
)

_SCOPE_CHILDREN: dict[str, frozenset[str]] = {
    "all": LINEAR_SCOPES - {"all"},
    "text": frozenset({"text_mlp", "text_attention"}),
    "vision": frozenset({"vision_blocks", "merger", "vision_mlp", "vision_attention"}),
    "vision_blocks": frozenset({"vision_mlp", "vision_attention"}),
    "action": frozenset({"action_mlp", "action_attention", "action_interfaces"}),
}

STRUCTURAL_METHODS = frozenset({"channels", "heads", "layers"})
SPARSITY_METHODS = frozenset({"magnitude", "wanda", "sparsegpt"})
QUANT_METHODS = frozenset({"bnb", "hqq", "awq_diagonal", "gptq", "attention_matching"})
EMBEDDING_METHODS = frozenset({"int8_embedding", "compact_embedding"})


def scopes_overlap(scope_a: str, scope_b: str) -> bool:
    """Return True if two linear/module scopes intersect on at least one Linear."""
    if scope_a == scope_b:
        return True
    if scope_b in _SCOPE_CHILDREN.get(scope_a, ()):
        return True
    return scope_a in _SCOPE_CHILDREN.get(scope_b, ())


def _validate_quantizer_dict(q: Any, label: str) -> dict[str, Any]:
    if not isinstance(q, dict):
        raise TypeError(f"{label}: must be an object")
    method = q.get("method")
    if method == "bnb":
        scope = q.get("scope")
        bits = q.get("bits")
        if scope not in LINEAR_SCOPES:
            raise ValueError(f"{label}: unsupported bnb scope {scope!r}")
        if not _is_int(bits) or bits not in {4, 8}:
            raise ValueError(f"{label}: bnb bits must be 4 or 8, got {bits!r}")
        return {"method": "bnb", "scope": scope, "bits": int(bits)}
    if method == "hqq":
        scope = q.get("scope")
        bits = q.get("bits")
        axis = q.get("axis", 1)
        gs = q.get("group_size", 64)
        backend = q.get("backend", "pytorch")
        if scope not in LINEAR_SCOPES:
            raise ValueError(f"{label}: unsupported hqq scope {scope!r}")
        if not _is_int(bits) or bits not in {2, 3, 4}:
            raise ValueError(f"{label}: hqq bits must be 2, 3, or 4, got {bits!r}")
        if not _is_int(axis) or axis not in {0, 1}:
            raise ValueError(f"{label}: hqq axis must be 0 or 1, got {axis!r}")
        if not _is_int(gs) or gs not in {32, 64, 128}:
            raise ValueError(f"{label}: hqq group_size must be 32, 64, or 128, got {gs!r}")
        if backend not in {"pytorch", "gemlite"}:
            raise ValueError(
                f"{label}: hqq backend must be 'pytorch' or 'gemlite', got {backend!r}"
            )
        if backend == "gemlite" and (axis != 1 or bits not in {2, 4}):
            raise ValueError(
                f"{label}: gemlite backend requires axis=1 and bits in {{2, 4}}, "
                f"got axis={axis}, bits={bits}"
            )
        return {
            "method": "hqq",
            "scope": scope,
            "bits": int(bits),
            "axis": int(axis),
            "group_size": int(gs),
            "backend": backend,
        }
    if method in {"awq_diagonal", "gptq"}:
        scope = q.get("scope")
        if scope not in LINEAR_SCOPES:
            raise ValueError(f"{label}: unsupported {method} scope {scope!r}")
        return {"method": method, "scope": scope}
    raise ValueError(f"{label}: unsupported quantizer method {method!r}")


def _validate_change_item(raw: Any, idx: int) -> dict[str, Any]:
    prefix = f"changes[{idx}]"
    if not isinstance(raw, dict):
        raise TypeError(f"{prefix}: change entry must be an object")
    method = raw.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"{prefix}: missing valid 'method' string")

    if method == "world_remove":
        return {"method": "world_remove"}
    if method in EMBEDDING_METHODS:
        return {"method": method}

    if method == "channels":
        scope = raw.get("scope")
        frac = raw.get("fraction")
        if scope not in CHANNEL_SCOPES:
            raise ValueError(f"{prefix}: channels scope {scope!r} not in {sorted(CHANNEL_SCOPES)}")
        if not _is_number(frac) or not (0.0 < float(frac) <= 0.75):
            raise ValueError(f"{prefix}: channels fraction {frac!r} must be in (0.0, 0.75]")
        return {"method": "channels", "scope": scope, "fraction": round(float(frac), 6)}

    if method == "heads":
        scope = raw.get("scope")
        frac = raw.get("fraction")
        if scope not in HEAD_SCOPES:
            raise ValueError(f"{prefix}: heads scope {scope!r} not in {sorted(HEAD_SCOPES)}")
        if not _is_number(frac) or not (0.0 < float(frac) <= 0.75):
            raise ValueError(f"{prefix}: heads fraction {frac!r} must be in (0.0, 0.75]")
        return {"method": "heads", "scope": scope, "fraction": round(float(frac), 6)}

    if method == "layers":
        scope = raw.get("scope")
        if scope not in LAYER_SCOPES:
            raise ValueError(f"{prefix}: layers scope {scope!r} not in {sorted(LAYER_SCOPES)}")
        criterion = raw.get("criterion", "block_influence")
        if criterion not in {"block_influence", "cka"}:
            raise ValueError(
                f"{prefix}: layers criterion must be 'block_influence' or 'cka', got {criterion!r}"
            )
        out: dict[str, Any] = {"method": "layers", "scope": scope, "criterion": criterion}
        if "count" in raw:
            cnt = raw["count"]
            if not _is_int(cnt) or not (1 <= cnt <= 24):
                raise ValueError(f"{prefix}: layers count {cnt!r} must be int in [1, 24]")
            out["count"] = int(cnt)
        elif "fraction" in raw:
            frac = raw["fraction"]
            if not _is_number(frac) or not (0.0 < float(frac) <= 0.75):
                raise ValueError(f"{prefix}: layers fraction {frac!r} must be in (0.0, 0.75]")
            out["fraction"] = round(float(frac), 6)
        else:
            raise ValueError(f"{prefix}: layers requires 'count' or 'fraction'")
        return out

    if method in SPARSITY_METHODS:
        scope = raw.get("scope")
        if scope not in LINEAR_SCOPES:
            raise ValueError(f"{prefix}: {method} scope {scope!r} not supported")
        pattern = raw.get("pattern")
        sparse_kernel = bool(raw.get("sparse_kernel", False))
        if pattern is not None and pattern != "2:4":
            raise ValueError(f"{prefix}: unsupported sparsity pattern {pattern!r}; expected '2:4'")
        if sparse_kernel and pattern != "2:4":
            raise ValueError(f"{prefix}: sparse_kernel=True requires pattern='2:4'")
        out = {"method": method, "scope": scope}
        if pattern is not None:
            out["pattern"] = pattern
        else:
            frac = raw.get("fraction")
            if not _is_number(frac) or not (0.0 < float(frac) <= 0.90):
                raise ValueError(f"{prefix}: {method} fraction {frac!r} must be in (0.0, 0.90]")
            out["fraction"] = round(float(frac), 6)
        if sparse_kernel:
            out["sparse_kernel"] = True
        return out

    if method in {"bnb", "hqq", "awq_diagonal", "gptq"}:
        return _validate_quantizer_dict(raw, prefix)

    if method == "attention_matching":
        control = bool(raw.get("control", False))
        out = {"method": "attention_matching"}
        if control:
            out["control"] = True
            return out
        if "quantizer" not in raw:
            raise ValueError(
                f"{prefix}: attention_matching requires 'quantizer' unless control=True"
            )
        out["quantizer"] = _validate_quantizer_dict(raw["quantizer"], f"{prefix}.quantizer")
        out["atm"] = bool(raw.get("atm", True))
        out["ohb"] = bool(raw.get("ohb", False))
        return out

    if method == "visual_tokens":
        frac = raw.get("fraction")
        criterion = raw.get("criterion", "magnitude")
        if not _is_number(frac) or not (0.0 < float(frac) <= 0.85):
            raise ValueError(f"{prefix}: visual_tokens fraction {frac!r} must be in (0.0, 0.85]")
        if criterion not in {"magnitude", "uniform"}:
            raise ValueError(
                f"{prefix}: visual_tokens criterion must be 'magnitude' or 'uniform', got {criterion!r}"
            )
        return {
            "method": "visual_tokens",
            "fraction": round(float(frac), 6),
            "criterion": criterion,
        }

    if method == "conditioning_tokens":
        scope = raw.get("scope")
        if scope == "latent":
            return {"method": "conditioning_tokens", "scope": "latent"}
        cnt = raw.get("count")
        if not _is_int(cnt) or not (0 <= cnt <= 32):
            raise ValueError(
                f"{prefix}: conditioning_tokens requires scope='latent' or int count in [0, 32]"
            )
        return {"method": "conditioning_tokens", "count": int(cnt)}

    if method == "solver_steps":
        cnt = raw.get("count")
        if not _is_int(cnt) or not (1 <= cnt <= 16):
            raise ValueError(f"{prefix}: solver_steps count {cnt!r} must be int in [1, 16]")
        return {"method": "solver_steps", "count": int(cnt)}

    if method == "action_horizon":
        cnt = raw.get("count")
        if not _is_int(cnt) or not (1 <= cnt <= 7):
            raise ValueError(f"{prefix}: action_horizon count {cnt!r} must be int in [1, 7]")
        return {"method": "action_horizon", "count": int(cnt)}

    if method == "replanning":
        cnt = raw.get("count")
        if not _is_int(cnt) or not (1 <= cnt <= 7):
            raise ValueError(f"{prefix}: replanning count {cnt!r} must be int in [1, 7]")
        return {"method": "replanning", "count": int(cnt)}

    if method == "compile_linears":
        scope = raw.get("scope")
        if scope not in LINEAR_SCOPES:
            raise ValueError(f"{prefix}: compile_linears scope {scope!r} not supported")
        out = {"method": "compile_linears", "scope": scope}
        gemm = raw.get("gemm_backend")
        if gemm is not None:
            if gemm != "triton":
                raise ValueError(f"{prefix}: compile_linears gemm_backend must be null or 'triton'")
            out["gemm_backend"] = gemm
        return out

    raise ValueError(f"{prefix}: unknown compression method {method!r}")


def _change_scope(ch: dict[str, Any]) -> str | None:
    if ch["method"] == "attention_matching":
        q = ch.get("quantizer")
        return q.get("scope") if isinstance(q, dict) else "action"
    return ch.get("scope")


def validate_change_order(
    changes: list[dict[str, Any]],
    *,
    engine: str | None = None,
) -> None:
    """Enforce physically valid stage ordering across `evaluation/compression/recipes.py` handlers."""
    seen_embed: set[str] = set()
    world_removed = False
    quantized_scopes: list[str] = []
    sparse_kernel_scopes: list[str] = []
    width_shrunk_scopes: list[str] = []
    compiled_scopes: list[str] = []
    horizon_val: int | None = None
    replan_seen = False

    for idx, ch in enumerate(changes):
        m = ch["method"]
        scope = _change_scope(ch)

        if m == "world_remove":
            if world_removed:
                raise ValueError(f"changes[{idx}]: duplicate 'world_remove'")
            world_removed = True
        elif m in EMBEDDING_METHODS:
            if seen_embed:
                raise ValueError(
                    f"changes[{idx}]: conflicting/duplicate embedding transform {m!r} "
                    f"(already have {sorted(seen_embed)})"
                )
            seen_embed.add(m)
        elif m in STRUCTURAL_METHODS:
            assert scope is not None
            if any(scopes_overlap(scope, qs) for qs in quantized_scopes):
                raise ValueError(
                    f"changes[{idx}]: structural pruning ({m} on {scope!r}) must run BEFORE "
                    f"weight quantization on overlapping scopes {quantized_scopes}"
                )
            if any(scopes_overlap(scope, cs) for cs in compiled_scopes):
                raise ValueError(
                    f"changes[{idx}]: structural pruning ({m} on {scope!r}) cannot follow compile_linears"
                )
            if m in {"channels", "heads"}:
                width_shrunk_scopes.append(scope)
        elif m in SPARSITY_METHODS:
            assert scope is not None
            if any(scopes_overlap(scope, qs) for qs in quantized_scopes):
                raise ValueError(
                    f"changes[{idx}]: sparsity ({m} on {scope!r}) must run BEFORE quantization"
                )
            if m == "wanda" and any(scopes_overlap(scope, ws) for ws in width_shrunk_scopes):
                raise ValueError(
                    f"changes[{idx}]: wanda on {scope!r} reads calibration['linear_inputs'] widths "
                    f"and cannot follow channels/heads width slicing on {width_shrunk_scopes}"
                )
            if ch.get("sparse_kernel"):
                sparse_kernel_scopes.append(scope)
        elif m in QUANT_METHODS:
            if (
                m == "awq_diagonal"
                and scope is not None
                and any(scopes_overlap(scope, ws) for ws in width_shrunk_scopes)
            ):
                raise ValueError(
                    f"changes[{idx}]: awq_diagonal on {scope!r} reads original calibration "
                    f"input widths and cannot follow channels/heads slicing on {width_shrunk_scopes}"
                )
            if scope is not None:
                if any(scopes_overlap(scope, sks) for sks in sparse_kernel_scopes):
                    raise ValueError(
                        f"changes[{idx}]: {m} on {scope!r} cannot quantize a sparse_kernel=True linear"
                    )
                if any(scopes_overlap(scope, cs) for cs in compiled_scopes):
                    raise ValueError(
                        f"changes[{idx}]: {m} on {scope!r} must precede compile_linears"
                    )
                if any(scopes_overlap(scope, qs) for qs in quantized_scopes):
                    raise ValueError(
                        f"changes[{idx}]: {m} on {scope!r} overlaps already-quantized scope {quantized_scopes}"
                    )
                quantized_scopes.append(scope)
        elif m == "visual_tokens":
            if engine == "graph":
                raise ValueError(
                    f"changes[{idx}]: visual_tokens uses decoder forward hooks bypassed by engine='graph'"
                )
        elif m == "action_horizon":
            if replan_seen:
                raise ValueError(
                    f"changes[{idx}]: action_horizon must precede replanning so horizon bounds are set"
                )
            horizon_val = int(ch["count"])
        elif m == "replanning":
            replan_seen = True
            if horizon_val is not None and int(ch["count"]) > horizon_val:
                raise ValueError(
                    f"changes[{idx}]: replanning count {ch['count']} exceeds action_horizon {horizon_val}"
                )
        elif m == "compile_linears":
            assert scope is not None
            if engine == "graph":
                raise ValueError(
                    f"changes[{idx}]: compile_linears conflicts with engine='graph' (use engine_options.use_compile)"
                )
            if any(qs == scope or scope in _SCOPE_CHILDREN.get(qs, ()) for qs in quantized_scopes):
                raise ValueError(
                    f"changes[{idx}]: compile_linears on {scope!r} has no unquantized nn.Linear "
                    f"modules left after quantization of {quantized_scopes}"
                )
            compiled_scopes.append(scope)


def recipe_vocab_type(changes: list[dict[str, Any]]) -> str:
    """Return 'restricted' if recipe uses compact_embedding, else 'full'."""
    if any(c.get("method") == "compact_embedding" for c in changes):
        return "restricted"
    return "full"


def canonical_recipe_signature(recipe: dict[str, Any]) -> tuple[Any, ...]:
    """Deterministic canonical signature preserving physical execution order & engine options."""
    engine = recipe.get("engine")
    eng_opts = recipe.get("engine_options") or {}
    opt_items = tuple(sorted((str(k), eng_opts[k]) for k in eng_opts))
    changes_sig = tuple(
        json.dumps(ch, sort_keys=True, separators=(",", ":")) for ch in recipe.get("changes", [])
    )
    return (engine, opt_items, changes_sig)


def validate_freeform_recipe(
    raw_recipe: Any,
    *,
    expected_id: str | None = None,
    seen_signatures: dict[tuple[Any, ...], str] | None = None,
    vocab_mode: str = "any",
) -> dict[str, Any]:
    """Strictly validate an unrestricted compression recipe and its application order."""
    if not isinstance(raw_recipe, dict):
        raise TypeError(f"recipe must be a JSON object, got {type(raw_recipe).__name__}")
    unknown = sorted(set(raw_recipe) - RECIPE_TOP_KEYS)
    if unknown:
        raise ValueError(f"recipe has unknown top-level keys {unknown}")
    for req in ("id", "family", "changes"):
        if req not in raw_recipe:
            raise ValueError(f"recipe missing required key {req!r}")

    cid = raw_recipe["id"]
    if not isinstance(cid, str) or not ID_RE.match(cid):
        raise ValueError(f"invalid recipe id {cid!r}; must match {ID_RE.pattern}")
    if expected_id is not None and cid != expected_id:
        raise ValueError(f"recipe id {cid!r} must match assigned node id {expected_id!r}")

    family = raw_recipe["family"]
    if not isinstance(family, str) or not family.strip():
        raise ValueError("recipe 'family' must be a non-empty string")

    engine = raw_recipe.get("engine")
    if engine is not None and engine != "graph":
        raise ValueError(f"unsupported engine {engine!r}; only None or 'graph' allowed")

    raw_opts = raw_recipe.get("engine_options")
    norm_opts: dict[str, Any] | None = None
    if raw_opts is not None:
        if engine != "graph":
            raise ValueError("engine_options is only valid when engine='graph'")
        if not isinstance(raw_opts, dict):
            raise TypeError("engine_options must be an object")
        allowed_opts = {"use_compile", "dense"}
        bad_opts = sorted(set(raw_opts) - allowed_opts)
        if bad_opts:
            raise ValueError(f"unknown engine_options keys {bad_opts}")
        norm_opts = {}
        if "use_compile" in raw_opts:
            norm_opts["use_compile"] = bool(raw_opts["use_compile"])
        if "dense" in raw_opts:
            norm_opts["dense"] = bool(raw_opts["dense"])

    raw_changes = raw_recipe["changes"]
    if not isinstance(raw_changes, list) or not raw_changes:
        raise ValueError("recipe 'changes' must be a non-empty list")

    norm_changes = [_validate_change_item(ch, idx) for idx, ch in enumerate(raw_changes)]
    validate_change_order(norm_changes, engine=engine)

    vocab = recipe_vocab_type(norm_changes)
    if vocab_mode == "full" and vocab != "full":
        raise ValueError("vocab_mode='full' forbids 'compact_embedding'; use 'int8_embedding'")
    if vocab_mode == "restricted" and vocab != "restricted":
        raise ValueError("vocab_mode='restricted' requires 'compact_embedding'")

    normalized: dict[str, Any] = {
        "id": cid,
        "family": family.strip(),
        "changes": norm_changes,
    }
    if engine is not None:
        normalized["engine"] = engine
        if norm_opts is not None:
            normalized["engine_options"] = norm_opts

    if seen_signatures is not None:
        sig = canonical_recipe_signature(normalized)
        if sig in seen_signatures:
            raise ValueError(
                f"recipe configuration duplicates already-proposed/probed cell '{seen_signatures[sig]}'"
            )
    return normalized


def freeform_action_space_doc() -> str:
    """Concise, accurate specification of compression handlers, scopes, and ordering rules."""
    spec = {
        "recipe_schema": {
            "id": "<assigned cell_id string>",
            "family": "<short mechanism family slug>",
            "engine": "optional 'graph' (CUDA-graph execution; set engine_options.dense=false to keep packed NF4 small, dense=true trades ~3 GB extra VRAM for speed)",
            "engine_options": {
                "use_compile": "bool (default true)",
                "dense": "bool (default true)",
            },
            "changes": "<ordered list of handler dicts below>",
        },
        "proper_order": [
            "1. world_remove + embedding (int8_embedding for full-vocab, compact_embedding for LIBERO-Spatial restricted-vocab; never both).",
            "2. Structural shape/layer surgery FIRST: channels, heads, layers (before quantization or wanda/awq_diagonal on overlapping scopes).",
            "3. Sparsity BEFORE weight quantization: magnitude, wanda, sparsegpt (2:4 or unstructured; sparse_kernel=true requires pattern='2:4' and cannot be followed by quantization).",
            "4. Weight quantization: bnb (4|8), hqq (bits 2|3|4, axis 0|1, group_size 32|64|128, backend pytorch|gemlite [gemlite needs axis=1, bits 2|4]), awq_diagonal, gptq, attention_matching.",
            "5. Optional execution/compile: visual_tokens (incompatible with engine='graph'), conditioning_tokens, solver_steps, action_horizon (must precede replanning), replanning, compile_linears (only on unquantized nn.Linear scopes; incompatible with engine='graph').",
        ],
        "scopes": {
            "channels": sorted(CHANNEL_SCOPES),
            "heads": sorted(HEAD_SCOPES),
            "layers": sorted(LAYER_SCOPES),
            "linear_scopes (bnb/hqq/awq/gptq/wanda/magnitude/sparsegpt/compile_linears)": sorted(
                LINEAR_SCOPES
            ),
        },
    }
    return json.dumps(spec, indent=2)
