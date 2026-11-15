"""Tests for Dream-RSI action space, validator, proposal runner, and journal conversion."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from rsi.action_space import (
    ACTION_SPACE,
    build_recipe_changes,
    canonical_signature_from_candidate,
    validate_proposal,
)
from rsi.live import (
    build_round1_candidates,
    build_round2_candidates,
    build_round3_candidates,
    build_round4_candidates,
    build_round5_candidates,
    build_round6_candidates,
)


def _valid_c1(cid: str = "r7-c1-test") -> dict:
    return {
        "id": cid,
        "base": "C1",
        "knobs": {
            "text_mlp_prune": 0.10,
            "vision_mlp_prune": 0.30,
            "group_size": 32,
            "vision_bits": 2,
            "action_precision": "bf16_attn_hqq2_mlp",
            "solver_steps": 1,
        },
        "hypothesis": "Tests BF16 action attention with 1-step solver on C1.",
        "expected_resident_mb": 1220.0,
    }


def _valid_t2(cid: str = "r7-t2-test") -> dict:
    return {
        "id": cid,
        "base": "T2",
        "knobs": {
            "text_mlp_prune": 0.15,
            "vision_mlp_prune": 0.35,
            "group_size": 32,
            "vision_bits": 2,
            "action_precision": "bf16_full",
            "solver_steps": 4,
        },
        "hypothesis": "Tests full BF16 action head on T2 with moderate MLP cuts.",
        "expected_resident_mb": 940.0,
    }


def test_r1_to_r6_candidates_unchanged_after_refactor() -> None:
    expected = {
        "r1": (9, "73b7352614aaf2c5bf17e1b11c3034b4c13840501acef559c837766c91bcdef3"),
        "r2": (6, "6ccb142471e2564384f96316b4be2ff1a60d0042ada53433ccbfe4abab7bd677"),
        "r3": (8, "3885d6dba3848ce9159aa055a7c3b9ac8805576d67bdaf23d28e1e1258eceaae"),
        "r4": (6, "83454a4efab711a16614c47e706d3becd5ce7bcc57dba278404c39ec9d6c8ac1"),
        "r5": (6, "25f28143da22f9cb703d40aaeb799dc77df9f8bc658e3c6b37a0dcceadbc8ab6"),
        "r6": (6, "b1ee6ba76f47b61e2d216cc0fc9593c946ef806da0377debb16b5f987bd6be11"),
    }
    builders = {
        "r1": build_round1_candidates,
        "r2": build_round2_candidates,
        "r3": build_round3_candidates,
        "r4": build_round4_candidates,
        "r5": build_round5_candidates,
        "r6": build_round6_candidates,
    }
    for rnd, fn in builders.items():
        cands = fn()
        count, digest = expected[rnd]
        assert len(cands) == count
        actual = hashlib.sha256(
            json.dumps([dataclasses.asdict(c) for c in cands], sort_keys=True).encode()
        ).hexdigest()
        assert actual == digest


def test_validator_accepts_valid_proposal_and_rejects_violations() -> None:
    c1 = _valid_c1()
    t2 = _valid_t2()
    c2 = {
        "id": "r7-c2-test",
        "base": "C2",
        "knobs": {
            "text_mlp_prune": 0.10,
            "vision_mlp_prune": 0.25,
            "vision_bits": 4,
            "action_precision": "nf4",
            "solver_steps": 1,
        },
        "hypothesis": "Tests mild MLP cuts on C2 one-step NF4.",
        "expected_resident_mb": 1240.0,
    }
    out = validate_proposal([c1, t2, c2], max_n=3)
    assert len(out) == 3
    assert out[2]["knobs"]["group_size"] is None
    assert "bases" in ACTION_SPACE and set(ACTION_SPACE["bases"]) == {"T2", "C1", "C2"}

    # Unknown candidate key
    bad = dict(c1, extra_key=1)
    with pytest.raises(ValueError, match="unknown keys"):
        validate_proposal([bad], max_n=2)

    # Unknown knob key
    bad_knob = _valid_c1()
    bad_knob["knobs"]["bogus"] = 1
    with pytest.raises(ValueError, match="unknown knob keys"):
        validate_proposal([bad_knob], max_n=2)

    # Out-of-range prune fraction
    oor = _valid_c1()
    oor["knobs"]["text_mlp_prune"] = 0.55
    with pytest.raises(ValueError, match="out of range"):
        validate_proposal([oor], max_n=2)

    # Disallowed base/knob combo (C2 with group_size=32)
    bad_c2 = dict(c2, knobs={**c2["knobs"], "group_size": 32})
    with pytest.raises(ValueError, match="group_size"):
        validate_proposal([bad_c2], max_n=2)

    # More than requested count
    with pytest.raises(ValueError, match="exceeding requested count"):
        validate_proposal([c1, t2], max_n=1)

    # Duplicate of already-probed id
    with pytest.raises(ValueError, match="duplicates already-probed cell"):
        validate_proposal([c1], max_n=2, probed_ids={c1["id"]})

    # Duplicate of already-probed config under a new id
    sig = canonical_signature_from_candidate(c1["base"], c1["knobs"])
    with pytest.raises(ValueError, match="duplicates already-probed cell 'r4-existing'"):
        validate_proposal([c1], max_n=2, probed_signatures={sig: "r4-existing"})

    # Within-proposal duplicate configuration rejected
    with pytest.raises(ValueError, match="duplicates earlier proposal candidate"):
        validate_proposal([_valid_t2("t2-a"), _valid_t2("t2-b")], max_n=2)

    # Less than half full-vocab rejected
    t2_b = _valid_t2("t2-b")
    t2_b["knobs"]["text_mlp_prune"] = 0.20
    with pytest.raises(ValueError, match="at least half"):
        validate_proposal([_valid_t2("t2-a"), t2_b], max_n=2)

    # Builder sanity on C2
    changes = build_recipe_changes(
        base="C2",
        text_mlp_prune=0.10,
        vision_mlp_prune=0.25,
        group_size=None,
        vision_bits=4,
        action_precision="nf4",
        solver_steps=1,
    )
    assert any(ch["method"] == "solver_steps" and ch["count"] == 1 for ch in changes)
