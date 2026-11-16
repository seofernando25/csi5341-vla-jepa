"""The generic state machine: parsing, legal and illegal moves, and JSON identity."""

from __future__ import annotations

import json
from enum import StrEnum

import pytest

from evaluation.fsm import IllegalTransition, StateMachine, UnknownState


class Light(StrEnum):
    OFF = "off"
    ON = "on"
    BROKEN = "broken"


LIGHT = StateMachine(
    "light",
    Light,
    {Light.OFF: {Light.ON}, Light.ON: {Light.OFF, Light.BROKEN}},
    initial={Light.OFF},
)


def test_advance_walks_a_legal_path_and_stores_plain_strings() -> None:
    record: dict = {}
    for state in ("off", "on", "off", "on", "broken"):
        LIGHT.advance(record, state)
        assert record["status"] == state and type(record["status"]) is str
    assert LIGHT.terminal() == {Light.BROKEN}


def test_errors_name_the_machine_the_move_and_the_options() -> None:
    with pytest.raises(UnknownState, match=r"light: unknown state 'dim'; expected one of off, on"):
        LIGHT.parse("dim")
    with pytest.raises(
        IllegalTransition, match=r"light: off -> broken is not allowed; from off: on"
    ):
        LIGHT.check("off", "broken")
    with pytest.raises(IllegalTransition, match=r"start -> on .* from start: off"):
        LIGHT.check(None, "on")
    with pytest.raises(IllegalTransition, match=r"from broken: none \(terminal\)"):
        LIGHT.check("broken", "off")


def test_members_serialize_exactly_like_the_old_strings() -> None:
    for member in Light:
        assert json.dumps({"status": member}) == json.dumps({"status": member.value})
        assert f"{member}" == member.value and member == member.value


def test_a_table_naming_an_unknown_state_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown states"):
        StateMachine("bad", Light, {"dim": {Light.ON}}, initial={})
