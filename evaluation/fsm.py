"""Explicit state machines for statuses that are persisted as JSON strings.

Each machine is a `StrEnum` of states plus one transition table. States are `StrEnum` members,
so a member compares equal to its string and serializes to the same JSON bytes: records
written before a machine existed load unchanged. Code that changes a status asks the machine
first, and an illegal change raises `IllegalTransition` naming the machine, both states and
what was allowed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum


class UnknownState(ValueError):
    """A recorded value that is not a state of the machine."""


class IllegalTransition(ValueError):
    """A status change the machine's transition table does not allow."""


class StateMachine[S: StrEnum]:
    """States of one `StrEnum`, the legal moves between them and the legal first states."""

    def __init__(
        self,
        name: str,
        states: type[S],
        transitions: Mapping[S, Iterable[S]],
        *,
        initial: Iterable[S],
    ) -> None:
        self.name = name
        self.states = states
        self.transitions = {state: frozenset(transitions.get(state, ())) for state in states}
        self.initial = frozenset(initial)
        unknown = set(transitions) - set(states)
        if unknown:
            raise ValueError(f"{name}: transition table names unknown states {sorted(unknown)}")

    def parse(self, value: object) -> S:
        """The state a recorded value names; raise `UnknownState` for anything else."""
        try:
            return self.states(value)
        except ValueError:
            expected = ", ".join(state.value for state in self.states)
            raise UnknownState(
                f"{self.name}: unknown state {value!r}; expected one of {expected}"
            ) from None

    def allowed(self, src: object | None) -> frozenset[S]:
        """States reachable in one move from `src` (from nothing: the initial states)."""
        return self.initial if src is None else self.transitions[self.parse(src)]

    def terminal(self) -> frozenset[S]:
        """States with no way out."""
        return frozenset(state for state, nxt in self.transitions.items() if not nxt)

    def check(self, src: object | None, dst: object) -> S:
        """Return `dst` as a state when `src -> dst` is legal; raise `IllegalTransition` if not."""
        target = self.parse(dst)
        allowed = self.allowed(src)
        if target not in allowed:
            origin = "start" if src is None else self.parse(src).value
            options = ", ".join(sorted(state.value for state in allowed)) or "none (terminal)"
            raise IllegalTransition(
                f"{self.name}: {origin} -> {target.value} is not allowed; from {origin}: {options}"
            )
        return target

    def advance(self, record: dict, dst: object, key: str = "status") -> S:
        """Move `record[key]` to `dst` after checking the move; the stored value is a plain str."""
        target = self.check(record.get(key), dst)
        record[key] = target.value
        return target
