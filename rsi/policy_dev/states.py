"""Status machine of one policy-development step, as `verdict.json` and `selection.json` record it.

Step 0 is the `incumbent`. Every other step starts `rejected` (nothing usable yet) and gets at
most two agent attempts:

    rejected   -> rejected | not_better | valid | timeout | timeout_salvaged   (attempt 1 or 2)
    not_better -> rejected | valid | timeout | timeout_salvaged                (attempt 2)
    valid      -> accepted | not_better                                        (selection)

`valid` is internal: selection resolves it before anything is written. `accepted`, `timeout`,
`timeout_salvaged` and `incumbent` are final; a step that ends `rejected` or `not_better`
keeps that status.
"""

from __future__ import annotations

from enum import StrEnum

from evaluation.fsm import StateMachine


class StepStatus(StrEnum):
    """`status` of one policy-development step."""

    INCUMBENT = "incumbent"
    REJECTED = "rejected"
    NOT_BETTER = "not_better"
    VALID = "valid"
    ACCEPTED = "accepted"
    TIMEOUT = "timeout"
    TIMEOUT_SALVAGED = "timeout_salvaged"


_P = StepStatus
STEP = StateMachine(
    "policy step",
    StepStatus,
    {
        _P.REJECTED: {_P.REJECTED, _P.NOT_BETTER, _P.VALID, _P.TIMEOUT, _P.TIMEOUT_SALVAGED},
        _P.NOT_BETTER: {_P.REJECTED, _P.VALID, _P.TIMEOUT, _P.TIMEOUT_SALVAGED},
        _P.VALID: {_P.ACCEPTED, _P.NOT_BETTER},
    },
    initial={_P.INCUMBENT, _P.REJECTED},
)
# Steps whose candidate replayed and may be selected.
SCORED = frozenset({_P.INCUMBENT, _P.ACCEPTED, _P.NOT_BETTER, _P.TIMEOUT_SALVAGED})
# Statuses a written verdict can hold (`valid` never reaches disk).
RECORDED = frozenset(StepStatus) - {_P.VALID}
