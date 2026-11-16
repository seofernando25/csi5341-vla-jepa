"""One Dream-RSI node expansion and its verdict.

`node` expands and promotes nodes, `context` holds their inputs and the evaluator, `verdict`
maps an evaluator result to a fail class and score, and `outputs` checks what the agent
returned and strips host paths from every record.
"""

from rsi.expansion.context import (
    DigestRegistry,
    Evaluator,
    NodeContext,
    NodeJob,
    evaluate_snapshot,
    sandboxed_evaluator,
)
from rsi.expansion.node import expand_node, promote_node
from rsi.expansion.outputs import (
    HOST_PATH_MARKERS,
    AgentOutputInvalid,
    placeholder_output_reason,
    redact_host_paths,
)
from rsi.expansion.verdict import Verdict, classify

__all__ = [
    "HOST_PATH_MARKERS",
    "AgentOutputInvalid",
    "DigestRegistry",
    "Evaluator",
    "NodeContext",
    "NodeJob",
    "Verdict",
    "classify",
    "evaluate_snapshot",
    "expand_node",
    "placeholder_output_reason",
    "promote_node",
    "redact_host_paths",
    "sandboxed_evaluator",
]
