"""Headless discovery and policy-development agents: backends, sandboxed runs and run records.

`spec` names the agent, `pi` and `claude` build each backend's argv, `command` dispatches,
`run` executes one agent in the `bwrap` sandbox, `output` parses its transcript and `result`
keeps the path-free record the budget guard and manifests read.
"""

from rsi.agents.command import agent_cli_version, agent_command
from rsi.agents.output import clean_text, parse_agent_output, pi_cost, pi_text
from rsi.agents.result import (
    AgentRunResult,
    agent_record,
    agent_result_payload,
    read_agent_result,
    timed_out_result,
    write_agent_result,
)
from rsi.agents.run import run_sandboxed_agent
from rsi.agents.spec import BACKENDS, TASK_PROMPT, AgentSpec, load_agent_spec

__all__ = [
    "BACKENDS",
    "TASK_PROMPT",
    "AgentRunResult",
    "AgentSpec",
    "agent_cli_version",
    "agent_command",
    "agent_record",
    "agent_result_payload",
    "clean_text",
    "load_agent_spec",
    "parse_agent_output",
    "pi_cost",
    "pi_text",
    "read_agent_result",
    "run_sandboxed_agent",
    "timed_out_result",
    "write_agent_result",
]
