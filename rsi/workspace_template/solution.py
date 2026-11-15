"""Compression entry point for the VLA-JEPA B16 policy.

compress(policy, calibration) receives the loaded policy (CUDA, bfloat16, eval mode, no
gradients) and the frozen calibration dict. It must return an object that still exposes
select_action and predict_action_chunk. Memory is measured on the returned object and on
every tensor reachable from this module and from modules imported out of the workspace.
"""

from __future__ import annotations


def compress(policy, calibration):
    """Return a policy with less resident memory; the identity keeps the baseline."""
    return policy
