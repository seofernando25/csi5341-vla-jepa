"""Payloads a solution keeps alive outside what it returns, for the fixed evaluator.

`ResidentWalk` follows what `compress` returns and what the workspace's modules hold. A
solution can park weights where that walk never looks: another module's namespace
(`torch._stash = [...]`), a class attribute, a closure. This module finds those payloads and
keys them exactly as `ResidentWalk` does, so the evaluator can merge both charges without
counting one storage twice.

Two sweeps, because neither alone is complete:

- `_module_payloads` visits every module in `sys.modules` and descends through containers
  only, so a stash attached to any imported module is found even when it is `bytes`, which
  the garbage collector does not track;
- `_collector_payloads` reads every live tensor storage of any device and every live buffer
  object from the garbage collector, which sees tensors wherever they are kept.

`skip` is the key set recorded before `compress` runs. Only payloads that appeared after it
are charged, so state that existed beforehand (the calibration bank, the loaded policy,
library constants) is never attributed to the solution.
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import torch

Key = tuple[object, ...]
BUFFERS = (bytes, bytearray, memoryview)
CONTAINERS = (dict, list, tuple, set, frozenset)
ATOMS = (str, int, float, complex, bool, type(None))
SWEEP_DEPTH = 8
SWEEP_NODES = 500_000


def _tensor(value: torch.Tensor) -> tuple[Key, int] | None:
    if value.device.type == "meta" or value.layout != torch.strided:
        return None
    try:
        storage = value.untyped_storage()
        return (
            "device",
            value.device.type,
            value.device.index,
            storage.data_ptr(),
        ), storage.nbytes()
    except RuntimeError:
        # FakeTensor, FunctionalTensor and friends have no storage to charge.
        return None


def _payload(value: object, kind: type) -> tuple[Key, int] | None:
    """Storage key and bytes of one payload object; None for anything else."""
    if issubclass(kind, torch.Tensor):
        return _tensor(value)
    if issubclass(kind, np.ndarray):
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        return ("ptr", owner.__array_interface__["data"][0]), owner.nbytes
    if issubclass(kind, BUFFERS):
        return ("obj", id(value)), memoryview(value).nbytes
    return None


def _children(value: object, kind: type) -> list[object]:
    """Container contents through the base class, so lazy mapping subclasses are not loaded."""
    try:
        if issubclass(kind, dict):
            return [*dict.keys(value), *dict.values(value)]
        return list(value)
    except Exception:  # noqa: BLE001 - a container that refuses iteration holds nothing we can size
        return []


def _module_payloads() -> dict[Key, int]:
    """Payloads held in the namespace of any imported module, containers only."""
    found: dict[Key, int] = {}
    seen: set[int] = set()
    nodes = 0
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if type(namespace) is not dict:
            continue
        stack: list[tuple[object, int]] = [(namespace, 0)]
        while stack:
            value, depth = stack.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            kind = type(value)
            if kind in ATOMS:
                continue
            nodes += 1
            if depth > SWEEP_DEPTH or nodes > SWEEP_NODES:
                continue
            payload = _payload(value, kind)
            if payload is not None:
                key, nbytes = payload
                found[key] = max(found.get(key, 0), nbytes)
            elif issubclass(kind, CONTAINERS):
                stack.extend((child, depth + 1) for child in _children(value, kind))
    return found


def _collector_payloads() -> dict[Key, int]:
    """Every live tensor storage and buffer object the garbage collector can see."""
    found: dict[Key, int] = {}
    for value in gc.get_objects():
        kind = type(value)
        if kind in ATOMS:
            continue
        payload = _payload(value, kind)
        if payload is not None:
            key, nbytes = payload
            found[key] = max(found.get(key, 0), nbytes)
    return found


def live_payloads() -> dict[Key, int]:
    """Every live payload in the process, keyed as `ResidentWalk` keys."""
    found = _collector_payloads()
    for key, nbytes in _module_payloads().items():
        found[key] = max(found.get(key, 0), nbytes)
    return found


def new_payloads(skip: frozenset[Key]) -> dict[Key, int]:
    """Live payloads whose key is not in `skip`, i.e. created after the snapshot."""
    return {key: nbytes for key, nbytes in live_payloads().items() if key not in skip}
