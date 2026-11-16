"""Resident bytes of a policy: a storage walk and the live CUDA figure.

`ResidentWalk` is the measurement the fixed evaluator charges a solution with, and the same
walk the workspace's own check tool uses. It follows tensors, arrays, buffers, containers,
instance dicts, slots, classes, closures and bound methods from a set of roots, counting each
storage once and recording off-device tensors, memory maps and walk-cap overruns as
violations. Nothing here imports the compression methods, so the walk can be read and copied
into a workspace without exposing them.
"""

from __future__ import annotations

import functools
import gc
import mmap
import types
import warnings

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor

RESIDENT_MAX_NODES = 2_000_000
RESIDENT_MAX_DEPTH = 256
_ATOMS = (str, int, float, complex, bool, type(None))


class ResidentWalk:
    """Storage bytes reachable from Python roots, each storage counted once.

    Follows tensors with their Python attributes and subclass fields, numpy arrays, buffer
    protocol objects, containers, instance dicts and slots, the class of every object, class
    dicts, closures, defaults and bound receivers. Module objects are not followed; callers
    pass module dicts as roots. Off-device and meta tensors, memory maps, and exceeding the
    node or depth cap are recorded as violations.
    """

    def __init__(self, required_device: str = "cuda") -> None:
        self.required_device = required_device
        self.storages: dict[tuple, int] = {}
        self.violations: list[str] = []
        self._seen: set[int] = set()
        self._keep: list[object] = []

    @property
    def nbytes(self) -> int:
        return sum(self.storages.values())

    def add(self, *roots: object) -> None:
        """Walk depth-first from each root, visiting every object once."""
        stack = [(root, 0) for root in roots]
        while stack:
            value, depth = stack.pop()
            if isinstance(value, _ATOMS) or id(value) in self._seen:
                continue
            if depth > RESIDENT_MAX_DEPTH or len(self._seen) >= RESIDENT_MAX_NODES:
                self.violations.append("walk cap exceeded; resident size cannot be certified")
                return
            self._seen.add(id(value))
            self._keep.append(value)
            stack.extend((child, depth + 1) for child in self._children(value))

    def _children(self, value: object) -> list[object]:
        if torch.is_tensor(value):
            return self._tensor(value)
        if isinstance(value, torch.UntypedStorage):
            self._storage(value.device, value.data_ptr(), value.nbytes())
            return []
        if isinstance(value, np.ndarray):
            return self._array(value)
        if isinstance(value, mmap.mmap):
            self.violations.append("memory-mapped buffer")
            return []
        if isinstance(value, dict):
            return [*value.keys(), *value.values()]
        if isinstance(value, (list, tuple, set, frozenset)):
            return list(value)
        if isinstance(value, types.ModuleType):
            return []
        if isinstance(value, type):
            return list(vars(value).values())
        if isinstance(value, functools.partial):
            return [value.func, *value.args, *value.keywords.values()]
        if isinstance(value, types.MethodType):
            return [value.__self__, value.__func__]
        if isinstance(value, types.FunctionType):
            defaults = (value.__defaults__ or ()) + tuple((value.__kwdefaults__ or {}).values())
            return [*_cell_contents(value), *defaults]
        self._buffer(value)
        children = [type(value), *getattr(value, "__dict__", {}).values()]
        for owner in type(value).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            for name in (slots,) if isinstance(slots, str) else slots:
                if hasattr(value, name):
                    children.append(getattr(value, name))
        return children

    def _tensor(self, tensor: torch.Tensor) -> list[object]:
        if tensor.device.type == "meta":
            self.violations.append("meta tensor")
            return []
        if tensor.device.type != self.required_device:
            self.violations.append(f"{tensor.device.type} tensor {tuple(tensor.shape)}")
        children = list(getattr(tensor, "__dict__", {}).values())
        if hasattr(tensor, "__tensor_flatten__"):
            names, _ = tensor.__tensor_flatten__()
            return children + [getattr(tensor, name) for name in names]
        if tensor.layout == torch.sparse_coo:
            return children + [tensor._values(), tensor._indices()]
        if tensor.layout != torch.strided:
            self.violations.append(f"unsupported layout {tensor.layout}")
            return children
        storage = tensor.untyped_storage()
        self._storage(tensor.device, storage.data_ptr(), storage.nbytes())
        return children

    def _array(self, array: np.ndarray) -> list[object]:
        if isinstance(array, np.memmap):
            self.violations.append("numpy memmap")
        owner = array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        self.storages[("ptr", owner.__array_interface__["data"][0])] = owner.nbytes
        if array.dtype == object:
            return list(array.ravel())
        return [] if owner.base is None else [owner.base]

    def _buffer(self, value: object) -> None:
        try:
            view = memoryview(value)
        except TypeError:
            return
        with view:
            self.storages[("obj", id(value))] = view.nbytes

    def _storage(self, device: torch.device, pointer: int, nbytes: int) -> None:
        self.storages[("device", device.type, device.index, pointer)] = nbytes


def _cell_contents(function: types.FunctionType) -> list[object]:
    values = []
    for cell in function.__closure__ or ():
        try:
            values.append(cell.cell_contents)
        except ValueError:
            continue
    return values


VIRTUAL_TENSORS = (FakeTensor, FunctionalTensor)


def live_cuda_bytes() -> int:
    """Bytes of every live CUDA strided tensor storage in the process, after a full collection."""
    gc.collect()
    storages: dict[tuple, int] = {}
    with warnings.catch_warnings():
        # Probing every object touches deprecated aliases that warn on `isinstance`.
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            if not (torch.is_tensor(obj) and obj.is_cuda and obj.layout == torch.strided):
                continue
            if isinstance(obj, VIRTUAL_TENSORS):
                # FakeTensor and FunctionalTensor claim a device but own no memory.
                continue
            try:
                storage = obj.untyped_storage()
            except (RuntimeError, NotImplementedError):
                continue
            storages[(obj.device.index, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())
