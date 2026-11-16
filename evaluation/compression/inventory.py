"""Parameter counts, resident storage and packed archives of a (compressed) policy."""

import functools
import gc
import mmap
import types
import warnings

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor

from evaluation.common import ROOT, write_json
from evaluation.compression.quantization import Int8Embedding
from evaluation.compression.topology import roots


def logical_parameter_count(module):
    """Count represented learned weights independently of their packed dtype.

    Quantization changes bytes, not model width. Packed INT4 modules/INT8
    embeddings store weights as buffers; sparse tensor numel is already logical.
    """
    count = sum(parameter.numel() for parameter in module.parameters())
    seen = set()
    covered = set()
    for child in module.modules():
        if id(child) in covered:
            continue
        if hasattr(child, "_compression_logical_parameter_elements"):
            count += child._compression_logical_parameter_elements - sum(
                p.numel() for p in child.parameters()
            )
            covered.update(id(nested) for nested in child.modules())
            continue
        if isinstance(child, nn.Linear):
            if id(child.weight) not in seen:
                count += child.in_features * child.out_features - child.weight.numel()
                seen.add(id(child.weight))
        elif hasattr(child, "packed_weight") and hasattr(child, "in_features"):
            count += child.in_features * child.out_features
        elif isinstance(child, Int8Embedding):
            count += child.num_embeddings * child.embedding_dim
    return count


def storage_inventory(policy):
    """Deduplicate actual packed tensors, tied weights and quantizer metadata."""
    result = _resident(policy)
    if hasattr(policy, "model") and hasattr(policy.model, "qwen"):
        result["inference_core_logical_parameter_elements"] = sum(
            logical_parameter_count(root) for root in roots(policy).values()
        )
        result["modules"] = {
            name: {
                "resident_tensor_storage_bytes": _resident(root)["resident_tensor_storage_bytes"],
                "logical_parameter_elements": logical_parameter_count(root),
            }
            for name, root in roots(policy).items()
        }
    return result


def _resident(policy):
    """Resident tensor storage and parameter counts of one module tree."""
    seen = set()
    total = 0

    def add(value):
        nonlocal total
        if not torch.is_tensor(value):
            return
        if hasattr(value, "__tensor_flatten__"):
            names, _ = value.__tensor_flatten__()
            for name in names:
                add(getattr(value, name))
            return
        try:
            storage = value.untyped_storage()
        except RuntimeError:
            add(value.values())
            add(value.indices())
            return
        key = (value.device.type, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
        state = getattr(value, "quant_state", None)
        while state is not None:
            for field in ["absmax", "code", "offset"]:
                add(getattr(state, field, None))
            state = getattr(state, "state2", None)
        for field in ["CB", "SCB"]:
            add(getattr(value, field, None))

    parameters = list(policy.parameters())
    for value in parameters + list(policy.buffers()):
        add(value)
    for module in policy.modules():
        state = getattr(module, "state", None)
        if state is not None:
            for field in ["CB", "SCB"]:
                add(getattr(state, field, None))
    result = {
        "registered_parameter_elements": sum(v.numel() for v in parameters),
        "represented_logical_parameter_elements": logical_parameter_count(policy),
        "resident_tensor_storage_bytes": total,
    }
    return result


def packed_state(policy):
    tensors, layouts, aliases, seen = {}, {}, {}, {}

    def visit(name, value):
        if hasattr(value, "__tensor_flatten__"):
            names, context = value.__tensor_flatten__()
            layouts[name] = {
                "class": f"{type(value).__module__}.{type(value).__name__}",
                "context": repr(context),
                "fields": names,
            }
            for field in names:
                visit(f"{name}.{field}", getattr(value, field))
        elif torch.is_tensor(value):
            if value.numel() == 0:
                return
            key = (
                str(value.device),
                value.untyped_storage().data_ptr(),
                value.storage_offset(),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
            )
            if key in seen:
                aliases[name] = seen[key]
                return
            seen[key] = name
            tensors[name] = value.detach().cpu().contiguous()
        elif value is not None:
            layouts[name] = {"value": repr(value)}

    for name, value in policy.state_dict().items():
        visit(name, value)
    # The upstream QuantVLA processor stores its calibrated output multiplier
    # as a Python scalar rather than a buffer. It affects inference and must
    # travel with the tensor payload.
    scalars = {
        f"{name}._ohb_beta_scalar": float(module._ohb_beta_scalar)
        for name, module in policy.named_modules()
        if hasattr(module, "_ohb_beta_scalar")
    }
    return tensors, {
        "tensor_subclass_layouts": layouts,
        "aliases": aliases,
        "inference_scalar_attributes": scalars,
    }


def serialize(policy, candidate, result_dir):
    folder = ROOT / "outputs/compression/serialized" / candidate / result_dir.name
    folder.mkdir(parents=True, exist_ok=False)
    tensors, layout = packed_state(policy)
    path = folder / "packed.safetensors"
    save_file(tensors, str(path), metadata={"compression_archive": "1"})
    write_json(result_dir / "tensor_layout.json", layout)
    summary = {
        "bytes": path.stat().st_size,
        "format": "safetensors packed tensor archive; subclass layouts and aliases recorded",
        "artifact": str(path.relative_to(ROOT)),
        "direct_HF_checkpoint": False,
    }
    # Compact evidence is saved before the large archive is removed. Candidates
    # remain reconstructible from the pinned original checkpoint and recipe.
    write_json(result_dir / "serialized.json", summary)
    del tensors
    path.unlink()
    summary["runtime_archive_removed_after_evidence_export"] = True
    return summary


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
