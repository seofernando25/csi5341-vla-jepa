"""Scoped Linear compilation controls for TorchAO's intended deployment path.

Compile the functional operation, preserving module paths, hooks and tensor
payloads. This is a Linear-only backend comparison, not whole-policy fusion.
"""

from functools import partial
from importlib.metadata import PackageNotFoundError, version
from types import MethodType

import torch
import torch._inductor.config as inductor_config
from torch.nn import functional as F

from evaluation.compression.transforms import selected_linears


def linear_operation(inputs, weight, bias):
    return F.linear(inputs, weight, bias)


def compiled_forward(operation, module, input):
    return operation(input, module.weight, module.bias)


def compile_linears(policy, change):
    gemm_backend = change.get("gemm_backend")
    if gemm_backend not in {None, "triton"}:
        raise ValueError("Unknown compiled GEMM backend")
    entries = selected_linears(policy, change["scope"])
    if not entries:
        raise ValueError("No Linear modules found for compilation")
    if any(hasattr(linear, "_compression_compile_options") for _, _, linear in entries.values()):
        raise ValueError("Linear compilation already applied in this scope")
    # One shared functional graph handles multiple projection widths and input
    # layouts. Exceeding this recorded limit fails instead of using eager code.
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.suppress_errors = False
    # Match the pinned TorchAO 0.18 recommended_inductor_config_setter for
    # both quantized deployments and their floating-point controls.
    settings = {
        "coordinate_descent_tuning": True,
        "coordinate_descent_check_all_directions": True,
        "force_fuse_int_mm_with_mul": True,
        "fx_graph_cache": True,
        "triton.unique_kernel_names": True,
    }
    if gemm_backend == "triton":
        # ATen's cuBLAS integer path rejects small action rows/widths. Keep
        # the original tensors and select Inductor's masked Triton templates.
        settings["max_autotune_gemm_backends"] = "TRITON"
    for name, value in settings.items():
        prefix, separator, leaf = name.partition(".")
        setattr(
            getattr(inductor_config, prefix) if separator else inductor_config,
            leaf if separator else prefix,
            value,
        )
    options = {
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs" if gemm_backend else "default",
        "fullgraph": True,
        "dynamic": True,
    }
    operation = torch.compile(linear_operation, **options)
    for _, _, linear in entries.values():
        linear.forward = MethodType(partial(compiled_forward, operation), linear)
        linear._compression_compile_options = options.copy()
    try:
        triton_version = version("triton")
    except PackageNotFoundError:
        triton_version = None
    return {
        "method": "compile_linears",
        "scope": change["scope"],
        "options": options,
        "recompile_limit": 128,
        "inductor_settings": settings,
        "compiler_versions": {"torch": torch.__version__, "triton": triton_version},
        "silent_eager_fallback": False,
        "compiled_modules": list(entries),
        **({"gemm_backend": gemm_backend} if gemm_backend else {}),
        "interpretation": "Scoped Linear compilation; remaining policy operations are eager",
    }
