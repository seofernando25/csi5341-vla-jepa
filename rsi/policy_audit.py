"""Static audit and loading of generated `OptimalPolicy` source (prefix-only contract)."""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from rsi.api import LLMDesignedMethod

if TYPE_CHECKING:
    from rsi.tree import DiscoveryTree

ALLOWED_MODULES = frozenset({"math"})
ALLOWED_FROM_MODULES = frozenset({"__future__", "rsi.api", "dataclasses", "typing"})
# Stdlib modules expose other modules as attributes (typing.sys), so only these names may be
# pulled in by from-import; rsi.api is ours and holds no module objects.
ALLOWED_FROM_NAMES = {
    "__future__": frozenset({"annotations"}),
    "dataclasses": frozenset({"dataclass", "field"}),
    "typing": frozenset(
        {"Any", "Callable", "ClassVar", "Iterable", "Literal", "Mapping", "Optional", "Sequence"}
    ),
}
FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "format",
        "getattr",
        "globals",
        "hasattr",
        "id",
        "input",
        "locals",
        "memoryview",
        "object",
        "open",
        "setattr",
        "type",
        "vars",
    }
)
FORBIDDEN_QUESTION_ATTRS = frozenset(
    {
        "best_so_far",
        "budget_spent",
        "candidates",
        "children",
        "decision_rounds",
        "effective_sequential_rounds",
        "evaluator",
        "events",
        "journal_path",
        "roots",
        "tree",
    }
)
FORBIDDEN_ATTRS = FORBIDDEN_QUESTION_ATTRS | frozenset(
    {
        "format",
        "format_map",
        "reset",
        # Frame, code and traceback introspection can reach a caller's locals.
        "ag_frame",
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "f_trace",
        "gi_frame",
        "gi_yieldfrom",
        "tb_frame",
        "tb_next",
    }
)
ALLOWED_DUNDER_DEFS = frozenset({"__init__", "__post_init__"})
# Scores that carry no information about a recorded world (the sweep's own endpoints).
TRIVIAL_SCORES = frozenset({0.0, 0.5, 1.0})


class PolicyAuditError(ValueError):
    """Raised when generated policy code violates the prefix-only or sandbox contract."""


def _is_own_private(node: ast.Attribute) -> bool:
    """True for `self._x`, `cls._x` and `super().__init__`; dunders are otherwise refused."""
    if node.attr == "__init__" and isinstance(node.value, ast.Call):
        func = node.value.func
        return isinstance(func, ast.Name) and func.id == "super"
    if node.attr.startswith("__"):
        return False
    return isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}


def _parse(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise PolicyAuditError(f"Syntax error in policy code: {exc}") from exc


def _is_dataclass_decorator(node: ast.expr) -> bool:
    """True for `@dataclass`, `@dataclasses.dataclass` and their call forms."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id == "dataclass"
    return isinstance(target, ast.Attribute) and target.attr == "dataclass"


def _is_allowed_class(node: ast.ClassDef) -> bool:
    return node.name == "OptimalPolicy" or any(
        _is_dataclass_decorator(dec) for dec in node.decorator_list
    )


def _is_pure_literal(node: ast.expr) -> bool:
    """Constants, and tuples/sets/lists of them (for frozenset arguments only)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
        return all(_is_pure_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id == "frozenset" and all(_is_pure_literal(a) for a in node.args)
    return False


def _is_mutable_value(node: ast.expr) -> bool:
    """True for values that create mutable or stateful objects at import time."""
    if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp)):
        return True
    if isinstance(node, (ast.SetComp, ast.GeneratorExp)):
        return True
    if isinstance(node, ast.Tuple):
        return any(_is_mutable_value(elt) for elt in node.elts)
    if isinstance(node, ast.Call):
        return not _is_pure_literal(node)
    return False


def _check_class_level_state(body: list[ast.stmt], where: str) -> None:
    """Reject module- or class-level assignments that create mutable state."""
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            value: ast.expr | None = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            value = stmt.value
        else:
            continue
        if value is not None and _is_mutable_value(value):
            raise PolicyAuditError(f"Mutable state at {where} is forbidden")


def _check_names(node: ast.AST) -> None:
    """Reject forbidden builtin names in any reference, binding, argument or alias."""
    if isinstance(node, ast.Name):
        if node.id in FORBIDDEN_CALLS:
            raise PolicyAuditError(f"Forbidden builtin name {node.id!r}")
        if node.id.startswith("__") and node.id != "__future__":
            raise PolicyAuditError(f"Forbidden dunder reference {node.id!r}")
    elif isinstance(node, ast.arg) and node.arg in FORBIDDEN_CALLS:
        raise PolicyAuditError(f"Forbidden builtin name {node.arg!r}")
    elif isinstance(node, ast.alias):
        for name in (node.name, node.asname or ""):
            if name in FORBIDDEN_CALLS:
                raise PolicyAuditError(f"Forbidden builtin name {name!r}")


def audit_policy_source(source: str) -> ast.Module:
    """Verify imports, builtins, attribute access, definitions and the OptimalPolicy shape."""
    tree = _parse(source)
    has_optimal_cls = False
    _check_class_level_state(tree.body, "module level")
    for node in ast.walk(tree):
        _check_names(node)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    raise PolicyAuditError(f"Forbidden import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in ALLOWED_FROM_MODULES:
                raise PolicyAuditError(f"Forbidden from-import {mod!r}")
            allowed = ALLOWED_FROM_NAMES.get(mod)
            for alias in node.names:
                if allowed is not None and alias.name not in allowed:
                    raise PolicyAuditError(f"Forbidden name {alias.name!r} from {mod!r}")
                if alias.name == "*":
                    raise PolicyAuditError(f"Star import from {mod!r} is forbidden")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise PolicyAuditError(f"Forbidden builtin call {node.func.id!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                raise PolicyAuditError(
                    f"Prefix-only violation: accessing unallowed attribute {node.attr!r}"
                )
            if node.attr.startswith("_") and not _is_own_private(node):
                raise PolicyAuditError(f"Forbidden private/dunder attribute access {node.attr!r}")
        elif isinstance(node, ast.ClassDef):
            if not _is_allowed_class(node):
                raise PolicyAuditError(f"Class {node.name!r} is not allowed in policy code")
            _check_class_level_state(node.body, f"class {node.name}")
            if node.name == "OptimalPolicy":
                has_optimal_cls = True
                _check_optimal_methods(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name.startswith("__") and node.name not in ALLOWED_DUNDER_DEFS
        ):
            raise PolicyAuditError(f"Dunder method {node.name!r} is not allowed")

    if not has_optimal_cls:
        raise PolicyAuditError("Policy file must define class OptimalPolicy(LLMDesignedMethod)")
    return tree


def _check_optimal_methods(node: ast.ClassDef) -> None:
    methods = {
        item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "solve" not in methods:
        raise PolicyAuditError("OptimalPolicy must implement solve(self, question, budget=None)")
    if "plan_grid" not in methods:
        raise PolicyAuditError("OptimalPolicy must override plan_grid(self, context)")


def _has_three_decimals(score: float) -> bool:
    text = repr(float(score))
    if "." not in text or "e" in text:
        return False
    return len(text.split(".", 1)[1].rstrip("0")) >= 3


def audit_no_world_literals(source: str, trees: Iterable[DiscoveryTree]) -> None:
    """Reject string literals naming a recorded cell or world, and float literals copying a score."""
    cell_ids: set[str] = set()
    scores: list[float] = []
    for tree in trees:
        cell_ids.add(tree.world_id)
        for cid in tree.order:
            cell_ids.add(cid)
            score = tree.nodes[cid].score
            if score is not None and score not in TRIVIAL_SCORES and _has_three_decimals(score):
                scores.append(float(score))
    for node in ast.walk(_parse(source)):
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        if isinstance(value, str) and value in cell_ids:
            raise PolicyAuditError(f"String literal {value!r} copies a recorded cell or world id")
        if isinstance(value, float):
            for score in scores:
                if abs(value - score) < 1e-9:
                    raise PolicyAuditError(f"Float literal {value!r} copies a recorded node score")


def load_policy_class(policy_path: Path, unique_tag: str = "") -> type[LLMDesignedMethod]:
    """Audit and dynamically load `OptimalPolicy` from a standalone `policy.py`."""
    path = Path(policy_path).resolve()
    audit_policy_source(path.read_text(encoding="utf-8"))

    mod_name = f"rsi_dynamic_policy_{path.stem}_{abs(hash((str(path), unique_tag)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise PolicyAuditError(f"Unable to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        raise PolicyAuditError(f"Failed importing policy {path}: {exc}") from exc

    cls = getattr(module, "OptimalPolicy", None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, LLMDesignedMethod):
        raise PolicyAuditError("OptimalPolicy must subclass rsi.api.LLMDesignedMethod")
    if getattr(cls, "NAME", None) != "OptimalPolicy":
        raise PolicyAuditError("OptimalPolicy.NAME must equal 'OptimalPolicy'")
    return cls
