"""Reachable-AST optimization features for Ascend kernel descriptors."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


_REDUCTION_CALLS = {"sum", "max", "min", "reduce", "associative_scan", "cumsum", "cumprod"}
_NONLINEAR_CALLS = {
    "abs", "cos", "erf", "exp", "exp2", "gelu", "log", "log2", "maximum",
    "minimum", "relu", "sigmoid", "silu", "sin", "softmax", "sqrt", "tanh", "where",
}
_ACCUMULATOR_NAMES = {
    "acc", "accumulator", "carry", "mean", "running_max", "running_sum", "row_max",
    "row_sum", "scale", "state", "sum", "total", "var", "variance",
}


@dataclass(frozen=True)
class DescriptorEvidence:
    dimension: str
    rule: str
    line: int
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "rule": self.rule,
            "line": self.line,
            "detail": self.detail,
        }


@dataclass
class SourceFeatures:
    cube_ops: int = 0
    vector_ops: int = 0
    fused_kernel: bool = False
    online_recurrence: bool = False
    scan: bool = False
    multi_stage: bool = False
    accumulator_reuse: bool = False
    contiguous_access: bool = False
    explicit_tiling: bool = False
    buffered_pipeline: bool = False
    explicit_cv_handoff: bool = False
    overlapped_cv_pipeline: bool = False
    tuned_direct: bool = False
    persistent: bool = False
    multi_path: bool = False
    native_call: bool = False
    manual_extension: bool = False
    extension_hint: bool = False
    compiler_managed: bool = False
    enabled_npu_options: set[str] = field(default_factory=set)
    evidence: list[DescriptorEvidence] = field(default_factory=list)

    def add(self, dimension: str, rule: str, node: ast.AST | None, detail: str) -> None:
        item = DescriptorEvidence(
            dimension=dimension,
            rule=rule,
            line=int(getattr(node, "lineno", 0) or 0),
            detail=detail,
        )
        if item not in self.evidence:
            self.evidence.append(item)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return ""


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower()


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    head, separator, tail = name.partition(".")
    resolved = aliases.get(head)
    if not resolved:
        return name
    return f"{resolved}.{tail}" if separator else resolved


def _is_cube_call(name: str) -> bool:
    lowered = name.lower()
    leaf = _leaf(name)
    return (
        leaf in {"dot", "dot_scaled", "matmul", "mm", "bmm", "batch_matmul", "npu_bmmv2"}
        or lowered == "torch.nn.functional.linear"
        or leaf.startswith("aclnnmatmul")
        or leaf.startswith("aclnnbatchmatmul")
    )


def _is_native_call(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("torch_npu.")
        or lowered.startswith("torch.nn.functional.")
        or "aclnn" in lowered
        or lowered in {"torch.bmm", "torch.matmul", "torch.mm"}
    )


def _native_module_attributes(
    tree: ast.AST, aliases: dict[str, str]
) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        constructor = _resolve_alias(_call_name(value.func), aliases)
        lowered = constructor.lower()
        if not (lowered.startswith("nn.") or lowered.startswith("torch.nn.")):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                attributes[target.attr] = constructor
    return attributes


def _function_index(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.setdefault(node.name, node)
    return result


def _reachable_functions(
    tree: ast.AST,
) -> tuple[dict[str, ast.FunctionDef | ast.AsyncFunctionDef], set[str]]:
    functions = _function_index(tree)
    model_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for child in node.body:
                if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == "forward"):
                    functions[child.name] = child
                    model_roots.add(child.name)
    if model_roots:
        roots = model_roots
    else:
        roots = {name for name in functions if name in {"launch", "run"}}
    if not roots:
        roots = {
            name for name, function in functions.items()
            if any("triton.jit" in _call_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator)
                for decorator in function.decorator_list)
        }
    if not roots:
        roots = set(functions)
    reachable: set[str] = set()
    queue = list(sorted(roots))
    while queue:
        name = queue.pop(0)
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            called = _leaf(_call_name(node.func))
            if called in functions and called not in reachable:
                queue.append(called)
    return functions, reachable


def _nodes_for(functions: Mapping[str, ast.AST], reachable: Iterable[str]) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for name in sorted(reachable):
        nodes.extend(ast.walk(functions[name]))
    return nodes


def _expanded_call_leaves(
    node: ast.AST, functions: Mapping[str, ast.AST]
) -> set[str]:
    leaves: set[str] = set()
    pending = [node]
    expanded: set[str] = set()
    while pending:
        current = pending.pop()
        for child in ast.walk(current):
            if not isinstance(child, ast.Call):
                continue
            called = _leaf(_call_name(child.func))
            leaves.add(called)
            if called in functions and called not in expanded:
                expanded.add(called)
                pending.append(functions[called])
    return leaves


def _enabled_keyword(value: ast.AST) -> bool:
    if isinstance(value, ast.Constant):
        return value.value not in (None, False, 0, "")
    return True


def _assigned_names(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> set[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    else:
        targets = [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _numerical_accumulator(nodes: Iterable[ast.AST]) -> tuple[bool, ast.AST | None]:
    nodes = list(nodes)
    for node in nodes:
        if isinstance(node, ast.Call) and _is_cube_call(_call_name(node.func)):
            if any(keyword.arg == "acc" for keyword in node.keywords):
                return True, node
    loops = [node for node in nodes if isinstance(node, (ast.For, ast.AsyncFor, ast.While))]
    for loop in loops:
        induction = set()
        if isinstance(loop, (ast.For, ast.AsyncFor)) and isinstance(loop.target, ast.Name):
            induction.add(loop.target.id)
        for node in ast.walk(loop):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            assigned = _assigned_names(node) - induction
            if not assigned:
                continue
            value = node.value
            if value is None:
                continue
            referenced = {
                child.id for child in ast.walk(value) if isinstance(child, ast.Name)
            }
            self_carried = assigned & referenced
            if isinstance(node, ast.AugAssign):
                self_carried = assigned
            if not self_carried:
                continue
            calls = [
                _call_name(child.func) for child in ast.walk(value)
                if isinstance(child, ast.Call)
            ]
            meaningful_name = any(
                name.lower() in _ACCUMULATOR_NAMES
                or name.lower().startswith(("acc", "sum", "running_"))
                for name in self_carried
            )
            numerical_value = any(
                _is_cube_call(name) or _leaf(name) in _REDUCTION_CALLS
                or _leaf(name) in {"load", "maximum", "minimum"}
                for name in calls
            )
            if meaningful_name or numerical_value:
                return True, node
    return False, None


def _kernel_launch(node: ast.Call, functions: Mapping[str, ast.AST]) -> str | None:
    if not isinstance(node.func, ast.Subscript):
        return None
    name = _leaf(_call_name(node.func))
    return name if name in functions else None


def _branch_launches(
    statements: Iterable[ast.stmt], functions: Mapping[str, ast.AST]
) -> set[str]:
    launches: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                target = _kernel_launch(node, functions)
                if target:
                    launches.add(target)
                elif _is_native_call(_call_name(node.func)):
                    launches.add(f"native:{_call_name(node.func)}")
    return launches


def _shape_or_regime_test(test: ast.AST) -> bool:
    nodes = list(ast.walk(test))
    has_threshold = any(
        isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool))
        for node in nodes
    )
    has_regime = any(
        (isinstance(node, ast.Attribute) and node.attr in {"shape", "dtype", "ndim", "numel"})
        or (isinstance(node, ast.Name) and node.id.lower() in {
            "b", "c", "d", "h", "k", "m", "n", "shape", "use_acl", "use_triton",
            "n_elements", "n_tiles", "total_tiles",
        })
        for node in nodes
    )
    has_regime_flag = any(
        isinstance(node, ast.Name)
        and node.id.lower() in {"use_acl", "use_triton", "force_persistent"}
        for node in nodes
    )
    return (has_threshold and has_regime) or has_regime_flag


def extract_source_features(
    source: str,
    launch_options: Mapping[str, Any],
    *,
    npu_options: set[str] | frozenset[str],
    compiler_managed_options: set[str],
    memory_tiling_options: set[str],
    memory_reuse_options: set[str],
    memory_pipeline_options: set[str],
    coordination_options: set[str],
    dispatch_options: set[str],
    environment: Mapping[str, Any] | None = None,
) -> SourceFeatures:
    """Extract optimization features only from public-entrypoint-reachable code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    functions, reachable = _reachable_functions(tree)
    nodes = (_nodes_for(functions, reachable)
             if functions else list(ast.walk(tree)))
    features = SourceFeatures()

    aliases = _import_aliases(tree)
    call_records = [
        (_resolve_alias(_call_name(node.func), aliases), node)
        for node in nodes if isinstance(node, ast.Call)
    ]
    native_modules = _native_module_attributes(tree, aliases)
    native_module_records = [
        (name, node, native_modules[_leaf(name)])
        for name, node in call_records
        if name.startswith("self.") and _leaf(name) in native_modules
    ]
    native_cube_records = [
        (name, node) for name, node, constructor in native_module_records
        if _leaf(constructor).startswith(("linear", "bilinear", "conv"))
    ]
    matmul_nodes = [node for node in nodes if isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.MatMult)]
    cube_records = [(name, node) for name, node in call_records if _is_cube_call(name)]
    features.cube_ops = (
        len(cube_records) + len(native_cube_records) + len(matmul_nodes))
    if cube_records or native_cube_records or matmul_nodes:
        node = (cube_records[0][1] if cube_records
                else native_cube_records[0][1] if native_cube_records
                else matmul_nodes[0])
        features.add("engine", "cube_matmul", node,
                     "reachable dot/BMM/matmul is assigned to Cube")

    vector_records = [
        (name, node) for name, node in call_records
        if _leaf(name) in _REDUCTION_CALLS | _NONLINEAR_CALLS
    ]
    features.vector_ops = len(vector_records)
    if vector_records:
        features.add("engine", "vector_stage", vector_records[0][1],
                     "reachable reduction/nonlinear work uses Vector")

    reachable_kernel_names = {
        target for node in nodes if isinstance(node, ast.Call)
        for target in [_kernel_launch(node, functions)] if target
    }
    reachable_kernel_names.update(
        name for name in reachable
        if any("triton.jit" in _call_name(decorator.func if isinstance(decorator, ast.Call)
                                           else decorator)
               for decorator in functions[name].decorator_list)
    )

    for name in sorted(reachable_kernel_names):
        function_nodes = list(ast.walk(functions[name]))
        leaves = {
            _leaf(_call_name(node.func)) for node in function_nodes if isinstance(node, ast.Call)
        }
        families = {
            family for family, present in {
                "cube": any(_is_cube_call(_call_name(node.func)) for node in function_nodes
                            if isinstance(node, ast.Call)),
                "reduction": bool(leaves & _REDUCTION_CALLS),
                "nonlinear": bool(leaves & _NONLINEAR_CALLS),
                "atomic": any(leaf.startswith("atomic_") for leaf in leaves),
            }.items() if present
        }
        if len(families) >= 2:
            features.fused_kernel = True
            features.add("algorithm", "fused_kernel", functions[name],
                         f"reachable kernel {name} combines {', '.join(sorted(families))}")
            break

    loop_nodes = [node for node in nodes if isinstance(node, (ast.For, ast.AsyncFor, ast.While))]
    call_leaves = {_leaf(name) for name, _ in call_records}
    features.scan = bool(call_leaves & {"associative_scan", "cumsum", "cumprod"})
    online_node = next((
        node for node in loop_nodes
        if {"max", "exp", "sum"}.issubset(
            _expanded_call_leaves(node, functions))
    ), None)
    features.online_recurrence = online_node is not None or features.scan
    if features.online_recurrence:
        features.add("algorithm", "online_or_scan_recurrence", online_node,
                     "reachable loop carries online reduction/scan state")

    accumulator, accumulator_node = _numerical_accumulator(nodes)
    features.accumulator_reuse = accumulator
    if accumulator:
        features.add("memory", "accumulator_reuse", accumulator_node,
                     "numerical state is explicitly reused across loop work")

    sequential_launches: set[str] = set()
    for root in reachable:
        function = functions[root]
        direct_launches: set[str] = set()
        for statement in function.body:
            if isinstance(statement, ast.If):
                continue
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                target = _kernel_launch(node, functions)
                if target:
                    direct_launches.add(target)
                elif _is_native_call(_call_name(node.func)):
                    direct_launches.add(f"native:{_call_name(node.func)}")
        if len(direct_launches) >= 2:
            sequential_launches.update(direct_launches)
    features.multi_stage = len(sequential_launches) >= 2
    if features.multi_stage:
        features.add("algorithm", "multi_stage_decomposition", None,
                     f"sequential launches: {', '.join(sorted(sequential_launches))}")

    enabled_options = {
        key for key, value in launch_options.items()
        if key in npu_options and value not in (None, False, 0, "")
    }
    for name, node in call_records:
        del name
        for keyword in node.keywords:
            if keyword.arg in npu_options and _enabled_keyword(keyword.value):
                enabled_options.add(str(keyword.arg))
    features.enabled_npu_options = enabled_options
    features.compiler_managed = bool(enabled_options & compiler_managed_options)

    contiguous_calls = [
        node for name, node in call_records
        if _leaf(name) in {"contiguous", "max_contiguous", "multiple_of"}
    ]
    features.contiguous_access = bool(contiguous_calls)
    if contiguous_calls:
        features.add("memory", "contiguous_aligned_access", contiguous_calls[0],
                     "reachable code asserts or materializes contiguous/aligned access")

    tiling_calls = [
        node for name, node in call_records
        if _leaf(name) in {"make_block_ptr", "advance", "alloc", "compile_hint"}
    ]
    swizzle_names = {
        node.id for node in nodes if isinstance(node, ast.Name)
        and node.id.upper() in {"GROUP_M", "GROUP_N", "NUM_PID_M", "NUM_PID_N"}
    }
    features.explicit_tiling = bool(
        tiling_calls or swizzle_names or enabled_options & (memory_tiling_options | memory_reuse_options)
    )
    if features.explicit_tiling:
        features.add("memory", "explicit_tile_or_reuse", tiling_calls[0] if tiling_calls else None,
                     "reachable block-pointer/local-buffer/swizzle reuse is present")

    buffering_calls = [
        node for name, node in call_records if _leaf(name) == "multibuffer"
    ]
    features.buffered_pipeline = bool(
        buffering_calls or enabled_options & memory_pipeline_options
    )
    if features.buffered_pipeline:
        features.add("memory", "buffered_hierarchy", buffering_calls[0] if buffering_calls else None,
                     "reachable multibuffer/preload/cache pipeline is enabled")

    scope_modes: set[str] = set()
    sync_nodes: list[ast.Call] = []
    extension_hint_nodes: list[ast.Call] = []
    manual_nodes: list[ast.Call] = []
    native_nodes: list[ast.Call] = []
    for name, node in call_records:
        leaf = _leaf(name)
        if leaf == "scope":
            manual_nodes.append(node)
            for keyword in node.keywords:
                if keyword.arg == "core_mode" and isinstance(keyword.value, ast.Constant):
                    scope_modes.add(str(keyword.value.value).lower())
        if leaf in {"sync_block_set", "sync_block_wait"}:
            sync_nodes.append(node)
            manual_nodes.append(node)
        if leaf in {"alloc", "sub_vec_id"} and name.startswith(("bl.", "al.")):
            manual_nodes.append(node)
        if leaf in {"compile_hint", "multibuffer", "cast", "extract_slice", "insert_slice"}:
            extension_hint_nodes.append(node)
        if _is_native_call(name):
            native_nodes.append(node)

    features.explicit_cv_handoff = bool(
        features.cube_ops and features.vector_ops
        and ({"cube", "vector"}.issubset(scope_modes) or sync_nodes)
    )
    if features.explicit_cv_handoff:
        features.add("engine", "explicit_cv_handoff", sync_nodes[0] if sync_nodes else None,
                     "reachable Cube/Vector scopes or synchronization form a handoff")
    features.overlapped_cv_pipeline = bool(
        features.explicit_cv_handoff
        and (features.buffered_pipeline or enabled_options & coordination_options)
    )
    if features.overlapped_cv_pipeline:
        features.add("engine", "overlapped_cv_pipeline", None,
                     "handoff combines buffering/preload or compiler CV coordination")

    features.manual_extension = bool(manual_nodes)
    features.extension_hint = bool(extension_hint_nodes)
    features.native_call = bool(native_nodes or native_module_records)

    autotune_nodes = [
        function for name in reachable for function in [functions[name]]
        if any("autotune" in _call_name(decorator.func if isinstance(decorator, ast.Call)
                                         else decorator)
               for decorator in function.decorator_list)
    ]
    features.tuned_direct = bool(autotune_nodes or swizzle_names)
    if features.tuned_direct:
        features.add("dispatch", "tuned_or_swizzled", autotune_nodes[0] if autotune_nodes else None,
                     "reachable autotune or GROUP_M/N swizzle changes direct scheduling")

    persistent_node = next((
        node for name, node in call_records if _leaf(name) == "num_programs"
    ), None)
    persistent_name = next((functions[name] for name in reachable if "persistent" in name.lower()), None)
    features.persistent = bool(
        persistent_node or persistent_name or enabled_options & dispatch_options
    )
    if features.persistent:
        features.add("dispatch", "persistent_or_grid_stride", persistent_node or persistent_name,
                     "reachable persistent/grid-stride traversal is present")

    multi_path_node: ast.If | None = None
    for name in sorted(reachable):
        function = functions[name]
        candidate_ifs: list[tuple[ast.If, list[ast.stmt]]] = []
        for index, statement in enumerate(function.body):
            if isinstance(statement, ast.If):
                fallback = statement.orelse or function.body[index + 1:]
                candidate_ifs.append((statement, fallback))
        for node in ast.walk(function):
            if isinstance(node, ast.If) and node.orelse:
                candidate_ifs.append((node, node.orelse))
        for node, fallback in candidate_ifs:
            body = _branch_launches(node.body, functions)
            other = _branch_launches(fallback, functions)
            if body and other and body != other and _shape_or_regime_test(node.test):
                multi_path_node = node
                break
        if multi_path_node:
            break
    features.multi_path = multi_path_node is not None
    if features.multi_path:
        features.add("dispatch", "shape_regime_multi_path", multi_path_node,
                     "reachable host branch selects distinct kernel/native paths")

    environment = environment or {}
    if environment.get("TRITON_ALL_BLOCKS_PARALLEL"):
        features.persistent = True
        features.compiler_managed = True
        features.add("dispatch", "all_blocks_parallel", None,
                     "TRITON_ALL_BLOCKS_PARALLEL enables compiler-managed traversal")

    return features
