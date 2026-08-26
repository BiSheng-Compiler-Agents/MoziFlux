"""Tests for the guided-search MAP-Elites plugin."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_PROJECT_DIR = Path(__file__).parent.parent.resolve()
_PLUGIN_DIR = _PROJECT_DIR / ".hermes" / "plugins" / "guided-search"


def _update_run(sql: str, params: tuple) -> None:
    with sqlite3.connect(os.environ["GUIDED_SEARCH_DB_PATH"]) as conn:
        conn.execute(sql, params)


def _load_plugin():
    name = "guided_search_test_plugin"
    spec = importlib.util.spec_from_file_location(
        name,
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guided(monkeypatch, tmp_path):
    for name in ("REMOTE_VERIFY_HOST", "REMOTE_VERIFY_USER",
                 "REMOTE_VERIFY_PASS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("GUIDED_SEARCH_DB_PATH",
                       str(tmp_path / "guided.sqlite3"))
    return _load_plugin()


@pytest.fixture
def search_workspace(monkeypatch, tmp_path):
    root = tmp_path / "kernels"
    workspace = root / "l1_25_Swish"
    workspace.mkdir(parents=True)
    (workspace / "25_Swish.py").write_text("# baseline\n", encoding="utf-8")
    (workspace / "opt_25_Swish.py").write_text("# candidate\n",
                                               encoding="utf-8")
    (workspace / ".pipeline_state.json").write_text(
        json.dumps({
            "baseline": "25_Swish.py",
            "current_stage": "search",
            "guided_search": {
                "enabled": True,
                "run_id": None,
                "budget": 20,
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("KERNEL_SANDBOX_ROOT", str(root))
    return workspace


class TestAvailability:

    def test_checks_database_readiness_without_feature_env(
            self, guided, monkeypatch, tmp_path):
        assert guided._guided_search_available() is True
        not_a_database = tmp_path / "not-a-database"
        not_a_database.mkdir()
        monkeypatch.setenv("GUIDED_SEARCH_DB_PATH", str(not_a_database))
        assert guided._guided_search_available() is False


class TestDescriptors:

    def test_npu_option_inventory_matches_installed_triton_ascend(
            self, guided):
        from dataclasses import fields
        from triton.backends.ascend.compiler import NPUOptions

        installed = {field.name for field in fields(NPUOptions)}
        assert guided.NPU_OPTIONS == installed

    def test_new_npu_options_drive_relevant_behavior_bins(self, guided):
        desc = guided.classify_candidate(
            "def launch(): return None",
            {
                "limit_auto_multi_buffer_of_local_buffer": "no-limit",
                "disable_auto_inject_block_sync": True,
                "enable_persistent": True,
            },
        )
        assert desc.memory == 3
        assert desc.dispatch == 2
        assert desc.mechanism == "compiler_managed"

    def test_disabled_and_generic_npu_options_do_not_claim_managed_behavior(
            self, guided):
        source = """
def launch(kernel):
    kernel[(1,)](enable_preload=False, num_warps=8, num_stages=2)
"""
        desc = guided.classify_candidate(source)
        assert desc.memory == 0
        assert desc.mechanism == "standard_triton"

    def test_every_ordinal_bin_has_semantic_description_and_guidance(
            self, guided):
        assert set(guided.DESCRIPTOR_SCHEMA) == {
            "algorithm",
            "engine",
            "memory",
            "dispatch",
        }
        for dimension in guided.DESCRIPTOR_SCHEMA.values():
            assert set(dimension["bins"]) == {0, 1, 2, 3}
            for bin_spec in dimension["bins"].values():
                assert bin_spec["name"]
                assert bin_spec["description"]
                assert bin_spec["guidance"]

    def test_compiler_only_candidate_has_compiler_mechanism(self, guided):
        source = """
import triton
import triton.language as tl
@triton.jit
def kernel(x, y, n: tl.constexpr):
    offs = tl.arange(0, n)
    tl.store(y + offs, tl.load(x + offs))
def launch(x, y):
    kernel[(1,)](x, y, 128, enable_vf_fusion=True, enable_flatten=True)
"""
        desc = guided.classify_candidate(source)
        assert desc.mechanism == "compiler_managed"
        assert desc.memory >= 1

    def test_comments_and_docstrings_do_not_change_coordinate(self, guided):
        source = '''
"""flash_attention tl.dot tl.max tl.exp tl.sum al.multibuffer sync_block_set"""
def helper():
    """register blocking, ping-pong, single-pass, @triton.autotune"""
    # tl.make_block_ptr() enable_preload=True torch_npu.npu_fusion_attention()
    # sync_block_wait() tl.atomic_add() flash_attention()
    return 1
'''
        desc = guided.classify_candidate(source)
        assert desc.coordinate() == (0, 0, 0, 0, "standard_triton")

    def test_concrete_constructs_not_narrative_phrases_drive_bins(
            self, guided):
        source = '''
@triton.jit
def kernel(a, b, out):
    offsets = tl.arange(0, 64)
    for _ in range(2):
        m = tl.max(a, axis=0)
        p = tl.exp(a - m)
        l = tl.sum(p, axis=0)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        al.sync_block_set("vector", "cube", 0)
    tl.store(out + offsets, p / l)
'''
        desc = guided.classify_candidate(source)
        assert desc.algorithm == 2
        assert desc.engine == 0
        assert desc.memory == 0
        assert desc.mechanism == "manual_extension"

    def test_manual_extension_candidate_is_separate_lane(self, guided):
        source = """
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
with al.scope(core_mode="cube"):
    acc = tl.dot(a, b)
    al.sync_block_set("cube", "vector", 0)
with al.scope(core_mode="vector"):
    al.sync_block_wait("cube", "vector", 0)
"""
        desc = guided.classify_candidate(source)
        assert desc.mechanism == "manual_extension"
        assert desc.engine == 0

    def test_algorithm_bin_three_uses_structure_not_flash_name(self, guided):
        named_only = guided.classify_candidate("""
def flash_attention(x):
    return tl.load(x)
""")
        assert named_only.algorithm == 0

        structural = guided.classify_candidate("""
def kernel(q, k, v):
    acc = 0.0
    for block in range(4):
        scores = tl.dot(q, k)
        m = tl.max(scores, axis=0)
        probs = tl.exp(scores - m)
        scale = tl.sum(probs, axis=0)
        acc = acc + tl.dot(probs, v)
    return acc / scale
""")
        assert structural.algorithm == 2

    @pytest.mark.parametrize("expression", [
        "tl.dot(a, b)",
        "torch.matmul(a, b)",
        "torch.bmm(a, b)",
        "a @ b",
        "torch_npu.npu_bmmV2(a, b)",
    ])
    def test_every_matmul_form_is_cube(self, guided, expression):
        desc = guided.classify_candidate(
            f"def kernel(a, b):\n    return {expression}\n")
        assert desc.engine == 0
        assert any(item.rule == "cube_matmul" for item in desc.evidence)

    def test_shape_specialized_multi_path_dispatch(self, guided):
        source = """
@triton.jit
def kernel_n64(x): return x
@triton.jit
def kernel_n128(x): return x
def launch(x, n):
    if n == 64:
        return kernel_n64[(1,)](x)
    return kernel_n128[(1,)](x)
"""
        assert guided.classify_candidate(source).dispatch == 3

    def test_compute_pipeline_depth_is_ordinal(self, guided):
        implicit = guided.classify_candidate("""
def kernel(a, b):
    scores = tl.dot(a, b)
    return tl.exp(scores)
""")
        serial = guided.classify_candidate("""
def kernel(a, b):
    with al.scope(core_mode="cube"):
        scores = tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        return tl.exp(scores)
""")
        overlap = guided.classify_candidate("""
def kernel(a, b):
    with al.scope(core_mode="cube"):
        scores = tl.dot(a, b)
        al.multibuffer(scores, 2)
        al.sync_block_set("cube", "vector", 0)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        return tl.exp(scores)
""")
        assert implicit.engine == 1
        assert serial.engine == 2
        assert overlap.engine == 3

    def test_memory_bins_distinguish_accumulator_tiling_and_pipeline(
            self, guided):
        accumulator = guided.classify_candidate("""
def kernel(x):
    acc = 0.0
    for i in range(4):
        acc += tl.load(x + i)
    return acc
""")
        tiled = guided.classify_candidate("""
def kernel(x):
    block = tl.make_block_ptr(base=x, shape=(128,), strides=(1,),
                              offsets=(0,), block_shape=(64,), order=(0,))
    return tl.load(block)
""")
        pipelined = guided.classify_candidate("def kernel(x): return x",
                                              {"enable_preload": True})
        assert accumulator.memory == 1
        assert tiled.memory == 2
        assert pipelined.memory == 3

    def test_unreachable_experiments_do_not_change_coordinate(self, guided):
        source = """
@triton.jit
def _direct(x, y):
    offs = tl.arange(0, 64)
    tl.store(y + offs, tl.load(x + offs))

@triton.jit
def _unused_persistent_multibuffer(x, y):
    tile = tl.program_id(0)
    while tile < 1024:
        value = tl.load(x + tile)
        al.multibuffer(value, 2)
        tile += tl.num_programs(0)
    tl.store(y, value)

class ModelNew:
    def forward(self, x, y):
        _direct[(1,)](x, y)
        return y
"""
        desc = guided.classify_candidate(source)
        assert desc.coordinate() == (0, 0, 0, 0, "standard_triton")

    def test_attention_name_is_not_a_public_root(self, guided):
        source = """
@triton.jit
def _direct(x, y):
    tl.store(y, tl.load(x))

def attention(x):
    value = tl.dot(x, x)
    al.multibuffer(value, 2)
    return tl.exp(value)

class ModelNew:
    def forward(self, x, y):
        _direct[(1,)](x, y)
        return y
"""
        desc = guided.classify_candidate(source)
        assert desc.coordinate() == (0, 0, 0, 0, "standard_triton")

    def test_triton_jit_kernels_are_fallback_roots(self, guided):
        source = """
@triton.jit
def _matmul_kernel(a, b, out):
    tl.store(out, tl.dot(a, b))

def unused_helper(x):
    return tl.exp(x)
"""
        desc = guided.classify_candidate(source)
        assert desc.engine == 0
        assert any(item.rule == "cube_matmul" for item in desc.evidence)
        assert not any(item.rule == "vector_stage" for item in desc.evidence)

    def test_wrapper_name_containing_matmul_is_not_cube_evidence(self, guided):
        source = """
@triton.jit
def _vector_kernel(x, y):
    tl.store(y, tl.sum(tl.load(x + tl.arange(0, 64))))

def matmul_gelu_softmax(x, y):
    _vector_kernel[(1,)](x, y)

class ModelNew:
    def forward(self, x, y):
        matmul_gelu_softmax(x, y)
        return y
"""
        desc = guided.classify_candidate(source)
        assert not any(item.rule == "cube_matmul" for item in desc.evidence)
        assert desc.engine == 0

    def test_reachable_native_linear_is_cube_and_native_hybrid(self, guided):
        source = """
class ModelNew(nn.Module):
    def __init__(self, k, n):
        self.linear = nn.Linear(k, n)

    def forward(self, x):
        logits = self.linear(x)
        return tl.exp(logits)
"""
        desc = guided.classify_candidate(source)
        assert desc.engine == 1
        assert desc.mechanism == "native_hybrid"
        assert any(item.rule == "cube_matmul" for item in desc.evidence)

    def test_reachable_native_relu_is_vector_not_cube(self, guided):
        source = """
class ModelNew(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)
"""
        desc = guided.classify_candidate(source)
        assert desc.engine == 0
        assert desc.mechanism == "native_hybrid"
        assert any(item.rule == "vector_stage" for item in desc.evidence)
        assert not any(item.rule == "cube_matmul" for item in desc.evidence)

    def test_functional_alias_linear_is_native_cube(self, guided):
        source = """
import torch
import torch.nn.functional as F


def fused(x, weight):
    logits = F.linear(x, weight)
    return torch.softmax(F.gelu(logits), dim=1)


class ModelNew:
    def forward(self, x, weight):
        return fused(x, weight)
"""
        desc = guided.classify_candidate(source)
        assert desc.engine == 1
        assert desc.mechanism == "native_hybrid"
        assert any(item.rule == "cube_matmul" for item in desc.evidence)
        assert any(item.rule == "vector_stage" for item in desc.evidence)

    def test_two_phase_reduction_is_multi_stage_and_decomposed_dispatch(
            self, guided):
        source = """
@triton.jit
def _partial(x, partial):
    tl.store(partial, tl.sum(tl.load(x + tl.arange(0, 64))))

@triton.jit
def _finalize(partial, out):
    tl.store(out, tl.sum(tl.load(partial + tl.arange(0, 32))))

def launch(x, partial, out):
    _partial[(32,)](x, partial)
    _finalize[(1,)](partial, out)
"""
        desc = guided.classify_candidate(source)
        assert desc.algorithm == 3
        assert desc.dispatch == 2
        assert any(item.rule == "multi_stage_decomposition"
                   for item in desc.evidence)

    def test_persistent_loop_counter_is_not_an_accumulator(self, guided):
        source = """
@triton.jit
def _kernel_persistent(x, y, n, n_programs):
    tile_id = tl.program_id(0)
    while tile_id < n:
        value = tl.load(x + tile_id)
        tl.store(y + tile_id, value)
        tile_id += n_programs
"""
        desc = guided.classify_candidate(source)
        assert desc.memory == 0
        assert desc.dispatch == 2

    def test_contiguous_access_has_line_level_evidence(self, guided):
        source = """
def kernel(x):
    offsets = tl.arange(0, 64)
    offsets = tl.multiple_of(offsets, 16)
    offsets = tl.max_contiguous(offsets, 64)
    return tl.load(x + offsets)
"""
        desc = guided.classify_candidate(source)
        assert desc.memory == 1
        item = next(item for item in desc.evidence
                    if item.rule == "contiguous_aligned_access")
        assert item.line > 0

    def test_helper_refactor_preserves_online_algorithm_coordinate(
            self, guided):
        source = """
def _online(scores):
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    return tl.sum(p, axis=0)

def kernel(x):
    state = 0.0
    for block in range(4):
        state = state + _online(tl.load(x + block))
    return state
"""
        desc = guided.classify_candidate(source)
        assert desc.algorithm == 2
        assert any(item.rule == "online_or_scan_recurrence"
                   for item in desc.evidence)

    def test_candidate_fingerprint_includes_options(self, guided):
        a = guided.candidate_fingerprint("kernel", {"enable_flatten": False},
                                         {})
        b = guided.candidate_fingerprint("kernel", {"enable_flatten": True},
                                         {})
        assert a != b


class TestFitness:

    def test_simulation_fitness_uses_normalized_work(self, guided):
        baseline = {"useful_work": 1024.0, "wall_cycles": 2048.0}
        candidate = {"useful_work": 2048.0, "wall_cycles": 2048.0}
        assert guided.simulation_fitness(candidate,
                                         baseline) == pytest.approx(2.0)

    def test_simulation_fitness_rejects_incomparable_contracts(self, guided):
        baseline = {
            "useful_work": 1024.0,
            "wall_cycles": 1024.0,
            "probe_contract": "a",
        }
        candidate = {
            "useful_work": 1024.0,
            "wall_cycles": 512.0,
            "probe_contract": "b",
        }
        with pytest.raises(ValueError, match="probe contract"):
            guided.simulation_fitness(candidate, baseline)

    def test_hardware_fitness_always_uses_pytorch_acl(self, guided):
        score = guided.hardware_fitness(
            candidate_ms=[2.0, 4.0],
            pytorch_acl_ms=[4.0, 8.0],
        )
        assert score == pytest.approx(2.0)


class TestSelectionAndGradients:

    def test_gradient_temporal_decay_sorts_oldest_to_newest(self, guided):
        gradients = guided.estimate_gradients([
            {
                "id": 2,
                "created_at": "2026-01-02T00:00:00+00:00",
                "parent": {
                    "memory": 1
                },
                "child": {
                    "memory": 2
                },
                "delta_fitness": 1.0,
                "improved": True,
            },
            {
                "id": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "parent": {
                    "memory": 1
                },
                "child": {
                    "memory": 2
                },
                "delta_fitness": -1.0,
                "improved": False,
            },
        ],
                                              decay=0.5)
        assert gradients["memory"]["fitness"] > 0

    def test_gradient_shrinks_low_support(self, guided):
        gradients = guided.estimate_gradients([{
            "parent": {
                "memory": 1
            },
            "child": {
                "memory": 2
            },
            "delta_fitness": 0.5,
            "improved": True,
        }])
        assert gradients["memory"]["combined"] > 0
        assert 0 < gradients["memory"]["confidence"] < 1

    def test_gradient_adds_weighted_exploration_component(self, guided):
        parent = {
            "algorithm": 0,
            "engine": 0,
            "memory": 0,
            "dispatch": 0,
            "mechanism": "standard_triton",
            "hardware_score": 1.0,
        }
        gradients = guided.estimate_gradients([],
                                              parent=parent,
                                              elites=[parent])
        assert gradients["algorithm"]["fitness_normalized"] == 0.0
        assert gradients["algorithm"]["improvement_rate"] == 0.0
        assert gradients["algorithm"]["exploration"] == pytest.approx(1.0)
        assert gradients["algorithm"]["combined"] == pytest.approx(0.2)

    def test_exploration_gradient_can_select_uncovered_dimension(self, guided):
        parent = {
            "id": 1,
            "algorithm": 1,
            "engine": 1,
            "memory": 1,
            "dispatch": 1,
            "mechanism": "standard_triton",
            "hardware_score": 1.0,
            "correct": 1,
            "safe": 1,
        }
        # Keep every ordinal cell occupied except two cells directly above the
        # parent on dispatch. Equal-quality occupied cells have zero potential.
        elites = []
        candidate_id = 1
        for algorithm in range(4):
            for engine in range(4):
                for memory in range(4):
                    for dispatch in range(4):
                        if ((algorithm, engine, memory) == (1, 1, 1)
                                and dispatch in (2, 3)):
                            continue
                        elites.append({
                            "id": candidate_id,
                            "algorithm": algorithm,
                            "engine": engine,
                            "memory": memory,
                            "dispatch": dispatch,
                            "mechanism": "standard_triton",
                            "hardware_score": 1.0,
                            "correct": 1,
                            "safe": 1,
                        })
                        candidate_id += 1
        unrelated_transition = {
            "parent_json": {
                "algorithm": 0,
                "engine": 0,
                "memory": 0,
                "dispatch": 0,
                "mechanism": "standard_triton",
            },
            "child_json": {
                "algorithm": 3,
                "engine": 0,
                "memory": 0,
                "dispatch": 0,
                "mechanism": "standard_triton",
            },
            "delta_fitness": 100.0,
        }
        target = guided.select_target(parent, [unrelated_transition],
                                      elites,
                                      generation=2)
        assert target["dimension"] == "dispatch"
        assert target["direction"] == 1
        assert target["gradient"]["support"] == 0
        assert target["gradient"]["exploration"] == pytest.approx(1.0)
        assert target["gradient"]["combined"] == pytest.approx(0.2)

    def test_parent_selection_is_reproducible_and_gradient_weighted(
            self, guided):
        candidates = [
            {
                "id": 1,
                "correct": 1,
                "safe": 1,
                "algorithm": 0,
                "engine": 0,
                "memory": 0,
                "dispatch": 0,
                "mechanism": "standard_triton",
                "simulation_score": 1.0,
                "hardware_score": None
            },
            {
                "id": 2,
                "correct": 1,
                "safe": 1,
                "algorithm": 1,
                "engine": 0,
                "memory": 1,
                "dispatch": 1,
                "mechanism": "compiler_managed",
                "simulation_score": 2.0,
                "hardware_score": None
            },
        ]
        transitions = [{
            "id": index,
            "created_at": f"2026-01-{index:02d}T00:00:00+00:00",
            "parent_json": {
                "algorithm": 1,
                "engine": 0,
                "memory": 1,
                "dispatch": 1,
                "mechanism": "compiler_managed",
            },
            "child_json": {
                "algorithm": 1,
                "engine": 0,
                "memory": 2,
                "dispatch": 1,
                "mechanism": "compiler_managed",
            },
            "delta_fitness": 1.0,
        } for index in range(1, 6)]
        first, mode = guided.select_parent(candidates,
                                           transitions,
                                           generation=7,
                                           run_id=3,
                                           archive_revision=9)
        repeated, _ = guided.select_parent(candidates,
                                           transitions,
                                           generation=7,
                                           run_id=3,
                                           archive_revision=9)
        assert mode == "gradient_weighted"
        assert first is not None and repeated is not None
        assert first["id"] == repeated["id"]

        counts = {1: 0, 2: 0}
        for generation in range(1, 201):
            selected, _ = guided.select_parent(
                candidates,
                transitions,
                generation=generation,
                run_id=3,
                archive_revision=generation,
            )
            counts[selected["id"]] += 1
        assert counts[2] > counts[1]

    def test_target_contains_descriptive_bin_transition(self, guided):
        parent = {
            "id": 1,
            "algorithm": 1,
            "engine": 2,
            "memory": 3,
            "dispatch": 1,
            "mechanism": "standard_triton",
        }
        target = guided.select_target(parent, [], generation=3)
        assert target["dimension"] == "memory"
        assert target["direction"] == -1  # +1 flips at the upper boundary.
        assert target["current_name"] == "pipelined_buffering"
        assert target["target_name"] == "explicit_tiling_or_local_buffer"
        assert "explicit tiling" in target["mutation_guidance"]

    def test_prompt_exposes_complete_gradient_and_nonmandatory_hint(
            self, guided):
        parent = {
            "id": 7,
            "source_hash": "parent-hash",
            "algorithm": 0,
            "engine": 0,
            "memory": 0,
            "dispatch": 0,
            "mechanism": "standard_triton",
            "hardware_score": 1.0,
            "correct": 1,
            "safe": 1,
            "descriptor_evidence_json": "[]",
        }
        selected, _ = guided.select_parent([parent], [],
                                           generation=1,
                                           run_id=3,
                                           archive_revision=1)
        target = guided.select_target(selected, [], [parent], generation=1)
        target["selection_mode"] = "gradient_weighted"
        attempt = {
            "id": 9,
            "generation": 1,
            "phase": "editing",
            "parent_candidate_id": 7,
            "target_json": json.dumps(target),
            "mutation_objective": "follow gradient evidence",
        }
        run = {
            "id": 3,
            "budget": 30,
            "promotion_domain": "hardware",
        }
        rendered = guided.render_attempt_prompt(
            run=run,
            attempt=attempt,
            parent=selected,
        )
        assert "PARENT GRADIENT SAMPLING" in rendered
        assert "FULL SIGNED GRADIENT VECTOR" in rendered
        for dimension in ("algorithm", "engine", "memory", "dispatch"):
            assert f"- {dimension}:" in rendered
        assert "not a mandatory target cell" in rendered

    def test_mechanism_target_is_descriptive_and_categorical(self, guided):
        parent = {
            "id": 1,
            "algorithm": 0,
            "engine": 0,
            "memory": 0,
            "dispatch": 0,
            "mechanism": "standard_triton",
        }
        target = guided.select_target(parent, [], generation=5)
        assert target["mechanism_target"] == "compiler_managed"
        assert target["mechanism_target_label"] == "compiler-managed Triton"
        assert "BishengIR/NPUOptions" in target["mechanism_guidance"]

    def test_mechanism_exploration_excludes_current_lane(self, guided):
        parent = {
            "id": 1,
            "algorithm": 0,
            "engine": 0,
            "memory": 0,
            "dispatch": 0,
            "mechanism": "compiler_managed",
        }
        transitions = [{
            "parent_json": parent,
            "child_json": {
                **parent, "mechanism": "compiler_managed"
            },
            "delta_fitness": 10.0,
        }, {
            "parent_json": parent,
            "child_json": {
                **parent, "mechanism": "manual_extension"
            },
            "delta_fitness": 1.0,
        }]
        target = guided.select_target(parent, transitions, generation=5)
        assert target["mechanism_target"] == "manual_extension"


class TestSqlArchive:

    def test_descriptor_evidence_is_persisted(self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            promotion_domain="simulation",
        )
        source = """
def kernel(x):
    offsets = tl.max_contiguous(tl.arange(0, 64), 64)
    return tl.load(x + offsets)
"""
        descriptor = guided.classify_candidate(source)
        candidate_id = guided.record_candidate(
            run_id=run_id,
            source=source,
            descriptor=descriptor,
            correct=True,
            safe=True,
            complete_attempt=False,
        )
        candidate = guided.get_candidate(candidate_id)
        evidence = json.loads(candidate["descriptor_evidence_json"])
        assert any(item["rule"] == "contiguous_aligned_access"
                   for item in evidence)

    def test_baseline_elite_is_required_before_mutation_budget(
            self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
            promotion_domain="simulation",
        )
        baseline_attempt = guided.ensure_active_attempt(run_id)
        assert baseline_attempt["kind"] == "baseline"
        assert baseline_attempt["generation"] == 0
        baseline = guided.get_candidate(
            baseline_attempt["parent_candidate_id"])
        assert baseline is not None
        guided.record_candidate(
            run_id=run_id,
            source=baseline["source_text"],
            descriptor=(
                baseline["algorithm"],
                baseline["engine"],
                baseline["memory"],
                baseline["dispatch"],
                baseline["mechanism"],
            ),
            correct=True,
            safe=True,
            simulation_score=1.0,
        )
        summary = guided.run_summary(run_id)
        assert summary["completed_attempts"] == 0
        mutation_attempt = guided.ensure_active_attempt(run_id)
        assert mutation_attempt["kind"] == "mutation"
        assert mutation_attempt["generation"] == 1

    def test_failed_baseline_is_terminal_and_cannot_finalize(
            self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
            promotion_domain="simulation",
        )
        attempt = guided.ensure_active_attempt(run_id)
        baseline = guided.get_candidate(attempt["parent_candidate_id"])
        assert baseline is not None
        guided.record_candidate(
            run_id=run_id,
            source=baseline["source_text"],
            descriptor=(
                baseline["algorithm"],
                baseline["engine"],
                baseline["memory"],
                baseline["dispatch"],
                baseline["mechanism"],
            ),
            correct=False,
            safe=True,
        )
        run = guided.get_run(run_id)
        assert run is not None
        assert run["status"] == "failed_baseline_evaluation"
        assert run["completed_attempts"] == 0
        with pytest.raises(RuntimeError, match="not active"):
            guided.ensure_active_attempt(run_id)
        with pytest.raises(RuntimeError, match="failed_baseline_evaluation"):
            guided.finalize_run(run_id)

    def test_attempt_is_sticky_until_terminal(self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=20,
        )
        first = guided.ensure_active_attempt(run_id)
        second = guided.ensure_active_attempt(run_id)
        assert first["id"] == second["id"]
        assert first["status"] == "active"

    def test_concurrent_attempt_creation_returns_one_sticky_attempt(
            self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=2,
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            attempts = list(
                pool.map(lambda _index: guided.ensure_active_attempt(run_id),
                         range(16)))
        assert len({attempt["id"] for attempt in attempts}) == 1
        with sqlite3.connect(os.environ["GUIDED_SEARCH_DB_PATH"]) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM attempts WHERE run_id=? AND status='active'",
                (run_id, )).fetchone()[0]
        assert count == 1

    def test_tool_event_recording_is_idempotent(self, guided,
                                                search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
        )
        guided.ensure_active_attempt(run_id)
        kwargs = {
            "run_id": run_id,
            "session_id": "kernelbench-l1_25_Swish",
            "tool_call_id": "same-tool-call",
            "tool_name": "remote_verify",
            "source_content_hash": "source-hash",
        }
        assert guided.record_tool_event(**kwargs,
                                        result={
                                            "success": False,
                                            "error": "first"
                                        }) is True
        assert guided.record_tool_event(**kwargs, result={"success":
                                                          True}) is False
        with sqlite3.connect(os.environ["GUIDED_SEARCH_DB_PATH"]) as conn:
            rows = conn.execute(
                "SELECT result_json FROM tool_events WHERE run_id=?",
                (run_id, )).fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0][0])["error"] == "first"

    def test_checkout_materializes_parent_only_once(self, guided,
                                                    search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=20,
        )
        attempt = guided.ensure_active_attempt(run_id)
        assert attempt["phase"] == "checkout"
        first = json.loads(
            guided._handle_checkout({
                "session_id": "kernelbench-l1_25_Swish",
            }))
        assert first["ok"] is True
        assert (search_workspace / "opt_25_Swish.py").read_text(
            encoding="utf-8") == "# baseline\n"
        second = json.loads(
            guided._handle_checkout({
                "session_id": "kernelbench-l1_25_Swish",
            }))
        assert "error" in second
        assert "refusing to overwrite" in second["error"]

    def test_elite_promotion_uses_run_evaluation_domain(
            self, guided, search_workspace):
        simulation_run = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=20,
            promotion_domain="simulation",
        )
        sim_id = guided.record_candidate(
            run_id=simulation_run,
            source="# sim",
            descriptor=(0, 0, 1, 1, "standard_triton"),
            correct=True,
            safe=True,
            simulation_score=1.5,
            complete_attempt=False,
        )
        guided.record_candidate(
            run_id=simulation_run,
            source="# hw",
            descriptor=(0, 0, 1, 1, "standard_triton"),
            correct=True,
            safe=True,
            hardware_score=1.2,
            complete_attempt=False,
        )
        elites = guided.get_cell_elites(simulation_run,
                                        (0, 0, 1, 1, "standard_triton"))
        assert elites["simulation_candidate_id"] == sim_id
        assert elites["hardware_candidate_id"] is None
        _update_run(
            "UPDATE search_runs SET status='ready_to_finalize', "
            "completed_attempts=budget WHERE id=?", (simulation_run, ))
        simulation_winner = guided.finalize_run(simulation_run)
        assert simulation_winner["candidate_id"] == sim_id

        pipeline_path = search_workspace / ".pipeline_state.json"
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        pipeline["guided_search"]["run_id"] = None
        pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
        hardware_run = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish-hardware",
            budget=20,
            promotion_domain="hardware",
        )
        ignored_sim = guided.record_candidate(
            run_id=hardware_run,
            source="# ignored sim",
            descriptor=(0, 0, 1, 1, "standard_triton"),
            correct=True,
            safe=True,
            simulation_score=9.0,
            complete_attempt=False,
        )
        _update_run(
            "UPDATE search_runs SET status='ready_to_finalize', "
            "completed_attempts=budget WHERE id=?", (hardware_run, ))
        with pytest.raises(RuntimeError, match="hardware-confirmed"):
            guided.finalize_run(hardware_run)
        promoted_hw = guided.record_candidate(
            run_id=hardware_run,
            source="# promoted hw",
            descriptor=(0, 0, 1, 1, "standard_triton"),
            correct=True,
            safe=True,
            hardware_score=1.1,
            complete_attempt=False,
        )
        hardware_elites = guided.get_cell_elites(
            hardware_run, (0, 0, 1, 1, "standard_triton"))
        assert ignored_sim != promoted_hw
        assert hardware_elites["simulation_candidate_id"] is None
        assert hardware_elites["hardware_candidate_id"] == promoted_hw

    def test_finalize_materializes_hardware_confirmed_winner(
            self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
            promotion_domain="hardware",
        )
        guided.ensure_active_attempt(run_id)
        source = (search_workspace / "25_Swish.py").read_text(encoding="utf-8")
        descriptor = guided.classify_candidate(source)
        guided.record_candidate(
            run_id=run_id,
            source=source,
            descriptor=descriptor,
            correct=True,
            safe=True,
            hardware_score=1.25,
        )
        guided.ensure_active_attempt(run_id)
        winner_source = source + "\n# mutation winner\n"
        candidate_id = guided.record_candidate(
            run_id=run_id,
            source=winner_source,
            descriptor=guided.classify_candidate(winner_source),
            correct=True,
            safe=True,
            hardware_score=1.3,
        )
        result = guided.finalize_run(run_id)
        assert result["candidate_id"] == candidate_id
        assert (search_workspace /
                "opt_25_Swish.py").read_text(encoding="utf-8") == winner_source
        run = guided.get_run(run_id)
        assert run is not None and run["status"] == "completed"

    def test_state_rejects_new_candidate_without_active_attempt(
            self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
            promotion_domain="simulation",
        )
        with pytest.raises(RuntimeError, match="requires an active"):
            guided.record_candidate(
                run_id=run_id,
                source="# unbound child",
                descriptor=(0, 0, 0, 0, "standard_triton"),
                correct=False,
                safe=False,
            )

    def test_state_rejects_premature_and_active_finalization(
            self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
            promotion_domain="simulation",
        )
        with pytest.raises(RuntimeError, match="budget is exhausted"):
            guided.finalize_run(run_id)
        guided.ensure_active_attempt(run_id)
        with pytest.raises(RuntimeError, match="attempt is active"):
            guided.finalize_run(run_id)

    def test_completed_run_rejects_archive_mutation(self, guided,
                                                    search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
        )
        _update_run("UPDATE search_runs SET status='completed' WHERE id=?",
                    (run_id, ))
        with pytest.raises(RuntimeError, match="does not accept candidates"):
            guided.record_candidate(
                run_id=run_id,
                source="# late candidate",
                descriptor=(0, 0, 0, 0, "standard_triton"),
                correct=True,
                safe=True,
                simulation_score=1.0,
                complete_attempt=False,
            )

    def test_evaluator_evidence_requires_semantic_success_or_failure(
            self, guided, search_workspace):
        run_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=1,
            promotion_domain="hardware",
        )
        attempt = guided.ensure_active_attempt(run_id)
        source_hash = "source-hash"
        guided.record_tool_event(
            run_id=run_id,
            session_id="kernelbench-l1_25_Swish",
            tool_call_id="contradictory",
            tool_name="remote_verify",
            source_content_hash=source_hash,
            result={
                "success": False,
                "test_passed": True,
                "bench_passed": True,
            },
        )
        assert guided.successful_tool_evidence(
            run_id,
            {"remote_verify"},
            source_content_hash=source_hash,
            attempt_id=attempt["id"],
        ) is None
        assert guided.has_matching_evaluator_event(
            run_id,
            source_content_hash=source_hash,
            attempt_id=attempt["id"],
        ) is True

        guided.record_tool_event(
            run_id=run_id,
            session_id="kernelbench-l1_25_Swish",
            tool_call_id="successful",
            tool_name="remote_verify",
            source_content_hash="successful-source",
            result={
                "success": True,
                "test_passed": True,
                "bench_passed": True,
            },
        )
        assert guided.has_matching_evaluator_event(
            run_id,
            source_content_hash="successful-source",
            attempt_id=attempt["id"],
        ) is False

    def test_pipeline_run_id_cannot_be_reused_by_another_workspace(
            self, guided, search_workspace, tmp_path):
        first_id = guided.initialize_search_run(
            workspace=search_workspace,
            session_id="kernelbench-l1_25_Swish",
            budget=20,
        )
        other = tmp_path / "kernels" / "l1_other"
        other.mkdir(parents=True)
        (other / "1_Other.py").write_text("# other\n", encoding="utf-8")
        (other / ".pipeline_state.json").write_text(json.dumps({
            "baseline":
            "1_Other.py",
            "current_stage":
            "search",
            "guided_search": {
                "enabled": True,
                "run_id": first_id,
                "budget": 20,
            },
        }),
                                                    encoding="utf-8")
        second_id = guided.initialize_search_run(
            workspace=other,
            session_id="kernelbench-l1_other",
            budget=20,
        )
        assert second_id != first_id
        second = guided.get_run(second_id)
        assert second is not None
        assert second["workspace"] == str(other.resolve())


class TestHooks:

    def test_pre_verify_does_not_own_tool_budget_abandonment(
            self, guided, search_workspace):
        guided._on_pre_llm_call(session_id="kernelbench-l1_25_Swish",
                                user_message="start")
        attempt = guided.get_active_attempt(1)
        assert attempt is not None
        for _ in range(guided._attempt_tool_budget()):
            guided.increment_attempt_counter(1, "tool_calls_used")
        result = guided._on_pre_verify(session_id="kernelbench-l1_25_Swish")
        active = guided.get_active_attempt(1)
        assert active is not None and active["id"] == attempt["id"]
        assert result is not None
        assert "still in phase" in result["message"]

    def test_pre_llm_reuses_active_attempt(self, guided, search_workspace):
        first = guided._on_pre_llm_call(
            session_id="kernelbench-l1_25_Swish",
            user_message="optimize",
        )
        second = guided._on_pre_llm_call(
            session_id="kernelbench-l1_25_Swish",
            user_message="continue",
        )
        assert first is not None and second is not None
        assert "GUIDED SEARCH" in first["context"]
        first_attempt = first["context"].split("ATTEMPT: ",
                                               1)[1].splitlines()[0]
        second_attempt = second["context"].split("ATTEMPT: ",
                                                 1)[1].splitlines()[0]
        assert first_attempt == second_attempt

    def test_llm_request_middleware_refreshes_phase_without_accumulation(
            self, guided, search_workspace):
        clean_request = {
            "messages": [{
                "role": "user",
                "content": "optimize"
            }],
        }
        first = guided._on_llm_execution(
            request=clean_request,
            next_call=lambda request: request,
            session_id="kernelbench-l1_25_Swish",
            api_request_id="turn:api:1",
            api_call_count=1,
        )
        assert first is not None
        first_content = first["messages"][-1]["content"]
        assert "PHASE: checkout" in first_content
        assert "BASELINE CALIBRATION" in first_content
        assert "not part of the mutation budget" in first_content

        guided._handle_checkout({"session_id": "kernelbench-l1_25_Swish"})
        second = guided._on_llm_execution(
            request=clean_request,
            next_call=lambda request: request,
            session_id="kernelbench-l1_25_Swish",
            api_request_id="turn:api:2",
            api_call_count=2,
        )
        assert second is not None
        second_content = second["messages"][-1]["content"]
        assert "PHASE: editing" in second_content
        assert "PHASE: checkout\n" not in second_content
        assert clean_request["messages"][0]["content"] == "optimize"

        # A transport retry has the same logical request ID and is not counted twice.
        guided._on_llm_execution(
            request=clean_request,
            next_call=lambda request: request,
            session_id="kernelbench-l1_25_Swish",
            api_request_id="turn:api:2",
            api_call_count=2,
        )
        status = json.loads(
            guided._handle_status({
                "session_id": "kernelbench-l1_25_Swish",
            }))
        assert status["active_attempt"]["llm_iterations_used"] == 2

    def test_llm_request_middleware_supports_responses_input_blocks(
            self, guided, search_workspace):
        request = {
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "optimize"
                }],
            }],
        }
        result = guided._on_llm_execution(
            request=request,
            next_call=lambda request: request,
            session_id="kernelbench-l1_25_Swish",
            api_request_id="turn:api:1",
            api_call_count=1,
            api_mode="codex_responses",
        )
        assert result is not None
        assert result["input"][-1]["role"] == "user"
        blocks = result["input"][-1]["content"]
        assert blocks[-1]["type"] == "input_text"
        assert "GUIDED SEARCH" in blocks[-1]["text"]

    def test_llm_request_middleware_appends_user_context_after_tool_only_input(
            self, guided, search_workspace):
        request = {
            "input": [{
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "tool result",
            }],
        }
        result = guided._on_llm_execution(
            request=request,
            next_call=lambda request: request,
            session_id="kernelbench-l1_25_Swish",
            api_request_id="turn:api:tool-only",
            api_call_count=2,
            api_mode="codex_responses",
        )
        assert result is not None
        assert result["input"][-1]["role"] == "user"
        assert result["input"][-1]["content"][-1]["type"] == "input_text"
        assert "GUIDED SEARCH" in result["input"][-1]["content"][-1]["text"]

    def test_disabled_workspace_is_noop(self, guided, search_workspace):
        state_path = search_workspace / ".pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["guided_search"]["enabled"] = False
        state_path.write_text(json.dumps(state), encoding="utf-8")
        result = guided._on_pre_llm_call(
            session_id="kernelbench-l1_25_Swish",
            user_message="optimize",
        )
        assert result is None

    def test_unevaluated_submission_does_not_consume_budget(
            self, guided, search_workspace):
        guided._on_pre_llm_call(session_id="kernelbench-l1_25_Swish",
                                user_message="start")
        guided._handle_checkout({"session_id": "kernelbench-l1_25_Swish"})
        candidate = search_workspace / "opt_25_Swish.py"
        candidate.write_text("# not evaluated\n", encoding="utf-8")
        rejected = json.loads(
            guided._handle_submit({
                "session_id": "kernelbench-l1_25_Swish",
                "candidate_path": str(candidate),
                "correct": True,
            }))
        assert "error" in rejected
        status = json.loads(
            guided._handle_status({
                "session_id": "kernelbench-l1_25_Swish",
            }))
        assert status["completed_attempts"] == 0
        assert status["active_attempt"] is not None

    def test_hardware_score_is_source_bound_and_derived_from_remote_output(
            self, guided, search_workspace, tmp_path, monkeypatch):
        monkeypatch.setenv("REMOTE_VERIFY_HOST", "npu")
        monkeypatch.setenv("REMOTE_VERIFY_USER", "user")
        monkeypatch.setenv("REMOTE_VERIFY_PASS", "pass")
        guided._on_pre_llm_call(session_id="kernelbench-l1_25_Swish",
                                user_message="start")
        guided._handle_checkout({"session_id": "kernelbench-l1_25_Swish"})
        candidate = search_workspace / "opt_25_Swish.py"
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / candidate.name).write_text(
            candidate.read_text(encoding="utf-8"), encoding="utf-8")
        bench = ("bench:\n"
                 "  label  PyTorch / ACL  Optimized Triton\n"
                 "0 shape             4.0               2.0\n")
        guided._on_post_tool_call(
            tool_name="remote_verify",
            args={"local_dir": str(artifact_dir)},
            result={
                "success": True,
                "test_passed": True,
                "bench_passed": True,
                "bench_output": bench,
            },
            session_id="kernelbench-l1_25_Swish",
            tool_call_id="remote-1",
        )
        submitted = json.loads(
            guided._handle_submit({
                "session_id": "kernelbench-l1_25_Swish",
                "candidate_path": str(candidate),
                "correct": True,
                "hardware": {
                    "candidate_ms": [1e-18],
                    "pytorch_acl_ms": [1e18],
                },
            }))
        assert submitted["hardware_score"] == pytest.approx(2.0)

        # The same evidence cannot authorize a changed source.
        candidate.write_text("# unrelated source\n", encoding="utf-8")
        rejected = json.loads(
            guided._handle_submit({
                "session_id": "kernelbench-l1_25_Swish",
                "candidate_path": str(candidate),
                "correct": True,
                "hardware": {},
            }))
        assert "error" in rejected
        assert "archived source" in rejected["error"]

    def test_simulation_score_comes_from_source_bound_trace_cycles(
            self, guided, search_workspace, tmp_path):
        guided._on_pre_llm_call(session_id="kernelbench-l1_25_Swish",
                                user_message="start")
        checkout = json.loads(
            guided._handle_checkout({
                "session_id": "kernelbench-l1_25_Swish",
            }))
        candidate = search_workspace / "opt_25_Swish.py"

        def record_trace(name, source, duration, tool_call_id):
            artifact = tmp_path / name
            artifact.mkdir()
            (artifact / candidate.name).write_text(source, encoding="utf-8")
            trace = artifact / "trace_core0.json"
            trace.write_text(json.dumps([
                {
                    "ph": "X",
                    "ts": 0,
                    "dur": duration,
                    "pid": 1,
                    "tid": 1
                },
            ]),
                             encoding="utf-8")
            guided._on_post_tool_call(
                tool_name="cannsim_local_run",
                args={"local_dir": str(artifact)},
                result={
                    "success": True,
                    "cannsim_log_tail": "[HOST] PASS",
                    "trace_local_path": str(trace),
                    "trace_json_size_bytes": 2000,
                },
                session_id="kernelbench-l1_25_Swish",
                tool_call_id=tool_call_id,
            )

        baseline_source = candidate.read_text(encoding="utf-8")
        baseline_hash = hashlib.sha256(
            baseline_source.encode("utf-8")).hexdigest()
        assert checkout["ok"] is True
        record_trace("baseline", baseline_source, 100, "sim-base")
        record_trace("candidate", baseline_source, 50, "sim-candidate")
        submitted = json.loads(
            guided._handle_submit({
                "session_id": "kernelbench-l1_25_Swish",
                "candidate_path": str(candidate),
                "correct": True,
                "simulation": {
                    "candidate": {
                        "probe_contract": "equal-k"
                    },
                    "baseline": {
                        "probe_contract": "equal-k",
                        "source_content_hash": baseline_hash,
                    },
                },
            }))
        assert submitted["simulation_score"] == pytest.approx(1.0)


class TestRegistration:

    def test_register_uses_matching_availability_gate(self, guided):

        class Context:

            def __init__(self):
                self.hooks = []
                self.middleware = []
                self.tools = []

            def register_hook(self, name, callback):
                self.hooks.append((name, callback))

            def register_middleware(self, name, callback):
                self.middleware.append((name, callback))

            def register_tool(self, **kwargs):
                self.tools.append(kwargs)

        ctx = Context()
        guided.register(ctx)
        assert {name
                for name, _ in ctx.hooks} >= {
                    "pre_tool_call",
                    "post_tool_call",
                    "post_llm_call",
                    "pre_verify",
                    "on_session_end",
                }
        assert "pre_llm_call" not in {name for name, _ in ctx.hooks}
        assert ctx.middleware == [("llm_execution", guided._on_llm_execution)]
        assert ctx.tools
        for tool in ctx.tools:
            assert tool["check_fn"] is guided._guided_search_available
            assert tool["requires_env"] == []
        submit = next(tool for tool in ctx.tools
                      if tool["name"] == "guided_search_submit_candidate")
        simulation = submit["schema"]["parameters"]["properties"]["simulation"]
        assert "useful_work" not in simulation["properties"]["candidate"][
            "properties"]
        assert "wall_cycles" not in simulation["properties"]["candidate"][
            "properties"]
