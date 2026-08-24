"""SQLite persistence for guided-search MAP-Elites runs."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .archive import select_parent, select_target
from .descriptors import BehaviorDescriptor, candidate_fingerprint, classify_candidate

SCHEMA_VERSION = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> Path:
    explicit = os.environ.get("GUIDED_SEARCH_DB_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return (home / "guided-search" / "guided_search.sqlite3").resolve()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema() -> None:
    with connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS search_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                kernel_name TEXT NOT NULL,
                target_soc TEXT NOT NULL DEFAULT '',
                compiler_fingerprint TEXT NOT NULL DEFAULT '',
                evaluation_fingerprint TEXT NOT NULL DEFAULT '',
                promotion_domain TEXT NOT NULL DEFAULT 'simulation',
                status TEXT NOT NULL DEFAULT 'running',
                failure_reason TEXT NOT NULL DEFAULT '',
                budget INTEGER NOT NULL DEFAULT 20,
                wall_time_limit_seconds REAL,
                completed_attempts INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0,
                winner_candidate_id INTEGER,
                winner_source_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_search_runs_session
                ON search_runs(session_id, status);
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
                parent_candidate_id INTEGER REFERENCES candidates(id),
                prompt_variant_id INTEGER,
                source_hash TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_text TEXT NOT NULL,
                algorithm INTEGER NOT NULL,
                engine INTEGER NOT NULL,
                memory INTEGER NOT NULL,
                dispatch INTEGER NOT NULL,
                mechanism TEXT NOT NULL,
                descriptor_evidence_json TEXT NOT NULL DEFAULT '[]',
                correct INTEGER NOT NULL DEFAULT 0,
                safe INTEGER NOT NULL DEFAULT 0,
                simulation_score REAL,
                hardware_score REAL,
                launch_options_json TEXT NOT NULL DEFAULT '{}',
                environment_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(run_id, source_hash)
            );
            CREATE TABLE IF NOT EXISTS elite_cells (
                run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
                algorithm INTEGER NOT NULL,
                engine INTEGER NOT NULL,
                memory INTEGER NOT NULL,
                dispatch INTEGER NOT NULL,
                mechanism TEXT NOT NULL,
                simulation_candidate_id INTEGER REFERENCES candidates(id),
                simulation_score REAL,
                hardware_candidate_id INTEGER REFERENCES candidates(id),
                hardware_score REAL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, algorithm, engine, memory, dispatch, mechanism)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
                generation INTEGER NOT NULL,
                parent_candidate_id INTEGER REFERENCES candidates(id),
                kind TEXT NOT NULL DEFAULT 'mutation',
                target_json TEXT NOT NULL,
                mutation_objective TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT 'checkout',
                status TEXT NOT NULL DEFAULT 'active',
                tool_calls_used INTEGER NOT NULL DEFAULT 0,
                llm_iterations_used INTEGER NOT NULL DEFAULT 0,
                last_llm_request_id TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                current_source_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_attempt
                ON attempts(run_id) WHERE status = 'active';
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
                attempt_id INTEGER REFERENCES attempts(id),
                parent_candidate_id INTEGER REFERENCES candidates(id),
                child_candidate_id INTEGER REFERENCES candidates(id),
                delta_fitness REAL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tool_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
                attempt_id INTEGER REFERENCES attempts(id),
                session_id TEXT NOT NULL,
                tool_call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                source_content_hash TEXT,
                result_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(session_id, tool_call_id)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
                attempt_id INTEGER REFERENCES attempts(id),
                candidate_id INTEGER REFERENCES candidates(id),
                artifact_type TEXT NOT NULL,
                path TEXT,
                content_hash TEXT,
                summary_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prompt_variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
                philosophy TEXT NOT NULL DEFAULT '',
                strategies TEXT NOT NULL DEFAULT '',
                pitfalls TEXT NOT NULL DEFAULT '',
                analysis_guidance TEXT NOT NULL DEFAULT '',
                fitness REAL,
                support_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """)
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION), ),
            )
        elif int(row["value"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported guided-search schema version: {row['value']}")
        candidate_columns = {
            item["name"]
            for item in conn.execute(
                "PRAGMA table_info(candidates)").fetchall()
        }
        if "descriptor_evidence_json" not in candidate_columns:
            conn.execute(
                """ALTER TABLE candidates ADD COLUMN descriptor_evidence_json
                   TEXT NOT NULL DEFAULT '[]'""")


def _pipeline_path(workspace: Path) -> Path:
    return workspace / ".pipeline_state.json"


def _load_pipeline(workspace: Path) -> dict[str, Any]:
    path = _pipeline_path(workspace)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_pipeline(workspace: Path, state: dict[str, Any]) -> None:
    path = _pipeline_path(workspace)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def initialize_search_run(
    *,
    workspace: Path,
    session_id: str,
    budget: int = 20,
    target_soc: str = "",
    compiler_fingerprint: str = "",
    evaluation_fingerprint: str = "",
    wall_time_limit_seconds: float | None = None,
    promotion_domain: str = "simulation",
) -> int:
    ensure_schema()
    workspace = Path(workspace).resolve()
    pipeline = _load_pipeline(workspace)
    guided = pipeline.setdefault("guided_search", {})
    existing_id = guided.get("run_id")
    if existing_id:
        with connect() as conn:
            existing = conn.execute(
                "SELECT workspace FROM search_runs WHERE id=?",
                (int(existing_id), )).fetchone()
            if (existing is not None
                    and Path(existing["workspace"]).resolve() == workspace):
                return int(existing_id)
        # A stale or injected run ID from another workspace is never reusable.
        guided["run_id"] = None

    now = utcnow()
    with connect() as conn:
        cursor = conn.execute(
            """INSERT INTO search_runs(
                   session_id, workspace, kernel_name, target_soc,
                   compiler_fingerprint, evaluation_fingerprint, promotion_domain, budget,
                   wall_time_limit_seconds, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id,
                str(workspace),
                workspace.name,
                target_soc,
                compiler_fingerprint,
                evaluation_fingerprint,
                "hardware" if promotion_domain == "hardware" else "simulation",
                max(1, int(budget)),
                wall_time_limit_seconds,
                now,
                now,
            ),
        )
        assert cursor.lastrowid is not None
        run_id = int(cursor.lastrowid)

    baseline_name = pipeline.get("baseline")
    baseline_path = workspace / baseline_name if baseline_name else None
    if baseline_path and baseline_path.exists():
        source = baseline_path.read_text(encoding="utf-8")
        descriptor = classify_candidate(source)
        record_candidate(
            run_id=run_id,
            source=source,
            descriptor=descriptor,
            correct=True,
            safe=True,
            complete_attempt=False,
        )

    seed_run_id = guided.get("seed_run_id")
    if seed_run_id:
        with connect() as conn:
            seed_rows = conn.execute(
                """SELECT DISTINCT c.* FROM candidates c
                   JOIN elite_cells e ON e.run_id=c.run_id AND
                     (e.simulation_candidate_id=c.id OR e.hardware_candidate_id=c.id)
                   WHERE c.run_id=?""",
                (int(seed_run_id), ),
            ).fetchall()
        for seed in seed_rows:
            record_candidate(
                run_id=run_id,
                source=seed["source_text"],
                descriptor=(seed["algorithm"], seed["engine"], seed["memory"],
                            seed["dispatch"], seed["mechanism"]),
                correct=True,
                safe=True,
                launch_options=json.loads(seed["launch_options_json"] or "{}"),
                environment=json.loads(seed["environment_json"] or "{}"),
                complete_attempt=False,
            )

    guided.update({
        "enabled": True,
        "run_id": run_id,
        "budget": max(1, int(budget)),
        "revision": int(guided.get("revision", 0)) + 1,
        "status": "running",
    })
    pipeline["current_stage"] = "search"
    _save_pipeline(workspace, pipeline)
    return run_id


def get_run(run_id: int) -> dict[str, Any] | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT * FROM search_runs WHERE id=?",
                           (run_id, )).fetchone()
        return dict(row) if row else None


def get_candidate(candidate_id: int | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id=?",
                           (candidate_id, )).fetchone()
        return dict(row) if row else None


def get_candidate_by_content_hash(run_id: int,
                                  content_hash: str) -> dict[str, Any] | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM candidates WHERE run_id=? AND content_hash=?
               ORDER BY id DESC LIMIT 1""",
            (run_id, content_hash),
        ).fetchone()
        return dict(row) if row else None


def get_active_attempt(run_id: int) -> dict[str, Any] | None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM attempts WHERE run_id=? AND status='active'",
            (run_id, )).fetchone()
        return dict(row) if row else None


def run_summary(run_id: int) -> dict[str, Any]:
    run = get_run(run_id)
    if run is None:
        raise KeyError(f"unknown guided-search run: {run_id}")
    active = get_active_attempt(run_id)
    with connect() as conn:
        occupied = conn.execute(
            "SELECT COUNT(*) AS n FROM elite_cells WHERE run_id=?",
            (run_id, )).fetchone()["n"]
        sim = conn.execute(
            "SELECT COUNT(*) AS n FROM elite_cells WHERE run_id=? "
            "AND simulation_candidate_id IS NOT NULL",
            (run_id, )).fetchone()["n"]
        hardware = conn.execute(
            "SELECT COUNT(*) AS n FROM elite_cells WHERE run_id=? "
            "AND hardware_candidate_id IS NOT NULL",
            (run_id, )).fetchone()["n"]
    return {
        **run,
        "active_attempt": active,
        "occupied_cells": int(occupied),
        "simulation_elites": int(sim),
        "hardware_elites": int(hardware),
    }


def _selection_data(
        conn: sqlite3.Connection,
        run_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run = conn.execute("SELECT promotion_domain FROM search_runs WHERE id=?",
                       (run_id, )).fetchone()
    if run is None:
        raise KeyError(f"unknown guided-search run: {run_id}")
    elite_column = ("hardware_candidate_id" if run["promotion_domain"]
                    == "hardware" else "simulation_candidate_id")
    elite_ids = [
        row[0] for row in conn.execute(
            f"SELECT {elite_column} FROM elite_cells WHERE run_id=? "
            f"AND {elite_column} IS NOT NULL",
            (run_id, ),
        ).fetchall()
    ]
    if elite_ids:
        placeholders = ",".join("?" for _ in elite_ids)
        candidate_rows = conn.execute(
            f"SELECT * FROM candidates WHERE id IN ({placeholders})",
            elite_ids,
        ).fetchall()
    else:
        # Before the first promotion, baseline/prior-run seeds are the only
        # eligible parents; unevaluated children never become parents.
        candidate_rows = conn.execute(
            """SELECT * FROM candidates WHERE run_id=? AND correct=1 AND safe=1
               AND simulation_score IS NULL AND hardware_score IS NULL""",
            (run_id, ),
        ).fetchall()
    candidates = [dict(row) for row in candidate_rows]
    transitions = []
    rows = conn.execute(
        """SELECT t.*, p.algorithm AS p_algorithm, p.engine AS p_engine,
                  p.memory AS p_memory, p.dispatch AS p_dispatch,
                  p.mechanism AS p_mechanism,
                  c.algorithm AS c_algorithm, c.engine AS c_engine,
                  c.memory AS c_memory, c.dispatch AS c_dispatch,
                  c.mechanism AS c_mechanism
           FROM transitions t
           LEFT JOIN candidates p ON p.id=t.parent_candidate_id
           LEFT JOIN candidates c ON c.id=t.child_candidate_id
           WHERE t.run_id=?
           ORDER BY t.created_at ASC, t.id ASC""",
        (run_id, ),
    ).fetchall()
    for row in rows:
        item = dict(row)
        if row["p_algorithm"] is not None and row["c_algorithm"] is not None:
            item["parent_json"] = {
                "algorithm": row["p_algorithm"],
                "engine": row["p_engine"],
                "memory": row["p_memory"],
                "dispatch": row["p_dispatch"],
                "mechanism": row["p_mechanism"],
            }
            item["child_json"] = {
                "algorithm": row["c_algorithm"],
                "engine": row["c_engine"],
                "memory": row["c_memory"],
                "dispatch": row["c_dispatch"],
                "mechanism": row["c_mechanism"],
            }
        transitions.append(item)
    return candidates, transitions


def _authoritative_elite_exists(conn: sqlite3.Connection, run_id: int,
                                promotion_domain: str) -> bool:
    candidate_column = ("hardware_candidate_id" if promotion_domain
                        == "hardware" else "simulation_candidate_id")
    row = conn.execute(
        f"""SELECT 1 FROM elite_cells WHERE run_id=?
            AND {candidate_column} IS NOT NULL LIMIT 1""",
        (run_id, ),
    ).fetchone()
    return row is not None


def _set_exhausted_status(conn: sqlite3.Connection, run: sqlite3.Row,
                          reason: str) -> str:
    has_elite = _authoritative_elite_exists(conn, int(run["id"]),
                                            str(run["promotion_domain"]))
    status = "ready_to_finalize" if has_elite else "failed_no_valid_elite"
    failure_reason = "" if has_elite else reason
    conn.execute(
        """UPDATE search_runs SET status=?, failure_reason=?, updated_at=?
           WHERE id=?""",
        (status, failure_reason, utcnow(), run["id"]),
    )
    return status


def ensure_active_attempt(run_id: int) -> dict[str, Any]:
    ensure_schema()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute(
            "SELECT * FROM attempts WHERE run_id=? AND status='active'",
            (run_id, )).fetchone()
        if active:
            return dict(active)
        run = conn.execute("SELECT * FROM search_runs WHERE id=?",
                           (run_id, )).fetchone()
        if run is None:
            raise KeyError(f"unknown guided-search run: {run_id}")
        if run["status"] not in ("running", "ready_to_finalize"):
            raise RuntimeError(f"run {run_id} is not active: {run['status']}")
        wall_limit = run["wall_time_limit_seconds"]
        if wall_limit is not None:
            started = datetime.fromisoformat(run["created_at"])
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            if elapsed >= float(wall_limit):
                status = _set_exhausted_status(
                    conn, run, "wall-time limit reached without a valid elite")
                raise RuntimeError(
                    "guided-search wall-time limit reached" if status ==
                    "ready_to_finalize" else
                    "guided-search failed: no valid elite before wall-time limit"
                )

        if not _authoritative_elite_exists(conn, run_id,
                                           str(run["promotion_domain"])):
            baseline_attempt = conn.execute(
                """SELECT * FROM attempts WHERE run_id=? AND kind='baseline'
                   ORDER BY id LIMIT 1""",
                (run_id, ),
            ).fetchone()
            if baseline_attempt is not None:
                reason = (
                    baseline_attempt["last_error"] or
                    "baseline evaluation did not produce an authoritative elite"
                )
                conn.execute(
                    """UPDATE search_runs SET status='failed_baseline_evaluation',
                       failure_reason=?, updated_at=? WHERE id=?""",
                    (reason, utcnow(), run_id),
                )
                raise RuntimeError(
                    f"guided-search baseline evaluation failed: {reason}")

            baseline = conn.execute(
                """SELECT * FROM candidates WHERE run_id=?
                   AND parent_candidate_id IS NULL ORDER BY id LIMIT 1""",
                (run_id, ),
            ).fetchone()
            if baseline is None:
                conn.execute(
                    """UPDATE search_runs SET status='failed_baseline_evaluation',
                       failure_reason='baseline candidate is missing', updated_at=?
                       WHERE id=?""",
                    (utcnow(), run_id),
                )
                raise RuntimeError(
                    "guided-search baseline candidate is missing")
            coordinate = [
                int(baseline["algorithm"]),
                int(baseline["engine"]),
                int(baseline["memory"]),
                int(baseline["dispatch"]),
                str(baseline["mechanism"]),
            ]
            target = {
                "kind": "baseline_evaluation",
                "parent_coordinate": coordinate,
                "selection_mode": "baseline",
                "promotion_domain": str(run["promotion_domain"]),
            }
            now = utcnow()
            cursor = conn.execute(
                """INSERT INTO attempts(
                       run_id, generation, parent_candidate_id, kind, target_json,
                       mutation_objective, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    0,
                    baseline["id"],
                    "baseline",
                    json.dumps(target),
                    "Evaluate the unchanged baseline with the run's authoritative "
                    "evaluator and submit its structured evidence. Do not optimize "
                    "or change direction during baseline calibration.",
                    now,
                    now,
                ),
            )
            assert cursor.lastrowid is not None
            return dict(
                conn.execute("SELECT * FROM attempts WHERE id=?",
                             (cursor.lastrowid, )).fetchone())

        if int(run["completed_attempts"]) >= int(run["budget"]):
            status = _set_exhausted_status(
                conn, run, "candidate budget exhausted without a valid elite")
            raise RuntimeError(
                "guided-search candidate budget exhausted" if status ==
                "ready_to_finalize" else
                "guided-search failed: candidate budget exhausted without a valid elite"
            )
        generation = int(run["completed_attempts"]) + 1
        candidates, transitions = _selection_data(conn, run_id)
        parent, selection_mode = select_parent(
            candidates,
            transitions,
            generation=generation,
            run_id=run_id,
            archive_revision=int(run["revision"]),
        )
        parent_id = int(parent["id"]) if parent else None
        target = select_target(parent,
                               transitions,
                               candidates,
                               generation=generation)
        target["selection_mode"] = selection_mode
        target["parent_gradient_magnitude"] = float(
            parent.get("_selection_gradient_magnitude", 0.0) if parent else 0.0
        )
        target["parent_selection_probability"] = float(
            parent.get("_selection_probability", 1.0) if parent else 1.0)
        dimension_label = str(target["dimension_label"])
        mechanism_note = (f" Also explore {target['mechanism_target_label']}: "
                          f"{target['mechanism_guidance']}"
                          if target.get("mechanism_target") else "")
        objective = (
            f"Use the selected parent's signed gradient vector as optimization "
            f"guidance. The dominant hint is {dimension_label} from "
            f"{target['current_name']} toward {target['target_name']}: "
            f"{target['mutation_guidance']} Behavioral bins are measured outcomes, "
            "not mandatory constraints. Preserve semantics and evaluate one focused "
            f"candidate before changing direction.{mechanism_note}")
        now = utcnow()
        cursor = conn.execute(
            """INSERT INTO attempts(
                   run_id, generation, parent_candidate_id, target_json,
                   mutation_objective, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?)""",
            (run_id, generation, parent_id, json.dumps(target), objective, now,
             now),
        )
        assert cursor.lastrowid is not None
        return dict(
            conn.execute("SELECT * FROM attempts WHERE id=?",
                         (cursor.lastrowid, )).fetchone())


def _descriptor_tuple(
    descriptor: tuple[int, int, int, int, str] | BehaviorDescriptor,
) -> tuple[int, int, int, int, str]:
    if isinstance(descriptor, BehaviorDescriptor):
        return descriptor.coordinate()
    if len(descriptor) != 5:
        raise ValueError(
            "descriptor must have four ordinal dimensions and a mechanism")
    return (
        int(descriptor[0]),
        int(descriptor[1]),
        int(descriptor[2]),
        int(descriptor[3]),
        str(descriptor[4]),
    )


def record_candidate(
    *,
    run_id: int,
    source: str,
    descriptor: tuple[int, int, int, int, str] | BehaviorDescriptor,
    correct: bool,
    safe: bool,
    simulation_score: float | None = None,
    hardware_score: float | None = None,
    launch_options: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    complete_attempt: bool = True,
) -> int:
    ensure_schema()
    algorithm, engine, memory, dispatch, mechanism = _descriptor_tuple(
        descriptor)
    descriptor_evidence = ([item.as_dict() for item in descriptor.evidence] if
                           isinstance(descriptor, BehaviorDescriptor) else [])
    source_hash = candidate_fingerprint(source, launch_options or {},
                                        environment or {})
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    now = utcnow()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run_row = conn.execute(
            "SELECT promotion_domain, status FROM search_runs WHERE id=?",
            (run_id, )).fetchone()
        if run_row is None:
            raise KeyError(f"unknown guided-search run: {run_id}")
        if run_row["status"] not in {"running", "ready_to_finalize"}:
            raise RuntimeError(
                f"run {run_id} does not accept candidates: {run_row['status']}"
            )
        promotion_domain = str(run_row["promotion_domain"])
        attempt = conn.execute(
            "SELECT * FROM attempts WHERE run_id=? AND status='active'",
            (run_id, )).fetchone()
        parent_id = int(
            attempt["parent_candidate_id"]
        ) if attempt and attempt["parent_candidate_id"] else None
        existing_candidate = conn.execute(
            "SELECT id FROM candidates WHERE run_id=? AND source_hash=?",
            (run_id, source_hash),
        ).fetchone()
        if complete_attempt and attempt is None and existing_candidate is None:
            raise RuntimeError(
                "a new candidate requires an active guided-search attempt")
        if attempt and attempt["kind"] == "baseline" and parent_id:
            baseline_parent = conn.execute(
                "SELECT content_hash FROM candidates WHERE id=?",
                (parent_id, )).fetchone()
            if (baseline_parent is None
                    or baseline_parent["content_hash"] != content_hash):
                raise ValueError(
                    "baseline calibration candidate must match the unchanged "
                    "baseline source")
        conn.execute(
            """INSERT INTO candidates(
                   run_id, parent_candidate_id, source_hash, content_hash, source_text,
                   algorithm, engine, memory, dispatch, mechanism, descriptor_evidence_json,
                   correct, safe, simulation_score, hardware_score,
                   launch_options_json, environment_json, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id, source_hash) DO UPDATE SET
                   source_text=excluded.source_text,
                   content_hash=excluded.content_hash,
                   algorithm=excluded.algorithm,
                   engine=excluded.engine,
                   memory=excluded.memory,
                   dispatch=excluded.dispatch,
                   mechanism=excluded.mechanism,
                   descriptor_evidence_json=excluded.descriptor_evidence_json,
                   correct=MAX(candidates.correct, excluded.correct),
                   safe=MAX(candidates.safe, excluded.safe),
                   simulation_score=COALESCE(excluded.simulation_score, candidates.simulation_score),
                   hardware_score=COALESCE(excluded.hardware_score, candidates.hardware_score)""",
            (
                run_id,
                parent_id,
                source_hash,
                content_hash,
                source,
                algorithm,
                engine,
                memory,
                dispatch,
                mechanism,
                json.dumps(descriptor_evidence),
                int(correct),
                int(safe),
                simulation_score,
                hardware_score,
                json.dumps(launch_options or {}, sort_keys=True),
                json.dumps(environment or {}, sort_keys=True),
                now,
            ),
        )
        candidate = conn.execute(
            "SELECT * FROM candidates WHERE run_id=? AND source_hash=?",
            (run_id, source_hash),
        ).fetchone()
        candidate_id = int(candidate["id"])

        promote_simulation = (promotion_domain == "simulation"
                              and simulation_score is not None)
        promote_hardware = (promotion_domain == "hardware"
                            and hardware_score is not None)
        if correct and safe and (promote_simulation or promote_hardware):
            conn.execute(
                """INSERT INTO elite_cells(
                       run_id, algorithm, engine, memory, dispatch, mechanism,
                       simulation_candidate_id, simulation_score,
                       hardware_candidate_id, hardware_score, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id, algorithm, engine, memory, dispatch, mechanism)
                   DO NOTHING""",
                (
                    run_id,
                    algorithm,
                    engine,
                    memory,
                    dispatch,
                    mechanism,
                    candidate_id if promote_simulation else None,
                    simulation_score if promote_simulation else None,
                    candidate_id if promote_hardware else None,
                    hardware_score if promote_hardware else None,
                    now,
                ),
            )
            cell = conn.execute(
                """SELECT * FROM elite_cells WHERE run_id=? AND algorithm=? AND
                   engine=? AND memory=? AND dispatch=? AND mechanism=?""",
                (run_id, algorithm, engine, memory, dispatch, mechanism),
            ).fetchone()
            if promote_simulation:
                assert simulation_score is not None
                if (cell["simulation_score"] is None
                        or simulation_score > float(cell["simulation_score"])):
                    conn.execute(
                        """UPDATE elite_cells SET simulation_candidate_id=?,
                           simulation_score=?, updated_at=? WHERE run_id=? AND algorithm=?
                           AND engine=? AND memory=? AND dispatch=? AND mechanism=?""",
                        (candidate_id, simulation_score, now, run_id,
                         algorithm, engine, memory, dispatch, mechanism),
                    )
            if promote_hardware:
                assert hardware_score is not None
                if (cell["hardware_score"] is None
                        or hardware_score > float(cell["hardware_score"])):
                    conn.execute(
                        """UPDATE elite_cells SET hardware_candidate_id=?,
                           hardware_score=?, updated_at=? WHERE run_id=? AND algorithm=?
                           AND engine=? AND memory=? AND dispatch=? AND mechanism=?""",
                        (candidate_id, hardware_score, now, run_id, algorithm,
                         engine, memory, dispatch, mechanism),
                    )

        if attempt and complete_attempt:
            active_score = (hardware_score if promotion_domain == "hardware"
                            else simulation_score)
            if correct and safe and active_score is None:
                raise ValueError(
                    f"{promotion_domain} evidence is required to complete a valid "
                    "candidate attempt for this run")
            parent_score = None
            if parent_id:
                parent = conn.execute(
                    "SELECT simulation_score, hardware_score FROM candidates WHERE id=?",
                    (parent_id, ),
                ).fetchone()
                if promotion_domain == "hardware":
                    parent_score = parent["hardware_score"]
                else:
                    parent_score = parent["simulation_score"]
            child_score = active_score
            delta = (float(child_score) -
                     float(parent_score) if child_score is not None
                     and parent_score is not None else None)
            is_baseline = attempt["kind"] == "baseline"
            outcome = ("baseline_calibrated" if is_baseline and correct
                       and safe else "baseline_failed" if is_baseline else
                       "evaluated" if correct and safe else "invalid")
            conn.execute(
                """INSERT INTO transitions(run_id, attempt_id, parent_candidate_id,
                   child_candidate_id, delta_fitness, outcome, created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (run_id, attempt["id"], parent_id, candidate_id, delta,
                 outcome, now),
            )
            conn.execute(
                "UPDATE attempts SET status='completed', phase='completed', updated_at=? WHERE id=?",
                (now, attempt["id"]),
            )
            if is_baseline:
                if correct and safe and active_score is not None:
                    conn.execute(
                        """UPDATE search_runs SET revision=revision+1,
                           failure_reason='', updated_at=? WHERE id=?""",
                        (now, run_id),
                    )
                else:
                    conn.execute(
                        """UPDATE search_runs
                           SET status='failed_baseline_evaluation',
                               failure_reason='baseline correctness/evaluation failed',
                               revision=revision+1, updated_at=? WHERE id=?""",
                        (now, run_id),
                    )
            else:
                conn.execute(
                    """UPDATE search_runs SET completed_attempts=completed_attempts+1,
                       revision=revision+1, updated_at=? WHERE id=?""",
                    (now, run_id),
                )
        return candidate_id


def get_cell_elites(
        run_id: int, coordinate: tuple[int, int, int, int,
                                       str]) -> dict[str, Any]:
    ensure_schema()
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM elite_cells WHERE run_id=? AND algorithm=? AND
               engine=? AND memory=? AND dispatch=? AND mechanism=?""",
            (run_id, *coordinate),
        ).fetchone()
        return dict(row) if row else {}


def increment_attempt_counter(run_id: int, field: str) -> None:
    if field not in {"tool_calls_used", "llm_iterations_used"}:
        raise ValueError(f"unsupported attempt counter: {field}")
    with connect() as conn:
        conn.execute(
            f"UPDATE attempts SET {field}={field}+1, updated_at=? "
            "WHERE run_id=? AND status='active'",
            (utcnow(), run_id),
        )


def record_llm_iteration(run_id: int, api_request_id: str) -> None:
    """Count one logical model request while ignoring provider retries."""
    if not api_request_id:
        increment_attempt_counter(run_id, "llm_iterations_used")
        return
    with connect() as conn:
        conn.execute(
            """UPDATE attempts SET llm_iterations_used=llm_iterations_used+1,
               last_llm_request_id=?, updated_at=?
               WHERE run_id=? AND status='active' AND last_llm_request_id != ?""",
            (api_request_id, utcnow(), run_id, api_request_id),
        )


def update_attempt_phase(run_id: int, phase: str, error: str = "") -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE attempts SET phase=?, last_error=?, updated_at=?
               WHERE run_id=? AND status='active'""",
            (phase, error[:2000], utcnow(), run_id),
        )


def mark_parent_checked_out(run_id: int, source_hash: str) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE attempts SET phase='editing', current_source_hash=?,
               updated_at=? WHERE run_id=? AND status='active' AND phase='checkout'""",
            (source_hash, utcnow(), run_id),
        )


def update_current_source_hash(run_id: int, source_hash: str,
                               phase: str) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE attempts SET current_source_hash=?, phase=?, updated_at=?
               WHERE run_id=? AND status='active'""",
            (source_hash, phase, utcnow(), run_id),
        )


def abandon_active_attempt(run_id: int, reason: str) -> int | None:
    """Close a stuck attempt durably so the next hook can select another parent."""
    now = utcnow()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        attempt = conn.execute(
            "SELECT * FROM attempts WHERE run_id=? AND status='active'",
            (run_id, )).fetchone()
        if attempt is None:
            return None
        conn.execute(
            """UPDATE attempts SET status='abandoned', phase='abandoned',
               last_error=?, updated_at=? WHERE id=?""",
            (reason[:2000], now, attempt["id"]),
        )
        conn.execute(
            """INSERT INTO transitions(run_id, attempt_id, parent_candidate_id,
               child_candidate_id, delta_fitness, outcome, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (run_id, attempt["id"], attempt["parent_candidate_id"], None, None,
             "abandoned", now),
        )
        if attempt["kind"] == "baseline":
            conn.execute(
                """UPDATE search_runs SET status='failed_baseline_evaluation',
                   failure_reason=?, revision=revision+1, updated_at=? WHERE id=?""",
                (reason[:2000], now, run_id),
            )
        else:
            conn.execute(
                """UPDATE search_runs SET completed_attempts=completed_attempts+1,
                   revision=revision+1, updated_at=? WHERE id=?""",
                (now, run_id),
            )
        return int(attempt["id"])


def record_tool_event(
    *,
    run_id: int,
    session_id: str,
    tool_call_id: str,
    tool_name: str,
    result: dict[str, Any],
    source_content_hash: str | None = None,
) -> bool:
    if not tool_call_id:
        return True
    with connect() as conn:
        attempt = conn.execute(
            "SELECT id FROM attempts WHERE run_id=? AND status='active'",
            (run_id, )).fetchone()
        cursor = conn.execute(
            """INSERT OR IGNORE INTO tool_events(
                   run_id, attempt_id, session_id, tool_call_id, tool_name,
                   source_content_hash, result_json, created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (run_id, attempt["id"] if attempt else None, session_id,
             tool_call_id, tool_name, source_content_hash,
             json.dumps(result, default=str), utcnow()),
        )
        return cursor.rowcount == 1


def successful_tool_evidence(
    run_id: int,
    tool_names: set[str],
    *,
    source_content_hash: str,
    attempt_id: int | None,
) -> dict[str, Any] | None:
    """Return source-bound successful evaluator evidence for this attempt/candidate."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT attempt_id, tool_name, source_content_hash, result_json
               FROM tool_events WHERE run_id=? AND source_content_hash=?
               ORDER BY id DESC""",
            (run_id, source_content_hash),
        ).fetchall()
    for row in rows:
        if row["tool_name"] not in tool_names:
            continue
        if attempt_id is not None and row["attempt_id"] != attempt_id:
            continue
        try:
            result = json.loads(row["result_json"])
        except json.JSONDecodeError:
            continue
        if row["tool_name"] == "remote_verify":
            if (result.get("success") is True
                    and result.get("test_passed") is True
                    and result.get("bench_passed") is not False):
                return result
        else:
            rendered = json.dumps(result)
            trace_path = result.get("trace_local_path")
            trace_size = int(
                result.get("trace_json_size_bytes")
                or len(str(result.get("trace_json") or "")))
            if (result.get("success") is True and "[HOST] PASS" in rendered
                    and trace_path and trace_size > 1000):
                return result
    return None


def has_matching_evaluator_event(run_id: int, *, source_content_hash: str,
                                 attempt_id: int | None) -> bool:
    with connect() as conn:
        sql = """SELECT tool_name, result_json FROM tool_events WHERE run_id=? AND
                 source_content_hash=? AND tool_name IN
                 ('cannsim_local_run','cannsim_remote_run','remote_verify')"""
        params: tuple[Any, ...] = (run_id, source_content_hash)
        if attempt_id is not None:
            sql += " AND attempt_id=?"
            params += (attempt_id, )
        rows = conn.execute(sql + " ORDER BY id DESC", params).fetchall()
    for row in rows:
        try:
            result = json.loads(row["result_json"])
        except json.JSONDecodeError:
            continue
        if result.get("success") is False:
            return True
        if (row["tool_name"] == "remote_verify"
                and (result.get("test_passed") is False
                     or result.get("bench_passed") is False)):
            return True
    return False


def finalize_run(run_id: int) -> dict[str, Any]:
    ensure_schema()
    with connect() as conn:
        run = conn.execute("SELECT * FROM search_runs WHERE id=?",
                           (run_id, )).fetchone()
        if run is None:
            raise KeyError(f"unknown guided-search run: {run_id}")
        if str(run["status"]).startswith("failed_"):
            raise RuntimeError(f"guided-search run failed ({run['status']}): "
                               f"{run['failure_reason'] or 'no valid elite'}")
        active = conn.execute(
            "SELECT 1 FROM attempts WHERE run_id=? AND status='active'",
            (run_id, )).fetchone()
        if active is not None:
            raise RuntimeError(
                "cannot finalize while a guided-search attempt is active")
        if (run["status"] != "completed"
                and int(run["completed_attempts"]) < int(run["budget"])):
            raise RuntimeError(
                "cannot finalize before the mutation budget is exhausted")
        hardware_domain = run["promotion_domain"] == "hardware"
        score_column = "hardware_score" if hardware_domain else "simulation_score"
        candidate_column = ("hardware_candidate_id"
                            if hardware_domain else "simulation_candidate_id")
        candidate = conn.execute(
            f"""SELECT c.* FROM elite_cells e
                 JOIN candidates c ON c.id=e.{candidate_column}
                 WHERE e.run_id=? AND c.correct=1 AND c.safe=1
                 AND e.{score_column} IS NOT NULL
                 ORDER BY e.{score_column} DESC LIMIT 1""",
            (run_id, ),
        ).fetchone()
        if candidate is None:
            raise RuntimeError("no hardware-confirmed valid elite is available"
                               if hardware_domain else
                               "no cannsim-confirmed valid elite is available")
        workspace = Path(run["workspace"])
        pipeline = _load_pipeline(workspace)
        baseline = pipeline.get("baseline") or "kernel.py"
        output = workspace / f"opt_{baseline}"
        output.write_text(candidate["source_text"], encoding="utf-8")
        winner_hash = hashlib.sha256(
            candidate["source_text"].encode("utf-8")).hexdigest()
        now = utcnow()
        conn.execute(
            """UPDATE search_runs SET status='completed', winner_candidate_id=?,
               winner_source_hash=?, revision=revision+1, updated_at=? WHERE id=?""",
            (candidate["id"], winner_hash, now, run_id),
        )
        guided = pipeline.setdefault("guided_search", {})
        guided.update({
            "enabled": True,
            "run_id": run_id,
            "status": "completed",
            "winner_candidate_id": int(candidate["id"]),
            "winner_source_hash": winner_hash,
            "revision": int(guided.get("revision", 0)) + 1,
        })
        _save_pipeline(workspace, pipeline)
        return {
            "run_id": run_id,
            "candidate_id": int(candidate["id"]),
            "winner_source_hash": winner_hash,
            "path": str(output),
        }
