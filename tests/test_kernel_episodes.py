"""Tests for kernel-episodes plugin.

Tests the episode CRUD operations and full-text search
without needing a running Hermes agent.
"""
import os
import sys
import importlib.util
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_DIR))

# Set required env for kernel-episodes plugin (use temp DB per test via monkeypatch)
os.environ.setdefault("KERNEL_EPISODES_DB",
                      str(_PROJECT_DIR / "tests" / "test_episodes.db"))

# Load kernel-episodes plugin directly (same pattern as Hermes)
_spec = importlib.util.spec_from_file_location(
    "kernel_episodes",
    str(_PROJECT_DIR / ".hermes" / "plugins" / "kernel-episodes" /
        "__init__.py"),
)
_kernel_episodes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kernel_episodes)

_write_episode = _kernel_episodes._write_episode
_retrieve_episodes = _kernel_episodes._retrieve_episodes
_list_episodes = _kernel_episodes._list_episodes
_update_episode = _kernel_episodes._update_episode
_delete_episode = _kernel_episodes._delete_episode
_restore_database_from_dump = _kernel_episodes._restore_database_from_dump
_sanitize_fts_query = _kernel_episodes._sanitize_fts_query
_row_to_dict = _kernel_episodes._row_to_dict
_db_path = _kernel_episodes._db_path


class TestSanitizeFtsQuery:
    """Test FTS5 query sanitization."""

    def test_simple_query(self):
        assert _sanitize_fts_query("softmax scalar") == '"softmax" OR "scalar"'

    def test_hyphenated_query(self):
        # Hyphens cause FTS5 "no such column" errors
        result = _sanitize_fts_query("element-wise softmax")
        assert "-" not in result
        assert '"element"' in result or '"wise"' in result

    def test_empty_query(self):
        assert _sanitize_fts_query("") == '""'

    def test_special_chars(self):
        result = _sanitize_fts_query("tl.dot fp16/fp32")
        # Should strip special FTS chars
        assert ":" not in result or "tl.dot" not in result


class TestEpisodeCRUD:
    """Test create, read, update, delete operations."""

    def _init_db(self, db_path):
        """Create the episodes database schema."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kernel_name TEXT NOT NULL,
                target TEXT NOT NULL,
                observation TEXT NOT NULL,
                thoughts TEXT NOT NULL,
                action TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                kernel_name, target, observation, thoughts, action, result,
                content='episodes',
                content_rowid='id'
            )
        """)
        # Create triggers to keep FTS index in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, kernel_name, target, observation, thoughts, action, result)
                VALUES (new.id, new.kernel_name, new.target, new.observation, new.thoughts, new.action, new.result);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, kernel_name, target, observation, thoughts, action, result)
                VALUES ('delete', old.id, old.kernel_name, old.target, old.observation, old.thoughts, old.action, old.result);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, kernel_name, target, observation, thoughts, action, result)
                VALUES ('delete', old.id, old.kernel_name, old.target, old.observation, old.thoughts, old.action, old.result);
                INSERT INTO episodes_fts(rowid, kernel_name, target, observation, thoughts, action, result)
                VALUES (new.id, new.kernel_name, new.target, new.observation, new.thoughts, new.action, new.result);
            END
        """)
        conn.commit()
        conn.close()

    def test_write_and_retrieve(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "episodes.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", db_path)
        self._init_db(db_path)

        ep = _write_episode(
            kernel_name="softmax",
            target="ascend950",
            observation="Baseline: aiv_scalar=85%",
            thoughts="Two-pass detected",
            action="Single-pass tl.sum(x,1)",
            result="3605us -> 752us",
        )
        assert ep["id"] == 1
        assert ep["kernel_name"] == "softmax"

    def test_retrieve_by_query(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "episodes.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", db_path)
        self._init_db(db_path)

        _write_episode("matmul", "ascend950", "cube low", "", "dot_pad", "2x")
        _write_episode("softmax", "ascend950", "scalar high", "",
                       "single_pass", "4x")

        results = _retrieve_episodes(query="softmax scalar",
                                     target="ascend950")
        assert len(results) >= 1
        assert any(r["kernel_name"] == "softmax" for r in results)

    def test_list_episodes(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "episodes.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", db_path)
        self._init_db(db_path)

        _write_episode("relu", "ascend950", "test", "", "action", "result")
        eps = _list_episodes()
        assert len(eps) >= 1

    def test_list_by_kernel_name(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "episodes.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", db_path)
        self._init_db(db_path)

        _write_episode("gelu", "ascend950", "obs", "thought", "act", "res")
        gelu_eps = _list_episodes(kernel_name="gelu")
        assert len(gelu_eps) >= 1
        assert all(e["kernel_name"] == "gelu" for e in gelu_eps)

    def test_update_episode(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "episodes.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", db_path)
        self._init_db(db_path)

        ep = _write_episode("relu", "ascend950", "obs", "thought", "act",
                            "res")
        ep_id = ep["id"]

        _update_episode(ep_id, result="updated result")
        results = _retrieve_episodes(query="relu")
        assert any(r["id"] == ep_id and r["result"] == "updated result"
                   for r in results)

    def test_delete_episode(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "episodes.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", db_path)
        self._init_db(db_path)

        ep = _write_episode("test_del", "ascend950", "obs", "thought", "act",
                            "res")
        ep_id = ep["id"]

        deleted = _delete_episode(ep_id)
        assert deleted["id"] == ep_id

        results = _retrieve_episodes(query="test_del")
        assert len(results) == 0

    def test_restore_database_from_dump(self, tmp_path, monkeypatch):
        dump_path = str(_PROJECT_DIR / "episodes.sql")
        db_path = str(tmp_path / "restored_episodes.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", db_path)

        restored = _restore_database_from_dump(dump_path=dump_path)

        assert restored["db_path"] == db_path
        assert restored["episode_count"] >= 1
        eps = _list_episodes(limit=1)
        assert len(eps) == 1
        assert "kernel_name" in eps[0]

    def test_restore_database_from_dump_no_overwrite(self, tmp_path):
        dump_path = str(_PROJECT_DIR / "episodes.sql")
        db_path = tmp_path / "existing.db"
        db_path.write_bytes(b"existing")

        import pytest
        with pytest.raises(FileExistsError):
            _restore_database_from_dump(dump_path=dump_path,
                                        db_path=str(db_path),
                                        overwrite=False)

    def test_restore_on_boot(self, tmp_path, monkeypatch):
        db_path = tmp_path / "boot_restored_episodes.db"
        dump_path = db_path.with_suffix(".sql")
        dump_path.write_text(
            (_PROJECT_DIR / "episodes.sql").read_text(encoding="utf-8"),
            encoding="utf-8")
        monkeypatch.setenv("KERNEL_EPISODES_DB", str(db_path))

        _kernel_episodes._restore_on_boot()

        eps = _list_episodes(limit=1)
        assert len(eps) == 1
        assert "kernel_name" in eps[0]

    def test_row_to_dict(self):
        """Test sqlite3.Row to dict conversion."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (a INT, b TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")
        row = conn.execute("SELECT * FROM t").fetchone()
        conn.close()

        d = _row_to_dict(row)
        assert d == {"a": 1, "b": "hello"}


class TestDbPath:
    """Test database path resolution."""

    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("KERNEL_EPISODES_DB", raising=False)
        path = _db_path()
        assert path.endswith("episodes.db")

    def test_custom_path(self, monkeypatch, tmp_path):
        custom = str(tmp_path / "custom.db")
        monkeypatch.setenv("KERNEL_EPISODES_DB", custom)
        assert _db_path() == custom
