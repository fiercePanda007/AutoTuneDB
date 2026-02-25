"""
Database manager for the Self-Improving Database Query Optimizer (DuckDB backend).

This module handles DuckDB connections, query execution, and metrics collection.
It keeps the same public API shape as the legacy backend to avoid call-site churn.
"""

import time
import psutil
import duckdb
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from pathlib import Path


class DatabaseManager:
    """Manages DuckDB connections and operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # DuckDB config lives under 'duckdb'. Fallbacks keep it robust.
        self.duck_config = config.get("duckdb", {})
        self.logger = logging.getLogger(__name__)

        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.extension_loaded = False

    def _extension_candidates(self, configured_path: Path) -> list[Path]:
        """Build candidate extension paths in priority order."""
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            configured_path,
            project_root / "intelligent-duck" / "build" / "release" / "extension" / "autodb" / "autoDB.duckdb_extension",
            project_root / "intelligent-duck" / "build" / "release" / "extension" / "autodb" / "autodb.duckdb_extension",
            project_root / "intelligent-duck" / "build" / "release" / "extension" / "alex" / "alex.duckdb_extension",
        ]

        try:
            duckdb_version, _ = self.conn.execute("PRAGMA version").fetchone()
            duckdb_platform = self.conn.execute("PRAGMA platform").fetchone()[0]
            repository_dir = (
                project_root
                / "intelligent-duck"
                / "build"
                / "release"
                / "repository"
                / duckdb_version
                / duckdb_platform
            )
            candidates.extend([
                repository_dir / "autoDB.duckdb_extension",
                repository_dir / "autodb.duckdb_extension",
                repository_dir / "alex.duckdb_extension",
            ])
        except Exception:
            pass

        seen: set[str] = set()
        unique: list[Path] = []
        for p in candidates:
            key = str(p)
            if key not in seen:
                unique.append(p)
                seen.add(key)
        return unique

    def _load_optional_extension(self):
        """Load configured extension if available; skip safely otherwise."""
        ext_path = (self.duck_config.get("extension_path") or "").strip()
        if not ext_path:
            return

        require_extension = bool(self.duck_config.get("require_extension", False))
        ext_file = Path(ext_path).expanduser()
        if not ext_file.is_absolute():
            ext_file = (Path.cwd() / ext_file).resolve()

        candidates = self._extension_candidates(ext_file)
        resolved_path = next((candidate for candidate in candidates if candidate.exists()), None)

        if not resolved_path:
            msg = (
                "DuckDB extension not found.\n"
                f"Configured path: {ext_file}\n"
                "Checked candidates:\n  - " + "\n  - ".join(str(path) for path in candidates)
            )
            if require_extension:
                raise FileNotFoundError(msg)
            self.logger.warning(f"{msg}\nContinuing without extension.")
            return

        if resolved_path != ext_file:
            self.logger.warning(f"Configured extension not found at {ext_file}")
            self.logger.warning(f"Using discovered extension path: {resolved_path}")

        try:
            escaped_path = resolved_path.as_posix().replace("'", "''")
            self.conn.execute(f"LOAD '{escaped_path}';")
            self.extension_loaded = True
            self.logger.info(f"Loaded DuckDB extension: {resolved_path}")
        except Exception as e:
            if require_extension:
                raise
            self.extension_loaded = False
            self.logger.warning(f"Could not load extension '{resolved_path}': {e}")
            self.logger.warning("Continuing without extension.")

    def _default_autodb_index_specs(self) -> list[Dict[str, str]]:
        """
        Default autoDB index bootstrap candidates.

        Each target column is followed by a numeric payload column in the schema.
        """
        return [
            {"table": "orders", "column": "order_id"},
            {"table": "products", "column": "supplier_id"},
            {"table": "transactions", "column": "transaction_id"},
            {"table": "transactions", "column": "product_id"},
        ]

    def _bootstrap_autodb_indexes(self):
        """Best-effort autoDB index bootstrap for extension-backed lookups."""
        if not self.extension_loaded:
            return
        if not self.duck_config.get("autodb_bootstrap_enabled", True):
            return

        specs = self.duck_config.get("autodb_bootstrap_indexes")
        if not isinstance(specs, list) or not specs:
            specs = self._default_autodb_index_specs()

        for spec in specs:
            if not isinstance(spec, dict):
                continue

            table_name = (spec.get("table") or "").strip()
            column_name = (spec.get("column") or "").strip()
            if not table_name or not column_name:
                continue

            table_escaped = table_name.replace("'", "''")
            column_escaped = column_name.replace("'", "''")

            try:
                self.conn.execute(
                    f"PRAGMA create_autoDB_index('{table_escaped}', '{column_escaped}');"
                )
                self.logger.info(
                    f"autoDB index bootstrapped for {table_name}.{column_name}"
                )
            except Exception as e:
                # Continue even if a specific candidate is not compatible.
                self.logger.warning(
                    f"autoDB bootstrap skipped for {table_name}.{column_name}: {e}"
                )

    def connect(self):
        """Establish DuckDB connection."""
        try:
            db_path = self.duck_config.get("path", ":memory:")
            read_only = bool(self.duck_config.get("read_only", False))
            self.extension_loaded = False

            # Ensure folder exists if using a file path
            if db_path not in (":memory:", ":memory:") and "/" in db_path:
                os.makedirs(os.path.dirname(db_path), exist_ok=True)

            connect_config: Dict[str, str] = {}
            if self.duck_config.get("allow_unsigned_extensions", True):
                connect_config["allow_unsigned_extensions"] = "true"

            if connect_config:
                self.conn = duckdb.connect(database=db_path, read_only=read_only, config=connect_config)
            else:
                self.conn = duckdb.connect(database=db_path, read_only=read_only)

            # Optional tuning knobs
            threads = self.duck_config.get("threads")
            if threads:
                try:
                    self.conn.execute(f"PRAGMA threads={int(threads)};")
                except Exception as e:
                    self.logger.debug(f"Could not set PRAGMA threads: {e}")

            # Optional: load intelligent-duck extension only when available.
            self._load_optional_extension()
            self._bootstrap_autodb_indexes()

            ver = self.get_version()
            self.logger.info(f"Connected to DuckDB: {ver}")

        except Exception as e:
            self.logger.error(f"Failed to connect to DuckDB: {e}")
            raise

    def get_version(self) -> str:
        """Get DuckDB version."""
        if not self.conn:
            raise Exception("Database not connected")
        return self.conn.execute("SELECT version();").fetchone()[0]

    # ---- Compatibility with legacy call sites ----
    def get_connection(self):
        """
        Kept for API compatibility. DuckDB Python connection is single-handle.
        """
        if not self.conn:
            raise Exception("Database not connected")
        return self.conn

    def return_connection(self, conn):
        """No-op for DuckDB (kept for API compatibility)."""
        return

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True
    ) -> Tuple[Optional[List], float, Dict[str, Any]]:
        """
        Execute a query and collect metrics.

        Returns:
            (results, execution_time_seconds, metrics_dict)
        """
        if not self.conn:
            raise Exception("Database not connected")

        start = time.perf_counter()
        results = None

        try:
            if params:
                cur = self.conn.execute(query, params)
            else:
                cur = self.conn.execute(query)

            if fetch:
                # fetchall works even for SELECT; for DDL/DML it returns []
                try:
                    results = cur.fetchall()
                except Exception:
                    results = None

            exec_time = time.perf_counter() - start

            metrics = self._collect_query_metrics(results)
            return results, exec_time, metrics

        except Exception as e:
            self.logger.error(f"Query execution error: {e}")
            raise

    def _collect_query_metrics(self, results: Optional[List]) -> Dict[str, Any]:
        """
        DuckDB does not expose backend cache hit stats the same way.
        We return: cpu, memory, rows, and cost if available from EXPLAIN.
        """
        metrics = {
            "cpu": 0.0,
            "memory": 0.0,
            "cache_hit_rate": 0.0,  # not available in DuckDB
            "rows": 0,
            "cost": 0.0
        }

        # rows
        if results is not None:
            metrics["rows"] = len(results)

        # Process CPU/memory (best-effort).
        try:
            process = psutil.Process()
            metrics["cpu"] = process.cpu_percent(interval=0.01)
            metrics["memory"] = process.memory_percent()
        except Exception as e:
            self.logger.debug(f"Could not collect system metrics: {e}")

        return metrics

    def get_query_plan(self, query: str) -> Dict[str, Any]:
        """
        Get query execution plan.

        DuckDB returns textual or JSON plans depending on explain options.
        We'll use EXPLAIN and store it as text for now.
        """
        if not self.conn:
            return {"plan": {}, "cost": 0, "rows": 0}

        try:
            # DuckDB returns rows like ("physical_plan", "…text…") depending on version.
            plan_rows = self.conn.execute(f"EXPLAIN {query}").fetchall()
            plan_text = "\n".join([str(r[-1]) for r in plan_rows])

            return {
                "plan": {"text": plan_text},
                "cost": 0,   # DuckDB doesn't expose PG Total Cost directly
                "rows": 0
            }
        except Exception as e:
            self.logger.warning(f"Failed to get query plan: {e}")
            return {"plan": {}, "cost": 0, "rows": 0}

    # ---- Methods kept for compatibility with legacy interfaces ----

    def get_cache_hit_rate(self) -> float:
        """Not available in DuckDB; return 0.0 for compatibility."""
        return 0.0

    def get_connection_count(self) -> int:
        """DuckDB embedded connection count not tracked; return 1 if connected."""
        return 1 if self.conn else 0

    def get_table_sizes(self) -> Dict[str, int]:
        """
        DuckDB table-size statistics are not exposed in a single stable API.
        Return an empty map for compatibility.
        """
        if not self.conn:
            return {}
        try:
            # This pragma exists in many DuckDB builds
            rows = self.conn.execute("PRAGMA storage_info;").fetchall()
            # storage_info is per table/column/segment; summarizing accurately is non-trivial.
            # Phase-1: return empty to avoid misleading numbers.
            return {}
        except Exception:
            return {}

    def get_index_usage(self) -> Dict[str, float]:
        """DuckDB index usage stats not like Postgres; return empty dict."""
        return {}

    def get_database_stats(self) -> Dict[str, Any]:
        return {
            "cache_hit_rate": self.get_cache_hit_rate(),
            "connection_count": self.get_connection_count(),
            "table_sizes": self.get_table_sizes(),
            "index_usage": self.get_index_usage()
        }

    def execute_explain(self, query: str) -> str:
        """Execute EXPLAIN on a query (DuckDB)."""
        if not self.conn:
            return ""
        try:
            rows = self.conn.execute(f"EXPLAIN {query}").fetchall()
            return "\n".join([str(r[-1]) for r in rows])
        except Exception as e:
            self.logger.warning(f"Failed to explain query: {e}")
            return ""

    def execute_analyze(self, query: str) -> str:
        """
        DuckDB doesn't have EXPLAIN ANALYZE exactly like Postgres.
        Use EXPLAIN ANALYZE if supported, else fallback to EXPLAIN.
        """
        if not self.conn:
            return ""
        try:
            rows = self.conn.execute(f"EXPLAIN ANALYZE {query}").fetchall()
            return "\n".join([str(r[-1]) for r in rows])
        except Exception:
            # fallback
            return self.execute_explain(query)

    def reset_statistics(self):
        """No-op for DuckDB in Phase-1."""
        self.logger.info("reset_statistics: no-op for DuckDB Phase-1")

    def disconnect(self):
        """Close DuckDB connection."""
        if self.conn:
            try:
                self.conn.close()
            finally:
                self.conn = None
            self.logger.info("DuckDB connection closed")
