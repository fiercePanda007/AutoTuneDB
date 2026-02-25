#!/usr/bin/env python3
"""
Benchmark a modest DuckDB dataset with three lookup modes:
1) Table scan (no index)
2) DuckDB CREATE INDEX (ART index)
3) autoDB extension index + autoDB_find
"""

import argparse
import ctypes
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@contextmanager
def suppress_stdout(enabled: bool = True):
    """
    Suppress stdout at file-descriptor level so C/C++ extension prints are muted.
    """
    if not enabled:
        yield
        return

    libc = None
    try:
        libc = ctypes.CDLL(None)
    except Exception:
        libc = None

    sys.stdout.flush()
    if libc is not None:
        try:
            libc.fflush(None)
        except Exception:
            pass

    saved_fd = os.dup(1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            yield
    finally:
        if libc is not None:
            try:
                libc.fflush(None)
            except Exception:
                pass
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)

    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = pos - lower
    return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def summarize_latency(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

    avg = sum(values_ms) / len(values_ms)
    return {
        "avg": avg,
        "p50": percentile(values_ms, 50),
        "p95": percentile(values_ms, 95),
        "p99": percentile(values_ms, 99),
        "min": min(values_ms),
        "max": max(values_ms),
    }


def connect_duckdb(db_path: Path, duck_cfg: Dict[str, Any]) -> duckdb.DuckDBPyConnection:
    connect_config: Dict[str, str] = {}
    if duck_cfg.get("allow_unsigned_extensions", True):
        connect_config["allow_unsigned_extensions"] = "true"

    if connect_config:
        conn = duckdb.connect(str(db_path), config=connect_config)
    else:
        conn = duckdb.connect(str(db_path))

    threads = duck_cfg.get("threads")
    if threads:
        try:
            conn.execute(f"PRAGMA threads={int(threads)};")
        except Exception:
            pass

    return conn


def extension_candidates(conn: duckdb.DuckDBPyConnection, configured_path: Path) -> List[Path]:
    project_root = Path(__file__).resolve().parent
    candidates = [
        configured_path,
        project_root / "intelligent-duck" / "build" / "release" / "extension" / "autodb" / "autoDB.duckdb_extension",
        project_root / "intelligent-duck" / "build" / "release" / "extension" / "autodb" / "autodb.duckdb_extension",
        project_root / "intelligent-duck" / "build" / "release" / "extension" / "alex" / "alex.duckdb_extension",
    ]

    try:
        duckdb_version, _ = conn.execute("PRAGMA version").fetchone()
        duckdb_platform = conn.execute("PRAGMA platform").fetchone()[0]
        repository_dir = (
            project_root
            / "intelligent-duck"
            / "build"
            / "release"
            / "repository"
            / duckdb_version
            / duckdb_platform
        )
        candidates.extend(
            [
                repository_dir / "autoDB.duckdb_extension",
                repository_dir / "autodb.duckdb_extension",
                repository_dir / "alex.duckdb_extension",
            ]
        )
    except Exception:
        pass

    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def load_autodb_extension(
    conn: duckdb.DuckDBPyConnection,
    config_path: Path,
    duck_cfg: Dict[str, Any],
) -> str:
    ext_path_raw = (duck_cfg.get("extension_path") or "").strip()
    if not ext_path_raw:
        raise RuntimeError("duckdb.extension_path is required for autoDB benchmark")

    configured = Path(ext_path_raw).expanduser()
    if not configured.is_absolute():
        configured = (config_path.parent / configured).resolve()

    candidates = extension_candidates(conn, configured)
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    if not resolved:
        checked = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"autoDB extension not found.\nConfigured: {configured}\nChecked:\n  - {checked}"
        )

    escaped = resolved.as_posix().replace("'", "''")
    conn.execute(f"LOAD '{escaped}';")
    return str(resolved)


def create_modest_dataset(
    conn: duckdb.DuckDBPyConnection,
    row_count: int,
    seed: int,
    batch_size: int,
) -> None:
    conn.execute("DROP TABLE IF EXISTS modest_lookup;")
    conn.execute(
        """
        CREATE TABLE modest_lookup (
            key_id BIGINT,
            payload BIGINT,
            bucket INTEGER
        );
        """
    )

    rng = random.Random(seed)
    for start in range(1, row_count + 1, batch_size):
        end = min(start + batch_size - 1, row_count)
        rows = []
        for key_id in range(start, end + 1):
            payload = rng.randint(1, 10_000_000)
            rows.append((key_id, payload, payload % 128))
        conn.executemany(
            "INSERT INTO modest_lookup (key_id, payload, bucket) VALUES (?, ?, ?);",
            rows,
        )

    try:
        conn.execute("ANALYZE modest_lookup;")
    except Exception:
        pass


def run_select_benchmark(
    conn: duckdb.DuckDBPyConnection,
    lookup_keys: List[int],
    warmup_keys: List[int],
) -> List[float]:
    sql = "SELECT payload FROM modest_lookup WHERE key_id = ?;"
    for key in warmup_keys:
        conn.execute(sql, [key]).fetchone()

    latencies_ms: List[float] = []
    for key in lookup_keys:
        start = time.perf_counter()
        conn.execute(sql, [key]).fetchone()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms


def run_autodb_benchmark(
    conn: duckdb.DuckDBPyConnection,
    lookup_keys: List[int],
    warmup_keys: List[int],
    suppress_autodb_output: bool,
) -> List[float]:
    with suppress_stdout(enabled=suppress_autodb_output):
        for key in warmup_keys:
            conn.execute(f"PRAGMA autoDB_find('bigint', '{key}');").fetchall()

        latencies_ms: List[float] = []
        for key in lookup_keys:
            start = time.perf_counter()
            conn.execute(f"PRAGMA autoDB_find('bigint', '{key}');").fetchall()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms


def build_duckdb_index(conn: duckdb.DuckDBPyConnection) -> float:
    conn.execute("DROP INDEX IF EXISTS idx_modest_lookup_key;")
    start = time.perf_counter()
    conn.execute("CREATE INDEX idx_modest_lookup_key ON modest_lookup(key_id);")
    return (time.perf_counter() - start) * 1000.0


def build_autodb_index(conn: duckdb.DuckDBPyConnection, suppress_autodb_output: bool) -> float:
    start = time.perf_counter()
    with suppress_stdout(enabled=suppress_autodb_output):
        conn.execute("PRAGMA create_autoDB_index('modest_lookup', 'key_id');")
    return (time.perf_counter() - start) * 1000.0


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    print("\nLatency Results (ms)")
    print("mode                 build_ms    avg      p50      p95      p99      min      max")
    print("-" * 84)
    for mode in ("table_scan", "duckdb_index_art", "autodb_index"):
        if mode not in results:
            continue
        stats = results[mode]
        print(
            f"{mode:<20} {stats['build_ms']:>8.2f} "
            f"{stats['avg']:>8.3f} {stats['p50']:>8.3f} {stats['p95']:>8.3f} "
            f"{stats['p99']:>8.3f} {stats['min']:>8.3f} {stats['max']:>8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a modest DuckDB dataset and compare table-scan vs ART index vs autoDB."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--db-path",
        default="data/modest_autodb_benchmark.duckdb",
        help="Benchmark DuckDB file path",
    )
    parser.add_argument("--rows", type=int, default=200_000, help="Row count for modest table")
    parser.add_argument("--queries", type=int, default=5_000, help="Number of lookup queries")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup lookups per mode")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Insert batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-autodb",
        action="store_true",
        help="Skip autoDB mode (run only table scan + DuckDB index)",
    )
    parser.add_argument(
        "--show-autodb-output",
        action="store_true",
        help="Show verbose stdout emitted by autoDB extension",
    )
    parser.add_argument(
        "--keep-existing-db",
        action="store_true",
        help="Reuse existing benchmark DB file instead of recreating it",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    duck_cfg = config.get("duckdb", {})

    db_path = Path(args.db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not args.keep_existing_db:
        db_path.unlink()

    conn = connect_duckdb(db_path, duck_cfg)
    try:
        print(f"Benchmark DB: {db_path}")
        print(f"Rows: {args.rows:,} | Queries: {args.queries:,} | Warmup: {args.warmup:,}")
        print("Note: DuckDB CREATE INDEX uses ART (Adaptive Radix Tree), not B+ tree.")

        create_modest_dataset(conn, args.rows, args.seed, args.batch_size)

        extension_path = ""
        if not args.skip_autodb:
            extension_path = load_autodb_extension(conn, config_path, duck_cfg)
            print(f"Loaded autoDB extension: {extension_path}")

        key_rng = random.Random(args.seed + 99)
        lookup_keys = [key_rng.randint(1, args.rows) for _ in range(args.queries)]
        warmup_keys = lookup_keys[: min(args.warmup, len(lookup_keys))]

        results: Dict[str, Dict[str, float]] = {}

        conn.execute("DROP INDEX IF EXISTS idx_modest_lookup_key;")
        scan_lat_ms = run_select_benchmark(conn, lookup_keys, warmup_keys)
        scan_stats = summarize_latency(scan_lat_ms)
        scan_stats["build_ms"] = 0.0
        results["table_scan"] = scan_stats

        duck_index_build_ms = build_duckdb_index(conn)
        duck_lat_ms = run_select_benchmark(conn, lookup_keys, warmup_keys)
        duck_stats = summarize_latency(duck_lat_ms)
        duck_stats["build_ms"] = duck_index_build_ms
        results["duckdb_index_art"] = duck_stats

        if not args.skip_autodb:
            suppress_autodb_output = not args.show_autodb_output
            autodb_build_ms = build_autodb_index(conn, suppress_autodb_output)
            autodb_lat_ms = run_autodb_benchmark(
                conn,
                lookup_keys,
                warmup_keys,
                suppress_autodb_output=suppress_autodb_output,
            )
            autodb_stats = summarize_latency(autodb_lat_ms)
            autodb_stats["build_ms"] = autodb_build_ms
            results["autodb_index"] = autodb_stats

        print_results(results)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
