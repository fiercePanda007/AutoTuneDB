from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent

con = duckdb.connect(
    database=":memory:",
    config={"allow_unsigned_extensions": "true"},
)

duckdb_version, _ = con.execute("PRAGMA version").fetchone()
duckdb_platform = con.execute("PRAGMA platform").fetchone()[0]

repository_dir = (
    PROJECT_ROOT
    / "intelligent-duck"
    / "build"
    / "release"
    / "repository"
    / duckdb_version
    / duckdb_platform
)

candidate_paths = [
    repository_dir / "autoDB.duckdb_extension",
    repository_dir / "autodb.duckdb_extension",
    repository_dir / "alex.duckdb_extension",
    (
        PROJECT_ROOT
        / "intelligent-duck"
        / "build"
        / "release"
        / "extension"
        / "autodb"
        / "autoDB.duckdb_extension"
    ),
    (
        PROJECT_ROOT
        / "intelligent-duck"
        / "build"
        / "release"
        / "extension"
        / "alex"
        / "alex.duckdb_extension"
    ),
]

extension_path = next((path for path in candidate_paths if path.exists()), None)
if extension_path is None:
    raise FileNotFoundError(
        "Matching extension binary not found.\n"
        f"DuckDB version/platform: {duckdb_version} / {duckdb_platform}\n"
        f"Tried paths:\n  - "
        + "\n  - ".join(str(path) for path in candidate_paths)
        + "\n"
        "Rebuild with:\n"
        f"  cd {PROJECT_ROOT / 'intelligent-duck'}\n"
        f"  make clean && DUCKDB_PLATFORM={duckdb_platform} make release\n"
    )

try:
    con.execute(f"LOAD '{extension_path.as_posix()}';")
except duckdb.InvalidInputException as exc:
    message = str(exc)
    if "does not match DuckDB loading it" in message:
        raise RuntimeError(
            "Extension ABI/platform mismatch detected.\n"
            "Rebuild with the exact runtime platform:\n"
            f"  cd {PROJECT_ROOT / 'intelligent-duck'}\n"
            f"  make clean && DUCKDB_PLATFORM={duckdb_platform} make release\n"
            "Then run this script again."
        ) from exc
    raise RuntimeError(
        "Extension failed to initialize.\n"
        f"Original error: {message}"
    ) from exc

print(f"Loaded successfully from: {extension_path}")
