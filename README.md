# Self-Improving Database Query Optimizer (DuckDB + intelligent-duck autoDB)

This repository contains:

- `intelligent-duck/`: the C++ DuckDB extension source (`autoDB.duckdb_extension`)
- the Python learning system (`main.py`, `run_demo_duckdb.py`) that loads and uses that extension

Recommended order:

1. Build the `intelligent-duck` extension
2. Set up Python environment in this main project
3. Verify extension load
4. Initialize data and telemetry
5. Run the full system

## Prerequisites

- Python 3.10+
- `make`, `cmake`, C++ compiler (`g++` or `clang++`)
- Git submodules available (for fresh clones)

## 1) Build `intelligent-duck` Extension First

From project root:

```bash
cd intelligent-duck
git submodule update --init --recursive
make release
cd ..
```

Expected binary:

```text
intelligent-duck/build/release/extension/autodb/autoDB.duckdb_extension
```

Note: `config.yaml` already points to this path by default.

## 2) Set Up Python Environment (Main Project)

From project root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3) Verify Extension Import/Load

Validate that the extension binary matches your DuckDB Python runtime:

```bash
python duck_connect_check.py
```

Expected result: `Loaded successfully from: ...autoDB.duckdb_extension`

If you see an ABI/platform mismatch, rebuild with the exact platform:

```bash
python - <<'PY'
import duckdb
con = duckdb.connect()
print(con.execute("PRAGMA platform").fetchone()[0])
PY

cd intelligent-duck
make clean && DUCKDB_PLATFORM=<platform_from_above> make release
cd ..
python duck_connect_check.py
```

## 4) Initialize Database and Telemetry

From project root with venv active:

```bash
python setup_duckdb.py --scale 0.1
```

- Use `--scale 0.05` or `0.1` for faster local testing
- Use `--scale 1.0` for full-sized demo data

## 5) Run the Whole System

Quick demo run (recommended):

```bash
python run_demo_duckdb.py --duration 0.1 --fast-mode
```

Direct orchestrator run (alternative):

```bash
python main.py --duration 0.1 --fast-mode
```

Optional dashboard in another terminal:

```bash
source venv/bin/activate
python dashboard.py
```

Open: `http://127.0.0.1:5000`

## All autoDB Pragmas (with examples)

The extension currently registers these pragma calls:

- `PRAGMA create_autoDB_index('<table>', '<key_column>')`
- `PRAGMA load_benchmark('<table>', '<benchmark_name>', <benchmark_size>, <num_batches_insert>)`
- `PRAGMA run_lookup_benchmark('<benchmark_name>', '<index_name>')`
- `PRAGMA create_art_index('<table>', '<key_column>')`
- `PRAGMA insert_into_table('<table>', '<type>', '<key>', '<payload>')`
- `PRAGMA run_benchmark_one_batch('<benchmark_name>', '<index_name>', '<type>')`
- `PRAGMA autoDB_find('<type>', '<key>')`
- `PRAGMA autoDB_size('<type>')`
- `PRAGMA auxillary_storage_size('<type_or_all>')`
- `PRAGMA run_insertion_benchmark('<benchmark_name>', '<type>', '<index_name>', <count>)`

Supported `<type>` values:

- `double`
- `bigint`
- `ubigint`
- `int` (or `integer`)

Supported benchmark names:

- `lognormal`
- `longitudes`
- `longlat`
- `ycsb`

### Core index workflow (copy/paste SQL)

```sql
-- If you are in plain DuckDB shell, load the extension first:
-- LOAD '/absolute/path/to/intelligent-duck/build/release/extension/autodb/autoDB.duckdb_extension';

CREATE TABLE demo_kv (
  id BIGINT,
  payload DOUBLE
);

INSERT INTO demo_kv VALUES
  (1, 10.5),
  (2, 22.0),
  (3, 31.7);

-- Build autoDB index on key column
PRAGMA create_autoDB_index('demo_kv', 'id');

-- Lookup by key using autoDB index
PRAGMA autoDB_find('bigint', '2');

-- Print index memory usage
PRAGMA autoDB_size('bigint');
PRAGMA auxillary_storage_size('bigint');

-- Insert a new key/payload pair into table + autoDB index
PRAGMA insert_into_table('demo_kv', 'bigint', '4', '40.2');
PRAGMA autoDB_find('bigint', '4');

-- Build ART index for comparison
PRAGMA create_art_index('demo_kv', 'id');
```

Important behavior notes:

- `create_autoDB_index` expects a key column type in `double/bigint/ubigint/int`.
- The payload is taken from the column immediately after the key column.
- `insert_into_table` inserts exactly two values `(key, payload)`, so use it with 2-column key/payload tables.
- `auxillary_storage_size` is intentionally spelled `auxillary` in this extension.

### Benchmark pragmas (requires benchmark files)

Before running benchmark pragmas, place dataset files in the benchmark directory and set:

```bash
export AUTODB_BENCHMARK_DIR=/path/to/benchmark/files
```

Expected benchmark files:

- `lognormal-190M.bin.data`
- `longitudes-200M.bin.data`
- `longlat-200M.bin.data`
- `ycsb-200M.bin.data`

Then run SQL like:

```sql
-- Load 1M rows from lognormal dataset into table bench_lognormal
-- 4th arg is currently required by signature but not used by loader logic.
PRAGMA load_benchmark('bench_lognormal', 'lognormal', 1000000, 0);

-- Build indexes on loaded benchmark table
PRAGMA create_autoDB_index('bench_lognormal', 'key');
PRAGMA create_art_index('bench_lognormal', 'key');

-- Full lookup benchmark workload (index_name: 'autodb' or 'art')
PRAGMA run_lookup_benchmark('lognormal', 'autodb');
PRAGMA run_lookup_benchmark('lognormal', 'art');

-- Single-batch lookup benchmark
PRAGMA run_benchmark_one_batch('lognormal', 'autodb', 'bigint');
PRAGMA run_benchmark_one_batch('lognormal', 'art', 'bigint');

-- Insertion benchmark (count = number of rows to insert)
PRAGMA run_insertion_benchmark('lognormal', 'bigint', 'autodb', 10000);
PRAGMA run_insertion_benchmark('lognormal', 'bigint', 'art', 10000);
```

## Useful Scripts

- `setup_duckdb.py`: creates DuckDB schema, demo data, and telemetry DB
- `run_demo_duckdb.py`: runs the 4-phase learning demo
- `main.py`: main orchestrator entry point
- `duck_connect_check.py`: checks extension load and compatibility
- `verify_windows_setup.py`: Windows setup diagnostics

## Troubleshooting

- `ModuleNotFoundError: duckdb`
  - Activate `venv` and reinstall dependencies: `pip install -r requirements.txt`
- `DuckDB extension not found`
  - Build extension first (`cd intelligent-duck && make release`)
  - Confirm `duckdb.extension_path` in `config.yaml`
- `does not match DuckDB loading it`
  - Rebuild extension for exact runtime platform/version, then rerun `python duck_connect_check.py`
- `Conflicting lock is held in python ...`
  - Stop other processes using the same DuckDB file before running setup/demo
