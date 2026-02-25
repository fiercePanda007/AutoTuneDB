# Self-Improving Database Query Optimizer (DuckDB + autoDB)

This project is a DuckDB-native version of the self-improving query optimizer:

- Level 0: DQN-driven query action selection
- Level 1: policy learning from telemetry
- Level 2: meta-learning of training parameters

The runtime pipeline is:

`query_optimizer -> DuckDB -> autoDB extension`

## Requirements

- Python 3.10+
- `duckdb` Python package
- Built `autoDB.duckdb_extension` from `intelligent-duck` (required by default config)

## Quick Start

```bash
# 1) Create and activate venv
python -m venv venv
source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Validate extension compatibility
python duck_connect_check.py

# 4) Initialize DuckDB data + telemetry
python setup_duckdb.py --scale 0.1

# 5) Quick demo run
python run_demo_duckdb.py --duration 0.1 --fast-mode
```

## Important Scripts

- `setup_duckdb.py`: creates schema, generates demo data, creates telemetry DB
- `run_demo_duckdb.py`: runs the 4-phase learning demo
- `duck_connect_check.py`: checks extension ABI/platform match and loadability
- `verify_windows_setup.py`: Windows-oriented environment checks for DuckDB pipeline

Compatibility wrappers:

- `setup_database.py` forwards to `setup_duckdb.py`
- `run_demo.py` forwards to `run_demo_duckdb.py`

## autoDB Notes

- Extension APIs used/available:
  - `PRAGMA create_autoDB_index('<table>', '<column>')`
  - `PRAGMA autoDB_find('<type>', '<key>')`
  - `PRAGMA autoDB_size('<type>')`
- The default config bootstraps a set of autoDB indexes at startup.
- Workload can include `autodb_lookup` queries when `workload.enable_autodb_queries: true`.

## Troubleshooting

- `ModuleNotFoundError: duckdb`
  - Use the project venv interpreter, or install requirements again.
- `Conflicting lock is held in python ...`
  - Stop other processes using the same DuckDB file before running the demo.
- Extension mismatch (`does not match DuckDB loading it`)
  - Rebuild `intelligent-duck` for the exact runtime platform/version and rerun `duck_connect_check.py`.
