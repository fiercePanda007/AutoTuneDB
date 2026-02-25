"""
Compatibility shim for projects importing `database.workload_generator`.

The codebase now uses the DuckDB workload generator implementation.
"""

from database.workload_generator_duck import WorkloadGenerator

__all__ = ["WorkloadGenerator"]
