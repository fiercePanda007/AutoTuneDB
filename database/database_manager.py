"""
Compatibility shim for projects importing `database.database_manager`.

The codebase now uses DuckDB only; this module re-exports the DuckDB manager.
"""

from database.database_manager_duck import DatabaseManager

__all__ = ["DatabaseManager"]
