import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class TelemetryStorage:
    """
    Storage backend for system telemetry with proper JSON handling.
    Uses SQLite for persistent storage with automatic deserialization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize telemetry storage.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.db_path = Path(config['paths']['telemetry_db'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database schema if not exists."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    phase TEXT NOT NULL,
                    query_type TEXT,
                    execution_time REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    cache_hit_rate REAL,
                    rows_processed INTEGER,
                    plan_cost REAL,
                    success INTEGER,
                    query_hash TEXT,
                    plan_info TEXT
                );
                
                CREATE TABLE IF NOT EXISTS policy_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    old_version INTEGER,
                    new_version INTEGER,
                    improvement REAL,
                    validation_score REAL,
                    changes TEXT
                );
                
                CREATE TABLE IF NOT EXISTS safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    severity TEXT NOT NULL,
                    event_type TEXT,
                    description TEXT,
                    action_taken TEXT,
                    context TEXT
                );
                
                CREATE TABLE IF NOT EXISTS meta_learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    generation INTEGER,
                    best_fitness REAL,
                    avg_fitness REAL,
                    hyperparameters TEXT,
                    improvements TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_phase ON metrics(phase);
                CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(query_type);
                CREATE INDEX IF NOT EXISTS idx_policy_timestamp ON policy_updates(timestamp);
                CREATE INDEX IF NOT EXISTS idx_safety_timestamp ON safety_events(timestamp);
            """)
            
            conn.commit()
            conn.close()
            
    def _deserialize_json_fields(
        self, 
        row: Dict[str, Any], 
        json_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Deserialize JSON string fields back to Python objects.
        
        Args:
            row: Dictionary representing a database row
            json_fields: List of field names that contain JSON strings
            
        Returns:
            Dictionary with JSON fields deserialized
        """
        result = dict(row)
        
        for field in json_fields:
            if field in result and result[field] is not None:
                try:
                    # Handle both string and already-deserialized objects
                    if isinstance(result[field], str):
                        result[field] = json.loads(result[field])
                    # If it's already a dict/list, leave it as is
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.warning(
                        f"Failed to deserialize {field}: {e}. "
                        f"Returning empty dict. Value was: {result[field]}"
                    )
                    result[field] = {}
                    
        return result
        
    def store_metric(self, metric: Dict[str, Any]):
        """
        Store a single metric entry.
        
        Args:
            metric: Dictionary containing metric data
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO metrics (
                    timestamp, phase, query_type, execution_time,
                    cpu_usage, memory_usage, cache_hit_rate,
                    rows_processed, plan_cost, success, query_hash, plan_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.get('timestamp', time.time()),
                metric.get('phase', 'unknown'),
                metric.get('query_type', 'unknown'),
                metric.get('execution_time', 0),
                metric.get('cpu_usage', 0),
                metric.get('memory_usage', 0),
                metric.get('cache_hit_rate', 0),
                metric.get('rows_processed', 0),
                metric.get('plan_cost', 0),
                1 if metric.get('success', True) else 0,
                metric.get('query_hash', ''),
                json.dumps(metric.get('plan_info', {}))
            ))
            
            conn.commit()
            conn.close()
    
    def store_metrics_batch(self, metrics: List[Dict[str, Any]]):
        """
        Store multiple metrics efficiently.
        
        Args:
            metrics: List of metric dictionaries
        """
        if not metrics:
            return
            
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            data = [
                (
                    m.get('timestamp', time.time()),
                    m.get('phase', 'unknown'),
                    m.get('query_type', 'unknown'),
                    m.get('execution_time', 0),
                    m.get('cpu_usage', 0),
                    m.get('memory_usage', 0),
                    m.get('cache_hit_rate', 0),
                    m.get('rows_processed', 0),
                    m.get('plan_cost', 0),
                    1 if m.get('success', True) else 0,
                    m.get('query_hash', ''),
                    json.dumps(m.get('plan_info', {}))
                )
                for m in metrics
            ]
            
            cursor.executemany("""
                INSERT INTO metrics (
                    timestamp, phase, query_type, execution_time,
                    cpu_usage, memory_usage, cache_hit_rate,
                    rows_processed, plan_cost, success, query_hash, plan_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            conn.commit()
            conn.close()
    
    def get_recent_metrics(
        self,
        minutes: Optional[int] = None,
        hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent metrics with proper JSON deserialization.
        
        Args:
            minutes: Number of minutes to look back
            hours: Number of hours to look back
            
        Returns:
            List of metric dictionaries with deserialized JSON fields
        """
        if hours:
            cutoff = time.time() - (hours * 3600)
        elif minutes:
            cutoff = time.time() - (minutes * 60)
        else:
            cutoff = time.time() - 3600  # Default 1 hour
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM metrics
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dicts and deserialize JSON fields
            results = []
            for row in rows:
                row_dict = dict(row)
                # Deserialize the plan_info JSON field
                deserialized = self._deserialize_json_fields(
                    row_dict, 
                    ['plan_info']
                )
                results.append(deserialized)
                
            return results
    
    def get_phase_metrics(self, phase: str) -> List[Dict[str, Any]]:
        """
        Get all metrics for a specific phase with proper JSON deserialization.
        
        Args:
            phase: Phase name
            
        Returns:
            List of metric dictionaries with deserialized JSON fields
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM metrics
                WHERE phase = ?
                ORDER BY timestamp ASC
            """, (phase,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dicts and deserialize JSON fields
            results = []
            for row in rows:
                row_dict = dict(row)
                # Deserialize the plan_info JSON field
                deserialized = self._deserialize_json_fields(
                    row_dict, 
                    ['plan_info']
                )
                results.append(deserialized)
                
            return results
    
    def store_policy_update(self, update: Dict[str, Any]):
        """
        Store policy update event.
        
        Args:
            update: Policy update information
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO policy_updates (
                    timestamp, old_version, new_version,
                    improvement, validation_score, changes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                update.get('timestamp', time.time()),
                update.get('old_version', 0),
                update.get('new_version', 0),
                update.get('improvement', 0),
                update.get('validation_score', 0),
                json.dumps(update.get('changes', {}))
            ))
            
            conn.commit()
            conn.close()
    
    def get_policy_updates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent policy updates with proper JSON deserialization.
        
        Args:
            limit: Maximum number of updates to return
            
        Returns:
            List of policy update dictionaries with deserialized JSON fields
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM policy_updates
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dicts and deserialize JSON fields
            results = []
            for row in rows:
                row_dict = dict(row)
                # Deserialize the changes JSON field
                deserialized = self._deserialize_json_fields(
                    row_dict, 
                    ['changes']
                )
                results.append(deserialized)
                
            return results
    
    def store_safety_event(self, event: Dict[str, Any]):
        """
        Store safety monitoring event.
        
        Args:
            event: Safety event information
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO safety_events (
                    timestamp, severity, event_type,
                    description, action_taken, context
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.get('timestamp', time.time()),
                event.get('severity', 'info'),
                event.get('event_type', 'unknown'),
                event.get('description', ''),
                event.get('action_taken', ''),
                json.dumps(event.get('context', {}))
            ))
            
            conn.commit()
            conn.close()
    
    def get_safety_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent safety events with proper JSON deserialization.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of safety event dictionaries with deserialized JSON fields
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM safety_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dicts and deserialize JSON fields
            results = []
            for row in rows:
                row_dict = dict(row)
                # Deserialize the context JSON field
                deserialized = self._deserialize_json_fields(
                    row_dict, 
                    ['context']
                )
                results.append(deserialized)
                
            return results
    
    def store_meta_learning_event(self, event: Dict[str, Any]):
        """
        Store meta-learning event.
        
        Args:
            event: Meta-learning event information
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO meta_learning_events (
                    timestamp, generation, best_fitness,
                    avg_fitness, hyperparameters, improvements
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.get('timestamp', time.time()),
                event.get('generation', 0),
                event.get('best_fitness', 0),
                event.get('avg_fitness', 0),
                json.dumps(event.get('hyperparameters', {})),
                json.dumps(event.get('improvements', {}))
            ))
            
            conn.commit()
            conn.close()
    
    def get_meta_learning_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent meta-learning events with proper JSON deserialization.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of meta-learning event dictionaries with deserialized JSON fields
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM meta_learning_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dicts and deserialize JSON fields
            results = []
            for row in rows:
                row_dict = dict(row)
                # Deserialize both hyperparameters and improvements JSON fields
                deserialized = self._deserialize_json_fields(
                    row_dict, 
                    ['hyperparameters', 'improvements']
                )
                results.append(deserialized)
                
            return results
    
    def print_summary(self):
        """Print summary statistics of stored telemetry."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM metrics")
            metric_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM policy_updates")
            policy_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM safety_events")
            safety_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM meta_learning_events")
            meta_count = cursor.fetchone()[0]
            
            # Get phase breakdown
            cursor.execute("""
                SELECT phase, COUNT(*) as count
                FROM metrics
                GROUP BY phase
            """)
            phases = cursor.fetchall()
            
            conn.close()
            
            print("\nTelemetry Storage Summary")
            print("="*50)
            print(f"Total Metrics: {metric_count:,}")
            print(f"Policy Updates: {policy_count:,}")
            print(f"Safety Events: {safety_count:,}")
            print(f"Meta-Learning Events: {meta_count:,}")
            print("\nMetrics by Phase:")
            for phase, count in phases:
                print(f"  {phase}: {count:,}")
            print("="*50)