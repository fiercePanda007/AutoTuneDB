import time
import psutil
import hashlib
from typing import Dict, Any, Optional
from collections import deque


class TelemetryCollector:
    """Collects real-time performance telemetry."""
    
    def __init__(self, config: Dict[str, Any], storage):
        """
        Initialize telemetry collector.
        
        Args:
            config: System configuration
            storage: TelemetryStorage instance
        """
        self.config = config
        self.storage = storage
        self.enabled = config['telemetry']['enabled']
        self.current_phase = "initialization"
        
        # Buffer for batch writes
        self.buffer = deque(maxlen=100)
        self.last_flush = time.time()
        self.flush_interval = 10  # seconds
        
        # System monitoring
        self.process = psutil.Process()
        
    def set_phase(self, phase: str):
        """Set current system phase for telemetry tagging."""
        self.current_phase = phase
        
    def record_execution(
        self,
        query: str,
        query_type: str,
        execution_time: float,
        resources: Dict[str, float],
        plan_info: Dict[str, Any],
        success: bool = True,
        action: Optional[str] = None, #added later
    ):
       
        # ensure plan_info is dict
        if plan_info is None:
            plan_info = {}
        elif not isinstance(plan_info, dict):
            plan_info = {"raw": str(plan_info)}

        # inject chosen action
        if action is not None:
            plan_info["action"] = action   # <-- CRITICAL

        """
        Record query execution metrics.
        
        Args:
            query: SQL query string
            query_type: Type of query
            execution_time: Time taken to execute (seconds)
            resources: Resource usage metrics
            plan_info: Query plan information
            success: Whether execution succeeded
        """
        if not self.enabled:
            return
        
        # Generate query hash for tracking similar queries
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        
        metric = {
            'timestamp': time.time(),
            'phase': self.current_phase,
            'query_type': query_type,
            'execution_time': execution_time,
            'cpu_usage': resources.get('cpu', 0),
            'memory_usage': resources.get('memory', 0),
            'cache_hit_rate': resources.get('cache_hit_rate', 0),
            'rows_processed': plan_info.get('rows', 0),
            'plan_cost': plan_info.get('cost', 0),
            'success': success,
            'query_hash': query_hash,
            'plan_info': plan_info
        }
        
        self.buffer.append(metric)
        
        # Flush buffer if needed
        if (len(self.buffer) >= 50 or 
            time.time() - self.last_flush > self.flush_interval):
            self.flush()
    
    def flush(self):
        """Flush buffered metrics to storage."""
        if not self.buffer:
            return
            
        metrics = list(self.buffer)
        self.buffer.clear()
        
        try:
            self.storage.store_metrics_batch(metrics)
            self.last_flush = time.time()
        except Exception as e:
            # Log error but don't fail the system
            import logging
            logging.getLogger(__name__).error(f"Error flushing metrics: {e}")
    
    def get_system_load(self) -> Dict[str, float]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with CPU and memory usage
        """
        try:
            return {
                'cpu_percent': self.process.cpu_percent(interval=0.1),
                'memory_percent': self.process.memory_percent(),
                'memory_mb': self.process.memory_info().rss / 1024 / 1024
            }
        except:
            return {'cpu_percent': 0, 'memory_percent': 0, 'memory_mb': 0}
    
    def record_policy_update(self, update_info: Dict[str, Any]):
        """Record a policy update event."""
        update_info['timestamp'] = time.time()
        self.storage.store_policy_update(update_info)
    
    def record_safety_event(self, event_info: Dict[str, Any]):
        """Record a safety monitoring event."""
        event_info['timestamp'] = time.time()
        self.storage.store_safety_event(event_info)
    
    def record_meta_learning(self, event_info: Dict[str, Any]):
        """Record a meta-learning event."""
        event_info['timestamp'] = time.time()
        self.storage.store_meta_learning_event(event_info)