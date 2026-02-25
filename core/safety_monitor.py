import logging
import time
import psutil
from typing import Dict, Any, List
from collections import deque


class SafetyMonitor:
    """
    Monitors system health and enforces safety constraints.
    Prevents the self-improving system from damaging itself.
    """
    
    def __init__(self, config: Dict[str, Any], telemetry_storage):
        """
        Initialize safety monitor.
        
        Args:
            config: System configuration
            telemetry_storage: TelemetryStorage instance
        """
        self.config = config
        self.telemetry = telemetry_storage
        self.logger = logging.getLogger(__name__)
        
        self.safety_config = config['safety']
        self.enabled = self.safety_config['enabled']
        
        # Thresholds
        self.memory_limit = self.safety_config['memory_limit_gb'] * 1024 * 1024 * 1024
        self.cpu_limit = self.safety_config['cpu_limit_percent']
        self.rollback_threshold = self.safety_config['rollback_threshold']
        self.max_failures = self.safety_config['max_consecutive_failures']
        
        # Monitoring state
        self.consecutive_failures = 0
        self.recent_metrics = deque(maxlen=100)
        self.baseline_performance = None
        
        # Process monitoring
        self.process = psutil.Process()
        
        self.logger.info("Safety Monitor initialized")
        
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.
        
        Returns:
            Dictionary containing health status and any issues
        """
        if not self.enabled:
            return {'healthy': True, 'issues': []}
        
        issues = []
        
        # Check resource usage
        resource_issues = self._check_resources()
        if resource_issues:
            issues.extend(resource_issues)
        
        # Check performance degradation
        perf_issues = self._check_performance()
        if perf_issues:
            issues.extend(perf_issues)
        
        # Check failure rate
        failure_issues = self._check_failures()
        if failure_issues:
            issues.extend(failure_issues)
        
        # Determine severity
        severity = 'ok'
        if issues:
            if any('critical' in str(issue) for issue in issues):
                severity = 'critical'
            else:
                severity = 'warning'
        
        return {
            'healthy': len(issues) == 0,
            'severity': severity,
            'issues': issues,
            'timestamp': time.time()
        }
    
    def _check_resources(self) -> List[str]:
        """Check system resource usage."""
        issues = []
        
        try:
            # Memory check
            memory_info = self.process.memory_info()
            if memory_info.rss > self.memory_limit:
                issues.append(
                    f"Memory usage ({memory_info.rss / 1024**3:.1f} GB) "
                    f"exceeds limit ({self.memory_limit / 1024**3:.1f} GB)"
                )
                self._record_safety_event('critical', 'memory_limit_exceeded')
            
            # CPU check
            cpu_percent = self.process.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_limit:
                issues.append(
                    f"CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.cpu_limit}%)"
                )
                self._record_safety_event('warning', 'cpu_limit_exceeded')
        
        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
        
        return issues
    
    def _check_performance(self) -> List[str]:
        """Check for performance degradation."""
        issues = []
        
        # Get recent metrics
        recent = self.telemetry.get_recent_metrics(hours=1)
        if not recent:
            return issues
        
        # Calculate current performance
        exec_times = [m.get('execution_time', 0) for m in recent]
        current_avg = sum(exec_times) / len(exec_times)
        
        # Compare to baseline
        if self.baseline_performance is None:
            # Set baseline from first measurements
            baseline_metrics = self.telemetry.get_phase_metrics('baseline')
            if baseline_metrics:
                baseline_times = [m.get('execution_time', 0) for m in baseline_metrics]
                self.baseline_performance = sum(baseline_times) / len(baseline_times)
        
        if self.baseline_performance:
            degradation = (current_avg - self.baseline_performance) / self.baseline_performance
            
            if degradation > self.rollback_threshold:
                issues.append(
                    f"Performance degraded by {degradation*100:.1f}% "
                    f"(threshold: {self.rollback_threshold*100:.1f}%)"
                )
                self._record_safety_event('critical', 'performance_degradation')
        
        return issues
    
    def _check_failures(self) -> List[str]:
        """Check for excessive failures."""
        issues = []
        
        # Get recent metrics
        recent = self.telemetry.get_recent_metrics(minutes=5)
        if not recent:
            return issues
        
        # Count consecutive failures
        failures = 0
        for metric in reversed(recent):
            if not metric.get('success', True):
                failures += 1
            else:
                break
        
        self.consecutive_failures = failures
        
        if failures >= self.max_failures:
            issues.append(
                f"Excessive consecutive failures: {failures} "
                f"(threshold: {self.max_failures})"
            )
            self._record_safety_event('critical', 'excessive_failures')
        
        return issues
    
    def _record_safety_event(self, severity: str, event_type: str):
        """Record a safety event."""
        self.telemetry.store_safety_event({
            'severity': severity,
            'event_type': event_type,
            'description': f"Safety monitor detected {event_type}",
            'action_taken': 'Monitoring' if severity == 'warning' else 'Alert triggered'
        })
    
    def validate_change(self, change_description: str) -> bool:
        """
        Validate a proposed system change.
        
        Args:
            change_description: Description of proposed change
            
        Returns:
            True if change is safe to apply
        """
        if not self.enabled:
            return True
        
        # Check current system health
        health = self.check_system_health()
        
        if not health['healthy'] and health['severity'] == 'critical':
            self.logger.warning(
                f"Change validation failed: system unhealthy. "
                f"Issues: {health['issues']}"
            )
            return False
        
        # In a full implementation, would run the change in a sandbox
        # For now, just check if validation is required
        if self.safety_config['validation_required']:
            self.logger.info(f"Validating change: {change_description}")
            # Simulate validation passing
            return True
        
        return True