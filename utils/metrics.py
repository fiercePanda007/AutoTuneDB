import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta


class MetricsCalculator:
    """Calculates various performance metrics from telemetry data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics calculator.
        
        Args:
            config: System configuration
        """
        self.config = config
        
    def calculate_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate summary statistics from a list of metrics.
        
        Args:
            metrics: List of metric dictionaries
            
        Returns:
            Dictionary containing summary statistics
        """
        if not metrics:
            return {
                'avg_latency': 0.0,
                'p50_latency': 0.0,
                'p95_latency': 0.0,
                'p99_latency': 0.0,
                'success_rate': 0.0,
                'cache_hit_rate': 0.0,
                'avg_cpu': 0.0,
                'avg_memory': 0.0,
                'total_queries': 0
            }
        
        # Extract execution times
        exec_times = [m.get('execution_time', 0) * 1000 for m in metrics]  # Convert to ms
        exec_times_array = np.array(exec_times)
        
        # Calculate latency percentiles
        summary = {
            'avg_latency': float(np.mean(exec_times_array)),
            'p50_latency': float(np.percentile(exec_times_array, 50)),
            'p95_latency': float(np.percentile(exec_times_array, 95)),
            'p99_latency': float(np.percentile(exec_times_array, 99)),
            'min_latency': float(np.min(exec_times_array)),
            'max_latency': float(np.max(exec_times_array)),
            'std_latency': float(np.std(exec_times_array))
        }
        
        # Calculate success rate
        successes = sum(1 for m in metrics if m.get('success', True))
        summary['success_rate'] = (successes / len(metrics)) * 100 if metrics else 0.0
        
        # Calculate cache hit rate
        cache_hits = [m.get('cache_hit_rate', 0) for m in metrics]
        summary['cache_hit_rate'] = float(np.mean(cache_hits)) if cache_hits else 0.0
        
        # Calculate resource usage
        cpu_usage = [m.get('cpu_usage', 0) for m in metrics]
        memory_usage = [m.get('memory_usage', 0) for m in metrics]
        summary['avg_cpu'] = float(np.mean(cpu_usage)) if cpu_usage else 0.0
        summary['avg_memory'] = float(np.mean(memory_usage)) if memory_usage else 0.0
        
        # Total queries
        summary['total_queries'] = len(metrics)
        
        return summary
        
    def calculate_improvement(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate percentage improvement from baseline to current.
        
        Args:
            baseline: Baseline metrics summary
            current: Current metrics summary
            
        Returns:
            Dictionary of improvement percentages
        """
        improvements = {}
        
        # For latency metrics, improvement means reduction
        for metric in ['avg_latency', 'p50_latency', 'p95_latency', 'p99_latency']:
            if baseline.get(metric, 0) > 0:
                improvement = ((baseline[metric] - current.get(metric, baseline[metric])) / 
                              baseline[metric]) * 100
                improvements[metric] = improvement
            else:
                improvements[metric] = 0.0
        
        # For success rate and cache hit rate, improvement means increase
        for metric in ['success_rate', 'cache_hit_rate']:
            if baseline.get(metric, 0) > 0:
                improvement = ((current.get(metric, 0) - baseline[metric]) / 
                              baseline[metric]) * 100
                improvements[metric] = improvement
            else:
                improvements[metric] = 0.0
        
        # Calculate overall resource efficiency improvement
        baseline_resources = baseline.get('avg_cpu', 0) + baseline.get('avg_memory', 0)
        current_resources = current.get('avg_cpu', 0) + current.get('avg_memory', 0)
        
        if baseline_resources > 0:
            improvements['resource_efficiency'] = ((baseline_resources - current_resources) / 
                                                   baseline_resources) * 100
        else:
            improvements['resource_efficiency'] = 0.0
        
        return improvements
        
    def calculate_sla_compliance(
        self,
        metrics: List[Dict[str, Any]],
        sla_threshold_ms: float = 100.0
    ) -> Dict[str, Any]:
        """
        Calculate SLA compliance metrics.
        
        Args:
            metrics: List of metric dictionaries
            sla_threshold_ms: SLA threshold in milliseconds
            
        Returns:
            Dictionary with SLA compliance metrics
        """
        if not metrics:
            return {
                'compliance_rate': 0.0,
                'violations': 0,
                'total': 0
            }
        
        exec_times = [m.get('execution_time', 0) * 1000 for m in metrics]
        violations = sum(1 for t in exec_times if t > sla_threshold_ms)
        
        return {
            'compliance_rate': ((len(metrics) - violations) / len(metrics)) * 100,
            'violations': violations,
            'total': len(metrics),
            'threshold_ms': sla_threshold_ms
        }
        
    def detect_anomalies(
        self,
        metrics: List[Dict[str, Any]],
        window_size: int = 100,
        std_threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalous queries using statistical methods.
        
        Args:
            metrics: List of metric dictionaries
            window_size: Size of rolling window
            std_threshold: Number of standard deviations for anomaly
            
        Returns:
            List of anomalous metric entries
        """
        if len(metrics) < window_size:
            return []
        
        anomalies = []
        exec_times = np.array([m.get('execution_time', 0) * 1000 for m in metrics])
        
        # Calculate rolling statistics
        for i in range(window_size, len(exec_times)):
            window = exec_times[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            # Check if current value is anomalous
            if abs(exec_times[i] - mean) > (std_threshold * std):
                anomaly_info = metrics[i].copy()
                anomaly_info['anomaly_score'] = abs(exec_times[i] - mean) / std if std > 0 else 0
                anomaly_info['expected_range'] = (mean - std_threshold * std, 
                                                  mean + std_threshold * std)
                anomalies.append(anomaly_info)
        
        return anomalies