# generate_report.py
from telemetry.storage import TelemetryStorage
from utils.metrics import MetricsCalculator
from datetime import datetime
from pathlib import Path

# Initialize
ts = TelemetryStorage({'paths': {'telemetry_db': 'data/telemetry.db'}})
calc = MetricsCalculator({'paths': {}})

# Get all metrics
all_metrics = ts.get_recent_metrics(hours=24*365)  # Get all

print(f"Generating report from {len(all_metrics)} metrics...")

# Calculate summary
summary = calc.calculate_summary(all_metrics)

# Generate report
report = f"""
{'='*70}
DATABASE QUERY OPTIMIZER - PERFORMANCE REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTION SUMMARY
{'='*70}
Total Queries Executed: {summary['total_queries']:,}
Average Latency: {summary['avg_latency']:.2f} ms
P50 Latency: {summary['p50_latency']:.2f} ms
P95 Latency: {summary['p95_latency']:.2f} ms
P99 Latency: {summary['p99_latency']:.2f} ms
Min Latency: {summary['min_latency']:.2f} ms
Max Latency: {summary['max_latency']:.2f} ms
Success Rate: {summary['success_rate']:.2f}%

RESOURCE USAGE
{'='*70}
Average CPU: {summary['avg_cpu']:.2f}%
Average Memory: {summary['avg_memory']:.2f}%
Cache Hit Rate: {summary['cache_hit_rate']:.2f}%

LEARNING EVENTS
{'='*70}
Policy Updates: {len(ts.get_policy_updates())}
Safety Events: {len(ts.get_safety_events())}
Meta-Learning Generations: {len(ts.get_meta_learning_events())}

{'='*70}
"""

# Save report
report_path = Path('data/final_report.txt')
report_path.write_text(report)

print(f"\nReport saved to: {report_path}")
print(report)