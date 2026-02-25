import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telemetry.storage import TelemetryStorage


def generate_comprehensive_report():
    """Generate a comprehensive performance report."""
    
    print("\n" + "="*70)
    print("GENERATING PERFORMANCE REPORT")
    print("="*70)
    
    # Initialize telemetry storage
    config = {'paths': {'telemetry_db': 'data/telemetry.db'}}
    
    try:
        ts = TelemetryStorage(config)
    except Exception as e:
        print(f"\n❌ Error: Could not connect to telemetry database: {e}")
        print("\nMake sure you've run the demonstration first:")
        print("  python run_demo_duckdb.py --duration 0.1 --fast-mode")
        return
    
    # Get all metrics
    print("\nRetrieving telemetry data...")
    all_metrics = ts.get_recent_metrics(hours=24*365)  # Get all data
    
    if not all_metrics:
        print("\n❌ No metrics found in database!")
        print("\nRun a demonstration first:")
        print("  python run_demo_duckdb.py --duration 0.1 --fast-mode")
        return
    
    print(f"Found {len(all_metrics)} query executions")
    
    # Calculate basic statistics
    import numpy as np
    
    exec_times = [m['execution_time'] * 1000 for m in all_metrics]  # Convert to ms
    exec_times_array = np.array(exec_times)
    
    successes = sum(1 for m in all_metrics if m.get('success', True))
    success_rate = (successes / len(all_metrics)) * 100
    
    cpu_usage = [m.get('cpu_usage', 0) for m in all_metrics if m.get('cpu_usage', 0) > 0]
    memory_usage = [m.get('memory_usage', 0) for m in all_metrics if m.get('memory_usage', 0) > 0]
    cache_hits = [m.get('cache_hit_rate', 0) for m in all_metrics if m.get('cache_hit_rate', 0) > 0]
    
    # Get learning events
    policy_updates = ts.get_policy_updates()
    safety_events = ts.get_safety_events()
    meta_events = ts.get_meta_learning_events()
    
    # Get phase breakdown
    phases = {}
    for m in all_metrics:
        phase = m.get('phase', 'unknown')
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(m['execution_time'] * 1000)
    
    # Generate report
    report = f"""
{'='*70}
DATABASE QUERY OPTIMIZER - PERFORMANCE REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Database: data/telemetry.db

EXECUTION SUMMARY
{'='*70}
Total Queries Executed: {len(all_metrics):,}
Success Rate: {success_rate:.2f}%
Failed Queries: {len(all_metrics) - successes:,}

QUERY LATENCY STATISTICS
{'='*70}
Average Latency: {np.mean(exec_times_array):.2f} ms
Median (P50) Latency: {np.percentile(exec_times_array, 50):.2f} ms
P95 Latency: {np.percentile(exec_times_array, 95):.2f} ms
P99 Latency: {np.percentile(exec_times_array, 99):.2f} ms
Min Latency: {np.min(exec_times_array):.2f} ms
Max Latency: {np.max(exec_times_array):.2f} ms
Standard Deviation: {np.std(exec_times_array):.2f} ms

RESOURCE USAGE
{'='*70}
Average CPU Usage: {np.mean(cpu_usage):.2f}% ({len(cpu_usage)} samples)
Average Memory Usage: {np.mean(memory_usage):.2f}% ({len(memory_usage)} samples)
Average Cache Hit Rate: {np.mean(cache_hits):.2f}% ({len(cache_hits)} samples)

LEARNING SYSTEM ACTIVITY
{'='*70}
Policy Updates (Level 1): {len(policy_updates)}
Safety Events Logged: {len(safety_events)}
Meta-Learning Generations (Level 2): {len(meta_events)}

PHASE BREAKDOWN
{'='*70}"""

    # Add phase statistics
    for phase_name, phase_times in sorted(phases.items()):
        phase_array = np.array(phase_times)
        report += f"""
{phase_name.upper()}:
  Queries: {len(phase_times):,}
  Avg Latency: {np.mean(phase_array):.2f} ms
  P95 Latency: {np.percentile(phase_array, 95):.2f} ms
  P99 Latency: {np.percentile(phase_array, 99):.2f} ms
"""

    # Add policy update details if any
    if policy_updates:
        report += f"""
POLICY UPDATE DETAILS
{'='*70}"""
        for i, update in enumerate(policy_updates[:5], 1):  # Show last 5
            report += f"""
Update {i}:
  Version: {update.get('old_version', 'N/A')} → {update.get('new_version', 'N/A')}
  Expected Improvement: {update.get('improvement', 0)*100:.2f}%
  Validation Score: {update.get('validation_score', 0):.4f}
  Timestamp: {datetime.fromtimestamp(update.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}
"""
        if len(policy_updates) > 5:
            report += f"\n... and {len(policy_updates) - 5} more policy updates\n"

    # Add meta-learning details if any
    if meta_events:
        report += f"""
META-LEARNING DETAILS (Level 2)
{'='*70}"""
        for i, event in enumerate(meta_events[:3], 1):  # Show last 3
            hyperparams = event.get('hyperparameters', {})
            report += f"""
Generation {event.get('generation', 'N/A')}:
  Best Fitness: {event.get('best_fitness', 0):.4f}
  Avg Fitness: {event.get('avg_fitness', 0):.4f}
  Learning Rate: {hyperparams.get('learning_rate', 'N/A')}
  Batch Size: {hyperparams.get('batch_size', 'N/A')}
  Timestamp: {datetime.fromtimestamp(event.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Add safety events if any critical ones
    critical_safety = [e for e in safety_events if e.get('severity') == 'critical']
    if critical_safety:
        report += f"""
CRITICAL SAFETY EVENTS
{'='*70}"""
        for event in critical_safety[:10]:  # Show up to 10
            report += f"""
  Type: {event.get('event_type', 'unknown')}
  Description: {event.get('description', 'N/A')}
  Action Taken: {event.get('action_taken', 'N/A')}
  Time: {datetime.fromtimestamp(event.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}
"""

    report += f"""
{'='*70}
END OF REPORT
{'='*70}
"""

    # Save report
    report_path = Path('data/final_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    
    # Print report
    print(report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Also save JSON version for programmatic access
    json_report = {
        'generated_at': datetime.now().isoformat(),
        'total_queries': len(all_metrics),
        'success_rate': success_rate,
        'latency_stats': {
            'avg_ms': float(np.mean(exec_times_array)),
            'p50_ms': float(np.percentile(exec_times_array, 50)),
            'p95_ms': float(np.percentile(exec_times_array, 95)),
            'p99_ms': float(np.percentile(exec_times_array, 99)),
            'min_ms': float(np.min(exec_times_array)),
            'max_ms': float(np.max(exec_times_array)),
            'std_ms': float(np.std(exec_times_array))
        },
        'resource_usage': {
            'avg_cpu_percent': float(np.mean(cpu_usage)) if cpu_usage else 0,
            'avg_memory_percent': float(np.mean(memory_usage)) if memory_usage else 0,
            'avg_cache_hit_rate': float(np.mean(cache_hits)) if cache_hits else 0
        },
        'learning_activity': {
            'policy_updates': len(policy_updates),
            'safety_events': len(safety_events),
            'meta_learning_generations': len(meta_events)
        },
        'phases': {
            phase: {
                'count': len(times),
                'avg_latency_ms': float(np.mean(times)),
                'p95_latency_ms': float(np.percentile(times, 95))
            }
            for phase, times in phases.items()
        }
    }
    
    json_path = Path('data/final_report.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"✓ JSON report saved to: {json_path}")
    
    return report


if __name__ == "__main__":
    try:
        generate_comprehensive_report()
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
