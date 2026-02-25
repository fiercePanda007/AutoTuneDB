import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telemetry.storage import TelemetryStorage


def investigate_phases():
    """Investigate why all metrics are in 'initialization' phase."""
    print("\n" + "="*70)
    print("PHASE INVESTIGATION")
    print("="*70)
    
    config = {'paths': {'telemetry_db': 'data/telemetry.db'}}
    ts = TelemetryStorage(config)
    
    # Get all metrics
    all_metrics = ts.get_recent_metrics(hours=24*365)
    
    # Count by phase
    phase_counts = {}
    for m in all_metrics:
        phase = m.get('phase', 'unknown')
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print(f"\nTotal metrics: {len(all_metrics)}")
    print("\nPhase distribution:")
    for phase, count in sorted(phase_counts.items()):
        percentage = (count / len(all_metrics)) * 100
        print(f"  {phase}: {count:,} ({percentage:.1f}%)")
    
    # Sample a few metrics to see what phase values look like
    print("\nSample metrics (first 5):")
    for i, m in enumerate(all_metrics[:5], 1):
        print(f"\nMetric {i}:")
        print(f"  Phase: '{m.get('phase')}'")
        print(f"  Query Type: {m.get('query_type')}")
        print(f"  Timestamp: {datetime.fromtimestamp(m.get('timestamp', 0))}")
        print(f"  Execution Time: {m.get('execution_time', 0)*1000:.2f} ms")
    
    # Check if phase field exists and has expected values
    expected_phases = ['baseline', 'level0_learning', 'level1_learning', 
                       'level2_learning', 'initialization']
    
    print("\n" + "-"*70)
    print("Expected vs Actual Phases:")
    for phase in expected_phases:
        count = phase_counts.get(phase, 0)
        status = "✓" if count > 0 else "❌"
        print(f"  {status} {phase}: {count:,}")
    
    if phase_counts.get('initialization', 0) == len(all_metrics):
        print("\n⚠️  PROBLEM: All metrics are in 'initialization' phase!")
        print("   This suggests the orchestrator isn't updating the phase field.")
        print("   Check main.py to ensure current_phase is being set correctly.")


def investigate_resource_metrics():
    """Investigate why resource metrics are missing."""
    print("\n" + "="*70)
    print("RESOURCE METRICS INVESTIGATION")
    print("="*70)
    
    config = {'paths': {'telemetry_db': 'data/telemetry.db'}}
    ts = TelemetryStorage(config)
    
    all_metrics = ts.get_recent_metrics(hours=24*365)
    
    # Count non-zero values
    cpu_samples = sum(1 for m in all_metrics if m.get('cpu_usage', 0) > 0)
    memory_samples = sum(1 for m in all_metrics if m.get('memory_usage', 0) > 0)
    cache_samples = sum(1 for m in all_metrics if m.get('cache_hit_rate', 0) > 0)
    
    print(f"\nTotal metrics: {len(all_metrics)}")
    print(f"\nResource metric samples:")
    print(f"  CPU Usage: {cpu_samples:,} samples ({cpu_samples/len(all_metrics)*100:.1f}%)")
    print(f"  Memory Usage: {memory_samples:,} samples ({memory_samples/len(all_metrics)*100:.1f}%)")
    print(f"  Cache Hit Rate: {cache_samples:,} samples ({cache_samples/len(all_metrics)*100:.1f}%)")
    
    # Sample actual values
    print("\nSample resource values (first 10 non-zero):")
    found = 0
    for m in all_metrics:
        if (m.get('cpu_usage', 0) > 0 or 
            m.get('memory_usage', 0) > 0 or 
            m.get('cache_hit_rate', 0) > 0):
            print(f"  CPU: {m.get('cpu_usage', 0):.2f}%, "
                  f"Memory: {m.get('memory_usage', 0):.2f}%, "
                  f"Cache: {m.get('cache_hit_rate', 0):.4f}")
            found += 1
            if found >= 10:
                break
    
    if found == 0:
        print("  ❌ No non-zero resource metrics found!")
        print("\n⚠️  PROBLEM: Resource metrics are not being collected!")
        print("   Check telemetry/collector.py to ensure resources are recorded.")
        print("   The record_execution method should include cpu_usage, memory_usage, etc.")


def investigate_safety_events():
    """Investigate the 51 safety events."""
    print("\n" + "="*70)
    print("SAFETY EVENTS INVESTIGATION")
    print("="*70)
    
    config = {'paths': {'telemetry_db': 'data/telemetry.db'}}
    ts = TelemetryStorage(config)
    
    events = ts.get_safety_events(limit=100)
    
    print(f"\nTotal safety events: {len(events)}")
    
    # Categorize by severity and type
    by_severity = {}
    by_type = {}
    
    for event in events:
        severity = event.get('severity', 'unknown')
        event_type = event.get('event_type', 'unknown')
        
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_type[event_type] = by_type.get(event_type, 0) + 1
    
    print("\nBy Severity:")
    for severity, count in sorted(by_severity.items()):
        print(f"  {severity}: {count}")
    
    print("\nBy Type:")
    for event_type, count in sorted(by_type.items()):
        print(f"  {event_type}: {count}")
    
    # Show recent events
    print("\nRecent safety events (last 10):")
    for i, event in enumerate(events[:10], 1):
        print(f"\n{i}. [{event.get('severity', 'unknown').upper()}] "
              f"{event.get('event_type', 'unknown')}")
        print(f"   Time: {datetime.fromtimestamp(event.get('timestamp', 0))}")
        print(f"   Description: {event.get('description', 'N/A')}")
        print(f"   Action: {event.get('action_taken', 'N/A')}")
        
        # Show context if available
        context = event.get('context', {})
        if context and isinstance(context, dict) and context:
            print(f"   Context: {context}")


def investigate_policy_validation():
    """Investigate why policy updates are failing validation."""
    print("\n" + "="*70)
    print("POLICY VALIDATION INVESTIGATION")
    print("="*70)
    
    config = {'paths': {'telemetry_db': 'data/telemetry.db'}}
    ts = TelemetryStorage(config)
    
    # Get metrics by phase for comparison
    all_metrics = ts.get_recent_metrics(hours=24*365)
    
    # Calculate baseline stats (if we have baseline phase)
    baseline = [m for m in all_metrics if m.get('phase') == 'baseline']
    current = [m for m in all_metrics if m.get('phase') != 'baseline']
    
    print(f"\nBaseline metrics: {len(baseline)}")
    print(f"Non-baseline metrics: {len(current)}")
    
    if len(baseline) == 0:
        print("\n⚠️  PROBLEM: No baseline metrics found!")
        print("   Without baseline data, the policy learner cannot calculate improvement.")
        print("   This is likely due to the phase tracking issue.")
    else:
        # Calculate performance comparison
        import numpy as np
        
        baseline_times = [m.get('execution_time', 0) * 1000 for m in baseline]
        current_times = [m.get('execution_time', 0) * 1000 for m in current] if current else baseline_times
        
        baseline_avg = np.mean(baseline_times)
        current_avg = np.mean(current_times)
        
        improvement = ((baseline_avg - current_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
        
        print(f"\nPerformance comparison:")
        print(f"  Baseline avg latency: {baseline_avg:.2f} ms")
        print(f"  Current avg latency: {current_avg:.2f} ms")
        print(f"  Improvement: {improvement:.2f}%")
        
        print("\n  Note: Policy validation requires minimum improvement threshold.")
        print("        Check config.yaml for level1.min_improvement setting.")


def main():
    """Run all investigations."""
    print("\n" + "="*70)
    print("TELEMETRY ISSUES DIAGNOSTIC TOOL")
    print("="*70)
    
    try:
        investigate_phases()
        investigate_resource_metrics()
        investigate_safety_events()
        investigate_policy_validation()
        
        print("\n" + "="*70)
        print("SUMMARY OF ISSUES")
        print("="*70)
        print("""
Key findings:
1. Phase tracking - Check if current_phase is being updated in main.py
2. Resource metrics - Check if telemetry collector records CPU/memory/cache
3. Safety events - Review what triggered these events
4. Policy validation - May need baseline phase data or adjusted thresholds

Recommendations:
- Review main.py orchestrator to ensure phase transitions
- Verify telemetry/collector.py includes resource metrics
- Check config.yaml for validation thresholds
- Consider running longer demonstrations for more data
        """)
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()