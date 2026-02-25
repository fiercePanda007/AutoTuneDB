"""
Fixed Dashboard for Database Query Optimizer
Works with actual telemetry.db schema
"""

import sys
from flask import Flask, render_template, jsonify
from pathlib import Path
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
import traceback

# Create Flask app with explicit paths
app = Flask(__name__, 
            template_folder='dashboard/templates',
            static_folder='dashboard/static')

app.config['PROPAGATE_EXCEPTIONS'] = True


class DashboardData:
    """Handles data retrieval for dashboard."""
    
    def __init__(self, db_path: str = "data/telemetry.db"):
        self.db_path = Path(db_path).resolve()
        print(f"Database path: {self.db_path}", file=sys.stderr)
    
    def get_connection(self) -> Optional[sqlite3.Connection]:
        """Get database connection."""
        try:
            if not self.db_path.exists():
                print(f"Database not found: {self.db_path}", file=sys.stderr)
                return None
            return sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        except Exception as e:
            print(f"Database connection error: {e}", file=sys.stderr)
            traceback.print_exc()
            return None
    
    def get_recent_metrics(self, limit: int = 200) -> List[Dict]:
        """Get recent query metrics from metrics table."""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    timestamp,
                    phase,
                    execution_time,
                    success
                FROM metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            metrics = []
            for row in cursor.fetchall():
                # Convert execution_time from seconds to milliseconds
                latency_ms = float(row[2]) * 1000 if row[2] else 0.0
                
                metrics.append({
                    'timestamp': float(row[0]) * 1000,  # Convert to JS timestamp (ms)
                    'phase': str(row[1]) if row[1] else 'unknown',
                    'latency_ms': round(latency_ms, 2),
                    'success': bool(row[3])
                })
            
            return metrics
        except Exception as e:
            print(f"Error fetching recent metrics: {e}", file=sys.stderr)
            traceback.print_exc()
            return []
        finally:
            conn.close()
    
    def get_phase_summary(self) -> Dict:
        """Get summary statistics by phase."""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    phase,
                    COUNT(*) as query_count,
                    AVG(execution_time) * 1000 as avg_latency,
                    MIN(execution_time) * 1000 as min_latency,
                    MAX(execution_time) * 1000 as max_latency,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM metrics
                WHERE phase IS NOT NULL
                GROUP BY phase
                ORDER BY phase
            """)
            
            summary = {}
            for row in cursor.fetchall():
                phase = str(row[0])
                summary[phase] = {
                    'query_count': int(row[1]),
                    'avg_latency': round(float(row[2]), 2) if row[2] else 0.0,
                    'min_latency': round(float(row[3]), 2) if row[3] else 0.0,
                    'max_latency': round(float(row[4]), 2) if row[4] else 0.0,
                    'success_rate': round(float(row[5]), 2) if row[5] else 0.0
                }
            
            return summary
        except Exception as e:
            print(f"Error fetching phase summary: {e}", file=sys.stderr)
            traceback.print_exc()
            return {}
        finally:
            conn.close()
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        conn = self.get_connection()
        if not conn:
            return {'policy_updates': 0, 'meta_learning_runs': 0, 'latest_loss': None}
        
        try:
            cursor = conn.cursor()
            
            # Count policy updates
            cursor.execute("SELECT COUNT(*) FROM policy_updates")
            policy_updates = cursor.fetchone()[0]
            
            # Count meta-learning events
            cursor.execute("SELECT COUNT(*) FROM meta_learning_events")
            meta_learning_runs = cursor.fetchone()[0]
            
            # Get latest recorded DQN training loss from execution telemetry.
            # Loss is stored in metrics.plan_info.training_loss.
            cursor.execute("""
                SELECT json_extract(plan_info, '$.training_loss')
                FROM metrics
                WHERE json_type(plan_info, '$.training_loss') IN ('real', 'integer')
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            result = cursor.fetchone()
            latest_loss = float(result[0]) if result else None
            
            return {
                'policy_updates': policy_updates,
                'meta_learning_runs': meta_learning_runs,
                'latest_loss': latest_loss
            }
        except Exception as e:
            print(f"Error fetching learning stats: {e}", file=sys.stderr)
            traceback.print_exc()
            return {'policy_updates': 0, 'meta_learning_runs': 0, 'latest_loss': None}
        finally:
            conn.close()
    
    def get_latency_distribution(self) -> Dict:
        """Get latency percentile distribution."""
        conn = self.get_connection()
        if not conn:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT execution_time * 1000 as latency_ms
                FROM metrics
                WHERE execution_time IS NOT NULL
                ORDER BY execution_time
            """)
            
            latencies = [float(row[0]) for row in cursor.fetchall() if row[0]]
            
            if not latencies:
                return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
            
            n = len(latencies)
            
            def get_percentile(data, pct):
                idx = max(0, min(int(len(data) * pct / 100.0), len(data) - 1))
                return round(data[idx], 2)
            
            return {
                'p50': get_percentile(latencies, 50),
                'p95': get_percentile(latencies, 95),
                'p99': get_percentile(latencies, 99),
                'p999': get_percentile(latencies, 99.9)
            }
        except Exception as e:
            print(f"Error fetching latency distribution: {e}", file=sys.stderr)
            traceback.print_exc()
            return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
        finally:
            conn.close()


dashboard_data = DashboardData()


@app.route('/')
def index():
    """Render main dashboard page."""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template error: {e}", file=sys.stderr)
        traceback.print_exc()
        # Return a basic HTML page if template fails
        return f"""
        <html>
        <head><title>Dashboard Error</title></head>
        <body>
            <h1>Template Error</h1>
            <p>Could not load dashboard template.</p>
            <p>Error: {str(e)}</p>
            <pre>{traceback.format_exc()}</pre>
            <p>Template folder: {app.template_folder}</p>
            <p>Expected: dashboard/templates/index.html</p>
        </body>
        </html>
        """, 500


@app.route('/api/metrics')
def api_metrics():
    """API endpoint for recent metrics."""
    try:
        metrics = dashboard_data.get_recent_metrics(limit=200)
        return jsonify(metrics)
    except Exception as e:
        print(f"Error in /api/metrics: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/phase-summary')
def api_phase_summary():
    """API endpoint for phase summary."""
    try:
        summary = dashboard_data.get_phase_summary()
        return jsonify(summary)
    except Exception as e:
        print(f"Error in /api/phase-summary: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning-stats')
def api_learning_stats():
    """API endpoint for learning statistics."""
    try:
        stats = dashboard_data.get_learning_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in /api/learning-stats: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/latency-distribution')
def api_latency_distribution():
    """API endpoint for latency distribution."""
    try:
        distribution = dashboard_data.get_latency_distribution()
        return jsonify(distribution)
    except Exception as e:
        print(f"Error in /api/latency-distribution: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    try:
        db_exists = dashboard_data.db_path.exists()
        
        # Get table info if DB exists
        tables = []
        row_counts = {}
        if db_exists:
            conn = dashboard_data.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Get row counts
                for table in tables:
                    if table != 'sqlite_sequence':
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        row_counts[table] = cursor.fetchone()[0]
                
                conn.close()
        
        return jsonify({
            'status': 'running' if db_exists else 'no_database',
            'timestamp': datetime.now().isoformat(),
            'db_exists': db_exists,
            'db_path': str(dashboard_data.db_path),
            'tables': tables,
            'row_counts': row_counts
        })
    except Exception as e:
        print(f"Error in /api/status: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Database Query Optimizer - Dashboard (FIXED)")
    print("="*70)
    
    # Check prerequisites
    data_dir = Path('data')
    if not data_dir.exists():
        print("Error: 'data' directory not found.")
        sys.exit(1)
    
    if not dashboard_data.db_path.exists():
        print(f"Warning: Database not found at {dashboard_data.db_path}")
    else:
        print(f"✓ Database found: {dashboard_data.db_path}")
    
    template_path = Path('dashboard/templates/index.html')
    if not template_path.exists():
        print(f"✗ Template missing: {template_path}")
        print("  Please check your project structure.")
    else:
        print(f"✓ Template found: {template_path}")
    
    print(f"\nDashboard URLs:")
    print(f"  - http://localhost:5000")
    print(f"  - http://127.0.0.1:5000")
    print("="*70)
    print("Starting server... Press CTRL+C to stop\n")
    
    try:
        app.run(
            debug=True,
            host='127.0.0.1',
            port=5000,
            use_reloader=False
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
