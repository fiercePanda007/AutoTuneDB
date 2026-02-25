import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import time
import duckdb
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import SystemOrchestrator
from utils.logger import setup_logger, get_logger


class DemoRunner:
    """Manages the demonstration execution."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize demo runner."""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        setup_logger(self.config)
        self.logger = get_logger(__name__)
        
    def _load_config(self, config_path: str):
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file not found: {config_path}")
            print("Please run setup_duckdb.py first")
            sys.exit(1)
            
    def print_banner(self):
        """Print demo banner."""
        print("\n" + "="*70)
        print("  Self-Improving Database Query Optimizer - Demonstration")
        print("="*70)
        print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
    def verify_prerequisites(self):
        """Verify system is ready to run."""
        print("Verifying prerequisites...")
        
        # Check database connection
        try:
            import duckdb
            duck_cfg = self.config["duckdb"]
            conn = duckdb.connect(duck_cfg.get("path", "demo.duckdb"), read_only=True)
            conn.execute("SELECT version();").fetchone()
            conn.close()
            print("  [OK] Database connection successful")
        except Exception as e:
            print(f"  [Fail] Database connection failed: {e}")
            print("\nPlease run: python setup_duckdb.py")
            return False
            
        # Check data directories
        data_dir = Path(self.config['paths']['data_dir'])
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            print("  [OK] Data directory created")
        else:
            print("  [OK] Data directory exists")
            
        # Check telemetry database
        telemetry_path = Path(self.config['paths']['telemetry_db'])
        if not telemetry_path.exists():
            print("  [Fail] Telemetry database not found")
            print("\nPlease run: python setup_duckdb.py")
            return False
        else:
            print("  [OK] Telemetry database exists")
            
        print("\nAll prerequisites verified!\n")
        return True
        
    def run_demo(
        self,
        duration_days: float = 14.0,
        fast_mode: bool = False,
        queries_per_hour: int = None,
        workload_type: str = 'mixed'
    ):
        """
        Run the complete demonstration.
        
        Args:
            duration_days: Duration in days (can be fractional)
            fast_mode: Accelerate time for testing
            queries_per_hour: Override query rate
            workload_type: Type of workload (mixed, analytical, transactional)
        """
        self.print_banner()
        
        if not self.verify_prerequisites():
            return
            
        # Apply demo-specific configuration
        if queries_per_hour:
            self.config['workload']['queries_per_hour'] = queries_per_hour
            
        # Display configuration
        self._display_configuration(duration_days, fast_mode, workload_type)
        
        # Create orchestrator and run
        try:
            print("\nInitializing system components...")
            orchestrator = SystemOrchestrator(self.config_path)
            orchestrator.initialize_components()
            
            print("\nStarting demonstration...")
            print("You can monitor progress at: http://localhost:5000")
            print("(Run 'python dashboard.py' in another terminal)")
            print("\nPress Ctrl+C to stop\n")
            
            time.sleep(2)  # Give user time to read
            
            orchestrator.start(
                duration_days=duration_days,
                fast_mode=fast_mode
            )
            
            self._print_completion_message()
            
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
        except Exception as e:
            print(f"\n\nError during demo: {e}")
            self.logger.error("Demo failed", exc_info=True)
            raise
            
    def _display_configuration(
        self,
        duration_days: float,
        fast_mode: bool,
        workload_type: str
    ):
        """Display demo configuration."""
        print("Demo Configuration:")
        print(f"  Duration: {duration_days} days")
        if fast_mode:
            actual_time = duration_days * 24 * 60 / 100  # Rough estimate
            print(f"  Fast Mode: Enabled (~{actual_time:.1f} minutes)")
        print(f"  Workload: {workload_type}")
        print(f"  Queries/Hour: {self.config['workload']['queries_per_hour']}")
        print(f"  Learning Levels: L0={self.config['level0']['learning_rate']}, "
              f"L1={self.config['level1']['enabled']}, "
              f"L2={self.config['level2']['enabled']}")
        print()
        
    def _print_completion_message(self):
        """Print completion message."""
        print("\n" + "="*70)
        print("  Demonstration Complete!")
        print("="*70)
        print("\nResults:")
        print("  - Check data/logs/optimizer.log for detailed logs")
        print("  - Check data/final_report.txt for performance analysis")
        print("  - Query telemetry database for detailed metrics")
        print("\nTo view results:")
        print("  python -c \"from telemetry.storage import TelemetryStorage; "
              "ts = TelemetryStorage({}); ts.print_summary()\"")
        print("\nTo start dashboard:")
        print("  python dashboard.py")
        print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Self-Improving Database Query Optimizer Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 2-week demo
  python run_demo_duckdb.py --duration 14
  
  # Quick test (1 hour simulated)
  python run_demo_duckdb.py --duration 0.04 --fast-mode
  
  # Custom configuration
  python run_demo_duckdb.py --duration 7 --queries 500 --workload analytical
        """
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=14.0,
        help="Duration to run in days (can be fractional, e.g., 0.04 = ~1 hour)"
    )
    
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run in fast mode for testing (100x time acceleration)"
    )
    
    parser.add_argument(
        "--queries",
        type=int,
        help="Override queries per hour from config"
    )
    
    parser.add_argument(
        "--workload",
        choices=['mixed', 'analytical', 'transactional'],
        default='mixed',
        help="Type of workload to generate"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Create demo runner and execute
    demo = DemoRunner(args.config)
    demo.run_demo(
        duration_days=args.duration,
        fast_mode=args.fast_mode,
        queries_per_hour=args.queries,
        workload_type=args.workload
    )


if __name__ == "__main__":
    main()
