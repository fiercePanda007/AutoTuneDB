"""
Main orchestrator for the Self-Improving Database Query Optimizer.

This module coordinates all system components across the three-tier
learning architecture.

FIXED: Added phase tracking for policy learner to enable phase-specific
performance comparison.
"""

import os
import sys
import time
import signal
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Add project root to path with absolute path resolution
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from core.query_optimizer import QueryOptimizer
from core.policy_learner import PolicyLearner
from core.meta_learner import MetaLearner
from core.safety_monitor import SafetyMonitor
from database.database_manager_duck import DatabaseManager
from database.workload_generator_duck import WorkloadGenerator
from telemetry.collector import TelemetryCollector
from telemetry.storage import TelemetryStorage
from utils.logger import setup_logger, get_logger
from utils.metrics import MetricsCalculator


class SystemOrchestrator:
    """Main orchestrator coordinating all system components."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the system orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        # Use Path object for config path
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config(str(self.config_path))
        self.running = False
        self.paused = False
        
        # Setup logging
        setup_logger(self.config)
        self.logger = get_logger(__name__)
        
        # Initialize components (will be populated in initialize_components)
        self.db_manager = None
        self.workload_generator = None
        self.query_optimizer = None
        self.policy_learner = None
        self.meta_learner = None
        self.safety_monitor = None
        self.telemetry_collector = None
        self.telemetry_storage = None
        self.metrics_calculator = None
        
        # State tracking
        self.start_time = None
        self.current_phase = "initialization"
        self.stats = {
            "queries_executed": 0,
            "policies_updated": 0,
            "meta_learner_runs": 0,
            "safety_events": 0
        }

        # Level 2 guardrails (do not alter Level 0 implementation details)
        level1_cfg = self.config.get('level1', {})
        level2_cfg = self.config.get('level2', {})
        self.policy_update_every = int(level1_cfg.get('update_every_iterations', 100))
        self.meta_update_every = int(level2_cfg.get('update_every_iterations', 1000))
        self.level2_freeze_policy_updates = bool(
            level2_cfg.get('freeze_policy_updates', True)
        )
        self.level2_freeze_weight_updates = bool(
            level2_cfg.get('freeze_level0_weight_updates', True)
        )
        self.level2_freeze_batch_size = int(
            level2_cfg.get('freeze_batch_size', 1_000_000)
        )
        self.level2_epsilon_floor = float(level2_cfg.get('epsilon_floor', 0.2))
        self.level2_seed_preferences_from_level1 = bool(
            level2_cfg.get('seed_preferences_from_level1', True)
        )
        self.level2_seed_min_action_samples = int(
            level2_cfg.get('seed_min_action_samples', 15)
        )
        self.level2_regression_guard_enabled = bool(
            level2_cfg.get('regression_guard_enabled', True)
        )
        self.level2_regression_threshold = float(
            level2_cfg.get('rollback_if_worse_pct', 0.05)
        )
        self.level2_guard_window_samples = int(
            level2_cfg.get('guard_window_samples', 200)
        )
        self.level2_guard_check_every = int(
            level2_cfg.get('guard_check_every_iterations', 100)
        )
        self.level2_guard_triggered = False
        self.level2_reference_avg = None
        self.level2_saved_batch_size = None
        self.phase_action_profiles = self.config.get('phase_action_profiles', {})
        
        self.logger.info("System Orchestrator initialized")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with proper encoding."""
        try:
            # Explicit UTF-8 encoding for Windows compatibility
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Convert all path strings to resolved absolute paths
            if 'paths' in config:
                for key, value in config['paths'].items():
                    if isinstance(value, str):
                        # Resolve relative paths to absolute paths
                        config['paths'][key] = str(Path(value).resolve())
            
            return config
            
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {config_path}")
            print("Please copy config.yaml.example to config.yaml and configure it.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error loading config: {e}")
            sys.exit(1)
            
    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing system components...")
            
            # Create necessary directories
            self._create_directories()
            
            # Initialize database manager
            self.logger.info("Initializing database manager...")
            self.db_manager = DatabaseManager(self.config)
            self.db_manager.connect()
            
            # Verify database connection
            if not self._verify_database_connection():
                raise Exception("Failed to connect to database")
            
            # Initialize telemetry storage
            self.logger.info("Initializing telemetry storage...")
            self.telemetry_storage = TelemetryStorage(self.config)
            
            # Initialize telemetry collector
            self.logger.info("Initializing telemetry collector...")
            self.telemetry_collector = TelemetryCollector(
                self.config,
                self.telemetry_storage
            )
            
            # Initialize workload generator
            self.logger.info("Initializing workload generator...")
            self.workload_generator = WorkloadGenerator(
                self.config,
                self.db_manager
            )
            
            # Initialize query optimizer (Level 0)
            self.logger.info("Initializing query optimizer (Level 0)...")
            self.query_optimizer = QueryOptimizer(
                self.config,
                self.db_manager,
                self.telemetry_collector
            )
            
            # Initialize policy learner (Level 1)
            if self.config['level1']['enabled']:
                self.logger.info("Initializing policy learner (Level 1)...")
                self.policy_learner = PolicyLearner(
                    self.config,
                    self.query_optimizer,
                    self.telemetry_storage
                )
            
            # Initialize meta-learner (Level 2)
            if self.config['level2']['enabled']:
                self.logger.info("Initializing meta-learner (Level 2)...")
                self.meta_learner = MetaLearner(
                    self.config,
                    self.policy_learner,
                    self.telemetry_storage
                )
            
            # Initialize safety monitor
            if self.config['safety']['enabled']:
                self.logger.info("Initializing safety monitor...")
                self.safety_monitor = SafetyMonitor(
                    self.config,
                    self.telemetry_storage
                )
            
            # Initialize metrics calculator
            self.logger.info("Initializing metrics calculator...")
            self.metrics_calculator = MetricsCalculator(self.config)
            
            # Register signal handlers with proper Windows handling
            if sys.platform == 'win32':
                # Windows doesn't support SIGTERM the same way
                signal.signal(signal.SIGINT, self._signal_handler)
            else:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", exc_info=True)
            raise
    
    def _verify_database_connection(self) -> bool:
        """Verify database connection is working."""
        try:
            version = self.db_manager.get_version()
            self.logger.info(f"Connected to DuckDB: {version}")
            self.logger.info("Database connection verified successfully")
            return True
        except Exception as e:
            self.logger.error(f"Database connection verification failed: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        paths = self.config['paths']
        for key, path_str in paths.items():
            if key.endswith('_dir'):
                path = Path(path_str)
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created/verified directory: {path}")
    
    def start(self, duration_days: float = 14.0, fast_mode: bool = False):
        """
        Start the system with proper phase transitions.
        
        Args:
            duration_days: Duration in days (can be fractional)
            fast_mode: Accelerate time for testing
        """
        try:
            self.running = True
            self.start_time = datetime.now()
            
            self.logger.info("="*70)
            self.logger.info(f"Starting system - Duration: {duration_days} days")
            self.logger.info(f"Fast mode: {fast_mode}")
            self.logger.info("="*70)
            
            # Calculate time scaling and phase durations
            time_scale = 100 if fast_mode else 1
            total_seconds = duration_days * 24 * 3600 / time_scale
            
            # Divide time equally among 4 phases
            phase_duration = total_seconds / 4
            
            # Define phases with their learning configurations
            phases = [
                ('baseline', phase_duration, False, False, False),
                ('level0_learning', phase_duration, True, False, False),
                ('level1_learning', phase_duration, True, True, False),
                ('level2_learning', phase_duration, True, True, True),
            ]
            
            self.logger.info(f"Phase duration: {phase_duration:.1f} seconds each")
            
            # Execute each phase in sequence
            for phase_name, duration, enable_l0, enable_l1, enable_l2 in phases:
                if not self.running:
                    break
                
                # Set phase in orchestrator
                self.current_phase = phase_name
                
                # CRITICAL: Set phase in telemetry collector
                if self.telemetry_collector:
                    self.telemetry_collector.set_phase(phase_name)
                
                # CRITICAL FIX: Set phase in policy learner for phase-specific comparison
                if self.policy_learner:
                    self.policy_learner.set_current_phase(phase_name)
                    self.logger.info(f"[OK] Policy learner tracking phase: '{phase_name}'")
                
                self.logger.info(f"="*50)
                self.logger.info(f"Starting phase: {phase_name}")
                self.logger.info(f"Learning levels - L0: {enable_l0}, L1: {enable_l1}, L2: {enable_l2}")
                self.logger.info(f"Duration: {duration:.1f} seconds")
                
                # Configure learning levels
                if self.query_optimizer:
                    self.query_optimizer.set_learning_enabled(enable_l0)
                    phase_overrides = self._get_phase_action_overrides(phase_name)
                    self.query_optimizer.set_query_type_action_overrides(phase_overrides)
                    
                if self.policy_learner:
                    self.policy_learner.set_enabled(enable_l1)
                    if phase_name == 'level2_learning' and self.level2_freeze_policy_updates:
                        self.policy_learner.set_enabled(False)
                        self.logger.info(
                            "Level2 policy updates frozen (using policy learned in Level1)"
                        )
                    
                if self.meta_learner:
                    self.meta_learner.set_enabled(enable_l2)

                if phase_name == 'level2_learning':
                    self.level2_guard_triggered = False
                    self.level2_reference_avg = self._get_phase_avg_latency(
                        'level1_learning', self.level2_guard_window_samples
                    )
                    self.level2_saved_batch_size = None
                    if self.query_optimizer and self.level2_freeze_weight_updates:
                        self.level2_saved_batch_size = int(self.query_optimizer.batch_size)
                        self.query_optimizer.batch_size = max(
                            self.level2_saved_batch_size,
                            self.level2_freeze_batch_size
                        )
                        self.logger.info(
                            "Level2 weight updates frozen "
                            f"(batch_size={self.query_optimizer.batch_size})"
                        )
                    if (
                        self.query_optimizer
                        and hasattr(self.query_optimizer, 'epsilon')
                        and self.level2_epsilon_floor > 0
                    ):
                        self.query_optimizer.epsilon = max(
                            float(self.query_optimizer.epsilon),
                            self.level2_epsilon_floor
                        )
                        self.logger.info(
                            f"Level2 epsilon floor applied at {self.level2_epsilon_floor:.3f}"
                        )
                    if (
                        self.query_optimizer
                        and self.level2_seed_preferences_from_level1
                        and not self._get_phase_action_overrides('level2_learning')
                    ):
                        seeded = self._build_level2_seed_preferences()
                        if seeded:
                            self.query_optimizer.set_action_preferences(seeded)
                            self.logger.info(
                                f"Level2 seeded action preferences from Level1: {seeded}"
                            )
                    if self.level2_reference_avg is not None:
                        self.logger.info(
                            f"Level2 guard reference latency: "
                            f"{self.level2_reference_avg * 1000:.2f}ms"
                        )
                    # Observability bootstrap: run one meta optimization cycle early in
                    # Level2 safe mode so short phases still emit meta-learning telemetry.
                    if (
                        self.meta_learner
                        and self.meta_learner.enabled
                        and not getattr(self.meta_learner, 'apply_live_updates', True)
                    ):
                        try:
                            self.meta_learner.optimize()
                            self.stats['meta_learner_runs'] += 1
                        except Exception as e:
                            self.logger.error(f"Meta-learning bootstrap failed: {e}")
                
                # Run phase
                phase_start = time.time()
                phase_end = phase_start + duration
                iteration = 0
                
                while self.running and time.time() < phase_end:
                    try:
                        if not self.paused:
                            self._execute_iteration(iteration)
                            iteration += 1
                        
                        # Sleep briefly to prevent CPU spinning
                        time.sleep(0.01 if fast_mode else 0.1)
                        
                    except KeyboardInterrupt:
                        self.logger.info("Received interrupt signal, shutting down...")
                        self.running = False
                        break
                    except Exception as e:
                        self.logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
                        if self.telemetry_collector:
                            self.telemetry_collector.record_safety_event({
                                'severity': 'error',
                                'event_type': 'iteration_failure',
                                'description': str(e)
                            })
                
                self.logger.info(f"Phase '{phase_name}' completed: {iteration} iterations")
                if (
                    phase_name == 'level2_learning'
                    and self.query_optimizer
                    and self.level2_saved_batch_size is not None
                ):
                    self.query_optimizer.batch_size = self.level2_saved_batch_size
                    self.level2_saved_batch_size = None
            
            # Shutdown
            self.shutdown()
            
            self.logger.info("="*70)
            self.logger.info("System execution completed")
            self.logger.info(f"Total queries: {self.stats['queries_executed']}")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error(f"Fatal error in system execution: {e}", exc_info=True)
            self.shutdown()
            raise

    def _execute_iteration(self, iteration: int):
        """Execute a single iteration of the system."""
        # Query execution with optimizer
        if self.workload_generator and self.query_optimizer:
            query, query_type = self.workload_generator.generate_query()
            
            try:
                state = {
                    'timestamp': time.time(),
                    'phase': self.current_phase,
                    'iteration': iteration
                }
                result = self.query_optimizer.execute_query(
                    query, query_type, state
                )
                if (
                    self.current_phase == 'level2_learning'
                    and hasattr(self.query_optimizer, 'epsilon')
                    and self.level2_epsilon_floor > 0
                ):
                    self.query_optimizer.epsilon = max(
                        float(self.query_optimizer.epsilon),
                        self.level2_epsilon_floor
                    )
                self.stats['queries_executed'] += 1
                
                # Collect telemetry
                if self.telemetry_collector:
                    plan_info = result.get('plan_info') if isinstance(result, dict) else None
                    if not isinstance(plan_info, dict):
                        plan_info = {}

                    # Observability-only diagnostics for dashboarding.
                    loss = result.get('loss') if isinstance(result, dict) else None
                    if loss is not None:
                        plan_info['training_loss'] = float(loss)

                    epsilon = result.get('epsilon') if isinstance(result, dict) else None
                    if epsilon is not None:
                        plan_info['epsilon'] = float(epsilon)

                    reward = result.get('reward') if isinstance(result, dict) else None
                    if reward is not None:
                        plan_info['reward'] = float(reward)

                    self.telemetry_collector.record_execution(
                        query=query,
                        query_type=query_type,
                        execution_time=result.get('execution_time', 0),
                        resources=result.get('resources', {}),
                        plan_info=plan_info,
                        success=result.get('success', True),
                        action=result["action"], 
                    )
                    
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                if self.telemetry_collector:
                    self.telemetry_collector.record_safety_event({
                        'severity': 'error',
                        'event_type': 'query_execution_failure',
                        'description': str(e)
                    })
        
        # Policy updates (Level 1)
        if (
            self.policy_learner
            and self.current_phase in {'level1_learning', 'level2_learning'}
            and self.policy_update_every > 0
            and iteration > 0
            and iteration % self.policy_update_every == 0
        ):
            try:
                if self.policy_learner.update_policy():
                    self.stats['policies_updated'] += 1
            except Exception as e:
                self.logger.error(f"Policy update failed: {e}")
        
        # Meta-learning (Level 2)
        if (
            self.current_phase == 'level2_learning'
            and self.meta_learner
            and self.meta_update_every > 0
            and iteration > 0
            and iteration % self.meta_update_every == 0
        ):
            try:
                self.meta_learner.optimize()
                self.stats['meta_learner_runs'] += 1
            except Exception as e:
                self.logger.error(f"Meta-learning failed: {e}")

        if self.current_phase == 'level2_learning':
            self._check_level2_regression_guard(iteration)

    def _get_phase_action_overrides(self, phase_name: str) -> Optional[Dict[str, str]]:
        """
        Return configured hard action overrides for a phase.

        Example config:
        phase_action_profiles:
          baseline: {__default__: threads_1}
          level0_learning: {__default__: default}
        """
        raw = self.phase_action_profiles.get(phase_name)
        if not raw or not isinstance(raw, dict):
            return None
        return {str(k): str(v) for k, v in raw.items() if v is not None}

    def _get_phase_avg_latency(self, phase: str, window: Optional[int] = None) -> Optional[float]:
        """Return average execution latency for a phase, optionally over a trailing window."""
        if not self.telemetry_storage:
            return None

        metrics = self.telemetry_storage.get_phase_metrics(phase)
        if not metrics:
            return None

        if window and window > 0:
            metrics = metrics[-window:]

        times = [
            float(m.get('execution_time', 0.0))
            for m in metrics
            if m.get('execution_time') is not None
        ]
        if not times:
            return None
        return sum(times) / len(times)

    def _check_level2_regression_guard(self, iteration: int):
        """Rollback/disarm Level 2 if it regresses beyond configured tolerance."""
        if not self.level2_regression_guard_enabled or self.level2_guard_triggered:
            return
        if iteration <= 0 or self.level2_guard_check_every <= 0:
            return
        if iteration % self.level2_guard_check_every != 0:
            return

        if self.level2_reference_avg is None:
            self.level2_reference_avg = self._get_phase_avg_latency(
                'level1_learning', self.level2_guard_window_samples
            )
            if self.level2_reference_avg is None:
                return

        level2_avg = self._get_phase_avg_latency(
            'level2_learning', self.level2_guard_window_samples
        )
        if level2_avg is None:
            return

        allowed = self.level2_reference_avg * (1.0 + self.level2_regression_threshold)
        if level2_avg <= allowed:
            return

        description = (
            "Level2 regression guard triggered: "
            f"level2_avg={level2_avg * 1000:.2f}ms, "
            f"level1_ref={self.level2_reference_avg * 1000:.2f}ms, "
            f"threshold={self.level2_regression_threshold * 100:.1f}%"
        )
        self.logger.warning(description)

        if self.policy_learner and self.policy_learner.enabled:
            self.policy_learner.rollback_policy()
            self.policy_learner.set_enabled(False)

        if self.meta_learner and self.meta_learner.enabled:
            self.meta_learner.set_enabled(False)

        if self.telemetry_storage:
            self.telemetry_storage.store_safety_event({
                'severity': 'warning',
                'event_type': 'level2_regression_guard',
                'description': description,
                'action_taken': 'disabled_level2_components'
            })

        self.level2_guard_triggered = True

    def _build_level2_seed_preferences(self) -> Optional[Dict[str, float]]:
        """
        Build action preference weights from Level1 telemetry.

        Uses the fastest sufficiently-sampled actions from Level1 and converts
        them to normalized inverse-latency weights.
        """
        if not self.telemetry_storage or not self.query_optimizer:
            return None

        metrics = self.telemetry_storage.get_phase_metrics('level1_learning')
        if not metrics:
            return None

        action_to_times: Dict[str, list] = {}
        known_actions = set(getattr(self.query_optimizer, 'action_name_to_idx', {}).keys())

        for metric in metrics:
            plan_info = metric.get('plan_info', {}) or {}
            if not isinstance(plan_info, dict):
                continue
            action = plan_info.get('action')
            execution_time = metric.get('execution_time')
            if (
                not action
                or action not in known_actions
                or execution_time is None
                or execution_time <= 0
            ):
                continue
            action_to_times.setdefault(action, []).append(float(execution_time))

        if not action_to_times:
            return None

        action_avg = {}
        for action, times in action_to_times.items():
            if len(times) < self.level2_seed_min_action_samples:
                continue
            action_avg[action] = sum(times) / len(times)

        # Fallback for sparse telemetry.
        if not action_avg:
            for action, times in action_to_times.items():
                if len(times) >= 5:
                    action_avg[action] = sum(times) / len(times)

        if not action_avg:
            return None

        fastest = sorted(action_avg.items(), key=lambda x: x[1])[:3]
        raw_weights = {action: 1.0 / (avg_time + 0.001) for action, avg_time in fastest}
        total = sum(raw_weights.values())
        if total <= 0:
            return None
        return {k: v / total for k, v in raw_weights.items()}
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    def shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("Initiating system shutdown...")
        
        # Flush any buffered telemetry
        if self.telemetry_collector:
            try:
                self.telemetry_collector.flush()
            except Exception as e:
                self.logger.error(f"Error flushing telemetry: {e}")
        
        # Close database connections
        if self.db_manager:
            try:
                self.db_manager.disconnect()
                self.logger.info("Database connections closed")
            except Exception as e:
                self.logger.error(f"Error disconnecting database: {e}")
        
        self.logger.info("System shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Self-Improving Database Query Optimizer'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=14.0,
        help='Simulation duration in days (default: 14.0)'
    )
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='Run in fast mode (100x speed) for testing'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run system
        orchestrator = SystemOrchestrator(args.config)
        orchestrator.initialize_components()
        orchestrator.start(duration_days=args.duration, fast_mode=args.fast_mode)
        
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
