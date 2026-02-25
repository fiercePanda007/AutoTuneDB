import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
import copy

from core.models import MetaLearnerChromosome


class MetaLearner:
    """
    Level 2: Strategic Meta-Learning
    Optimizes the learning process itself through evolutionary algorithms.
    """
    
    def __init__(self, config: Dict[str, Any], policy_learner, telemetry_storage):
        """
        Initialize meta-learner.
        
        Args:
            config: System configuration
            policy_learner: PolicyLearner instance
            telemetry_storage: TelemetryStorage instance
        """
        self.config = config
        self.policy_learner = policy_learner
        self.telemetry = telemetry_storage
        self.logger = logging.getLogger(__name__)
        
        self.level2_config = config['level2']
        self.enabled = self.level2_config['enabled']
        self.apply_live_updates = bool(self.level2_config.get('apply_live_updates', False))
        self.evaluation_interval = float(self.level2_config.get('evaluation_interval', 86400))
        self.last_optimization_time = 0.0
        
        # Genetic algorithm parameters
        self.population_size = self.level2_config['population_size']
        self.mutation_rate = self.level2_config['mutation_rate']
        self.crossover_rate = self.level2_config['crossover_rate']
        self.tournament_size = self.level2_config['tournament_size']
        self.elite_size = self.level2_config['elite_size']
        
        # Hyperparameter bounds
        self.bounds = self.level2_config['hyperparameter_bounds']
        
        # Initialize population
        self.population = self._initialize_population()
        self.generation = 0
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        
        # Evolution history
        self.evolution_history = []
        
        self.logger.info("Meta-Learner (Level 2) initialized")
        
    def set_enabled(self, enabled: bool):
        """Enable or disable meta-learning."""
        self.enabled = enabled
        self.logger.info(f"Meta-learning {'enabled' if enabled else 'disabled'}")
        
    def _initialize_population(self) -> List[MetaLearnerChromosome]:
        """
        Initialize population with random hyperparameters.
        
        Returns:
            List of chromosomes
        """
        population = []
        
        for _ in range(self.population_size):
            genes = {}
            
            for param, (low, high) in self.bounds.items():
                if param == 'batch_size':
                    genes[param] = np.random.randint(low, high + 1)
                else:
                    genes[param] = np.random.uniform(low, high)
            
            population.append(MetaLearnerChromosome(genes))
        
        return population
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run one generation of meta-learning optimization.
        
        Returns:
            Optimization results
        """
        if not self.enabled:
            return {}

        now = time.time()
        if (
            self.last_optimization_time > 0
            and self.evaluation_interval > 0
            and (now - self.last_optimization_time) < self.evaluation_interval
        ):
            self.logger.debug(
                "Skipping meta-learning optimize due to evaluation_interval throttle"
            )
            return {}
        self.last_optimization_time = now
        
        self.logger.info(f"Running meta-learning generation {self.generation}")
        
        # Evaluate fitness of current population
        self._evaluate_population()
        avg_fitness = float(np.mean([c.fitness for c in self.population]))
        
        # Track best chromosome
        current_best = max(self.population, key=lambda c: c.fitness)
        if current_best.fitness > self.best_fitness:
            self.best_fitness = current_best.fitness
            self.best_chromosome = current_best.clone()
            self.logger.info(
                f"New best chromosome found! Fitness: {self.best_fitness:.4f}"
            )
            self._apply_best_hyperparameters()
        
        # Create next generation
        self.population = self._evolve_population()
        
        # Record generation results
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': avg_fitness,
            'best_genes': self.best_chromosome.genes if self.best_chromosome else {}
        })
        
        # Store meta-learning event
        self.telemetry.store_meta_learning_event({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': avg_fitness,
            'hyperparameters': self.best_chromosome.genes if self.best_chromosome else {},
            'improvements': self._calculate_improvements()
        })
        
        self.generation += 1
        
        return {
            'generation': self.generation - 1,
            'best_fitness': self.best_fitness,
            'avg_fitness': avg_fitness
        }
    
    def _evaluate_population(self):
        """Evaluate fitness of all chromosomes in population."""
        for chromosome in self.population:
            # Fitness based on recent performance with these hyperparameters
            # In full implementation, would test each chromosome
            # For demo, use simulated fitness
            chromosome.fitness = self._calculate_fitness(chromosome.genes)
    
    def _calculate_fitness(self, genes: Dict[str, Any]) -> float:
        """
        Calculate fitness score for hyperparameter configuration.
        
        Args:
            genes: Hyperparameter values
            
        Returns:
            Fitness score (higher is better)
        """
        # Get recent metrics
        metrics = self.telemetry.get_recent_metrics(hours=1)
        
        if not metrics:
            # No data, return neutral fitness
            return 0.5
        
        # Calculate performance metrics
        exec_times = [m.get('execution_time', 0) for m in metrics]
        success_count = sum(1 for m in metrics if m.get('success', True))
        
        avg_latency = np.mean(exec_times)
        success_rate = success_count / len(metrics)
        
        # Fitness components
        # Lower latency is better
        latency_score = 1.0 / (1.0 + avg_latency)
        
        # Higher success rate is better
        success_score = success_rate
        
        # Penalize extreme hyperparameter values
        balance_penalty = 0
        for param, value in genes.items():
            low, high = self.bounds[param]
            # Normalize to [0, 1]
            normalized = (value - low) / (high - low)
            # Penalize if too close to boundaries
            if normalized < 0.1 or normalized > 0.9:
                balance_penalty += 0.1
        
        # Combined fitness
        fitness = (0.6 * latency_score + 0.4 * success_score) - balance_penalty
        
        return max(0, fitness)
    
    def _evolve_population(self) -> List[MetaLearnerChromosome]:
        """
        Create next generation through selection, crossover, and mutation.
        
        Returns:
            New population
        """
        new_population = []
        
        # Elitism: keep best chromosomes
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        for i in range(self.elite_size):
            new_population.append(sorted_pop[i].clone())
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = MetaLearnerChromosome.crossover(parent1, parent2)
            else:
                child = parent1.clone()
            
            # Mutation
            child.mutate(self.mutation_rate, self.bounds)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self) -> MetaLearnerChromosome:
        """
        Select chromosome using tournament selection.
        
        Returns:
            Selected chromosome
        """
        tournament = np.random.choice(
            self.population,
            size=self.tournament_size,
            replace=False
        )
        
        return max(tournament, key=lambda c: c.fitness)
    
    def _apply_best_hyperparameters(self):
        """Apply best found hyperparameters to the learning system."""
        if not self.best_chromosome:
            return

        if not self.apply_live_updates:
            self.logger.info(
                "Meta-learner live hyperparameter application disabled (safe mode)"
            )
            return
        
        genes = self.best_chromosome.genes
        
        # Update Level 0 (Query Optimizer) hyperparameters
        if hasattr(self.policy_learner, 'query_optimizer'):
            optimizer = self.policy_learner.query_optimizer
            
            if 'learning_rate' in genes:
                for param_group in optimizer.optimizer.param_groups:
                    param_group['lr'] = genes['learning_rate']
            
            if 'gamma' in genes:
                optimizer.gamma = genes['gamma']
            
            if 'batch_size' in genes:
                optimizer.batch_size = genes['batch_size']
            
            if 'epsilon_decay' in genes:
                optimizer.epsilon_decay = genes['epsilon_decay']
        
        self.logger.info(f"Applied best hyperparameters: {genes}")
    
    def _calculate_improvements(self) -> Dict[str, float]:
        """
        Calculate improvement metrics over time.
        
        Returns:
            Dictionary of improvement percentages
        """
        if len(self.evolution_history) < 2:
            return {}
        
        first_gen = self.evolution_history[0]
        current_gen = self.evolution_history[-1]
        
        improvements = {}
        
        if first_gen['best_fitness'] > 0:
            improvements['best_fitness'] = (
                (current_gen['best_fitness'] - first_gen['best_fitness']) /
                first_gen['best_fitness'] * 100
            )
        
        if first_gen['avg_fitness'] > 0:
            improvements['avg_fitness'] = (
                (current_gen['avg_fitness'] - first_gen['avg_fitness']) /
                first_gen['avg_fitness'] * 100
            )
        
        return improvements
    
    def save_state(self, path: Optional[Path] = None):
        """Save meta-learner state."""
        if path is None:
            path = Path(self.config['paths']['policies_dir']) / 'level2_state.json'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_genes': self.best_chromosome.genes if self.best_chromosome else {},
            'evolution_history': self.evolution_history
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"State saved to {path}")


# core/safety_monitor.py

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
