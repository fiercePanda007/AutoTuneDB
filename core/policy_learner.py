"""
Level 1: Tactical Policy Learning

This module implements the tactical layer that analyzes execution telemetry
and updates operational policies to improve query optimization.

FIXED: The _apply_policy method now actually modifies the query optimizer's
action selection behavior instead of just logging changes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time

from core.models import PolicyNetwork


class PolicyLearner:
    """
    Level 1: Tactical Policy Learning
    Analyzes execution telemetry and updates operational policies.
    
    FIXED: Now properly applies policy changes to the query optimizer,
    creating a real feedback loop between Level 1 and Level 0.
    """
    
    def __init__(self, config: Dict[str, Any], query_optimizer, telemetry_storage):
        """
        Initialize policy learner.
        
        Args:
            config: System configuration
            query_optimizer: QueryOptimizer instance
            telemetry_storage: TelemetryStorage instance
        """
        self.config = config
        self.query_optimizer = query_optimizer
        self.telemetry = telemetry_storage
        self.logger = logging.getLogger(__name__)
        
        self.level1_config = config['level1']
        self.enabled = self.level1_config['enabled']
        
        # Policy version tracking
        self.policy_version = 0
        self.last_update = time.time()
        
        # Track current phase
        self.current_phase = None
        
        # Policy network for learning improved decision rules
        self.policy_network = PolicyNetwork(
            input_dim=64,  # Telemetry features
            output_dim=32,  # Policy parameters
            hidden_dims=[256, 128]
        )
        
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.level1_config['learning_rate']
        )
        
        # Performance history
        self.performance_history = []
        
        self.logger.info("Policy Learner (Level 1) initialized")
        
    def set_enabled(self, enabled: bool):
        """Enable or disable policy learning."""
        self.enabled = enabled
        self.logger.info(f"Policy learning {'enabled' if enabled else 'disabled'}")
    
    def set_current_phase(self, phase: str):
        """
        Set the current phase for phase-specific learning.
        
        Args:
            phase: Name of current phase
        """
        self.current_phase = phase
        self.logger.debug(f"Policy learner tracking phase: {phase}")
        
    def update_policy(self) -> bool:
        """
        Analyze telemetry and update operational policy.
        
        Compares baseline to CURRENT PHASE ONLY, not all phases.
        
        Returns:
            True if policy was updated
        """
        if not self.enabled:
            return False
        
        self.logger.info("Analyzing telemetry for policy update...")
        
        # Get baseline metrics
        baseline_metrics = self.telemetry.get_phase_metrics('baseline')
        
        if not baseline_metrics:
            self.logger.info("No baseline metrics available")
            return False
        
        # Get metrics from CURRENT PHASE ONLY
        if self.current_phase and self.current_phase != 'baseline':
            current_metrics = self.telemetry.get_phase_metrics(self.current_phase)
            phase_name = self.current_phase
        else:
            # Fallback: get recent metrics from any non-baseline phase
            all_metrics = self.telemetry.get_recent_metrics(hours=1)
            current_metrics = [m for m in all_metrics if m.get('phase') != 'baseline']
            phase_name = 'current'
        
        if len(current_metrics) < self.level1_config['validation_samples']:
            self.logger.info(
                f"Insufficient samples in {phase_name} phase "
                f"({len(current_metrics)} < {self.level1_config['validation_samples']})"
            )
            return False
        
        # Analyze performance
        baseline_performance = self._analyze_performance(baseline_metrics)
        current_performance = self._analyze_performance(current_metrics)
        
        self.logger.info(
            f"Baseline: {baseline_performance['avg_latency']*1000:.2f}ms, "
            f"{phase_name}: {current_performance['avg_latency']*1000:.2f}ms"
        )
        
        # Check if improvement is possible
        if not self._should_update(baseline_performance, current_performance):
            self.logger.info("No significant improvement opportunity detected")
            return False
        
        # Generate new policy
        new_policy = self._generate_improved_policy(
            current_metrics, 
            baseline_performance,
            current_performance
        )
        
        # Validate new policy
        if self._validate_policy(new_policy, baseline_performance, current_performance):
            self._apply_policy(new_policy)
            
            # Update version and timestamp
            self.policy_version += 1
            self.last_update = time.time()
            
            # Record update
            self.telemetry.store_policy_update({
                'old_version': self.policy_version - 1,
                'new_version': self.policy_version,
                'improvement': new_policy['expected_improvement'],
                'validation_score': new_policy['validation_score'],
                'changes': new_policy['changes']
            })
            
            self.logger.info(
                f"Policy updated to version {self.policy_version} "
                f"(expected improvement: {new_policy['expected_improvement']:.2%})"
            )
            
            return True
        else:
            self.logger.info(
                f"New policy failed validation "
                f"(improvement: {new_policy['expected_improvement']:.2%}, "
                f"validation score: {new_policy['validation_score']:.2f})"
            )
            return False
    
    def _analyze_performance(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze current performance from metrics.
        
        Args:
            metrics: List of execution metrics
            
        Returns:
            Performance summary
        """
        if not metrics:
            return {
                'avg_latency': 0,
                'p95_latency': 0,
                'p99_latency': 0,
                'success_rate': 0,
                'total_samples': 0
            }
        
        exec_times = [m.get('execution_time', 0) for m in metrics]
        success_count = sum(1 for m in metrics if m.get('success', True))
        
        return {
            'avg_latency': np.mean(exec_times),
            'p95_latency': np.percentile(exec_times, 95),
            'p99_latency': np.percentile(exec_times, 99),
            'success_rate': success_count / len(metrics) if metrics else 0,
            'total_samples': len(metrics)
        }
    
    def _should_update(
        self, 
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float]
    ) -> bool:
        """
        Determine if policy update is warranted.
        
        Args:
            baseline_performance: Baseline performance metrics
            current_performance: Current phase performance metrics
            
        Returns:
            True if update should proceed
        """
        # Always allow update attempts (let validation decide)
        # This allows the system to learn from both improvements and degradations
        return True
    
    def _generate_improved_policy(
        self,
        metrics: List[Dict[str, Any]],
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate an improved policy based on telemetry analysis.
        
        Args:
            metrics: Execution metrics from current phase
            baseline_performance: Baseline performance summary
            current_performance: Current performance summary
            
        Returns:
            New policy specification
        """
        # Analyze which actions performed best
        action_performance = {}
        
        for metric in metrics:
            plan_info = metric.get('plan_info', {})
            if isinstance(plan_info, str):
                try:
                    plan_info = json.loads(plan_info)
                except:
                    plan_info = {}
            
            action = plan_info.get('action', 'default')
            exec_time = metric.get('execution_time', 0)
            
            if action not in action_performance:
                action_performance[action] = []
            action_performance[action].append(exec_time)
        
        # Calculate average performance per action
        action_scores = {}
        for action, times in action_performance.items():
            action_scores[action] = {
                'avg_time': np.mean(times),
                'count': len(times)
            }
        
        # Determine policy adjustments
        # Identify best-performing actions (lowest execution time)
        if action_scores:
            best_actions = sorted(
                action_scores.items(),
                key=lambda x: x[1]['avg_time']
            )[:3]
        else:
            best_actions = []
        
        policy_changes = {
            'prioritize_actions': [action for action, _ in best_actions],
            'action_scores': action_scores,
            'baseline_avg': baseline_performance['avg_latency'],
            'current_avg': current_performance['avg_latency']
        }
        
        # Calculate expected improvement
        baseline_avg = baseline_performance['avg_latency']
        current_avg = current_performance['avg_latency']
        
        if baseline_avg > 0:
            expected_improvement = (baseline_avg - current_avg) / baseline_avg
        else:
            expected_improvement = 0
        
        return {
            'changes': policy_changes,
            'expected_improvement': expected_improvement,
            'validation_score': 0.0  # To be filled by validation
        }
    
    def _validate_policy(
        self,
        new_policy: Dict[str, Any],
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float]
    ) -> bool:
        """
        Validate new policy before deployment.
        
        Args:
            new_policy: Proposed policy
            baseline_performance: Baseline performance
            current_performance: Current performance
            
        Returns:
            True if policy passes validation
        """
        expected_improvement = new_policy['expected_improvement']
        threshold = self.level1_config['min_improvement']
        
        # Check if improvement meets minimum threshold
        if expected_improvement < threshold:
            self.logger.debug(
                f"Improvement {expected_improvement:.2%} < threshold {threshold:.2%}"
            )
            return False
        
        # Calculate validation confidence
        # Higher improvement relative to threshold = higher confidence
        confidence = min(expected_improvement / threshold, 1.0)
        
        new_policy['validation_score'] = confidence
        
        # Check if confidence meets validation threshold
        validation_threshold = self.level1_config['validation_threshold']
        
        if confidence < validation_threshold:
            self.logger.debug(
                f"Confidence {confidence:.2f} < threshold {validation_threshold:.2f}"
            )
            return False
        
        return True
    
    def _apply_policy(self, policy: Dict[str, Any]):
        """
        Apply new policy to query optimizer by updating action preferences.
        
        
        Args:
            policy: Policy to apply with prioritized actions and scores
        """
        changes = policy['changes']
        prioritized = changes.get('prioritize_actions', [])
        action_scores = changes.get('action_scores', {})
        
        # Convert action scores to preference weights
        # Lower execution time = higher preference weight
        action_weights = {}
        for action_name, score_data in action_scores.items():
            avg_time = score_data['avg_time']
            count = score_data['count']
            
            # Only include actions with sufficient samples
            if count >= 5:
                # Inverse of time as weight: faster actions get higher weights
                # Add small epsilon to avoid division by zero
                action_weights[action_name] = 1.0 / (avg_time + 0.001) if avg_time > 0 else 1.0
        
        # Normalize weights so they sum to 1.0
        total_weight = sum(action_weights.values())
        if total_weight > 0:
            action_weights = {k: v/total_weight for k, v in action_weights.items()}
        else:
            # Fallback: uniform weights if no valid actions
            self.logger.warning("No valid action weights, using uniform distribution")
            action_weights = None
        
        # Apply preferences to the query optimizer
        if action_weights:
            self.query_optimizer.set_action_preferences(action_weights)
            
            self.logger.info(f"Applied policy: prioritizing {prioritized}")
            self.logger.info(f"Action weights: {action_weights}")
        else:
            self.logger.warning("Could not apply policy: no valid action weights")
        
        # Store in history
        self.performance_history.append({
            'version': self.policy_version,
            'timestamp': time.time(),
            'changes': changes,
            'expected_improvement': policy['expected_improvement'],
            'action_weights': action_weights
        })
    
    def rollback_policy(self):
        """Rollback to previous policy version."""
        if self.policy_version > 0:
            self.policy_version -= 1
            self.logger.warning(f"Rolled back to policy version {self.policy_version}")
            
            # Restore previous policy weights if available
            if self.policy_version > 0 and len(self.performance_history) >= 2:
                previous_policy = self.performance_history[-2]
                previous_weights = previous_policy.get('action_weights')
                if previous_weights:
                    self.query_optimizer.set_action_preferences(previous_weights)
                    self.logger.info(f"Restored action weights from version {self.policy_version}")
            else:
                # Clear preferences, return to default behavior
                self.query_optimizer.set_action_preferences(None)
                self.logger.info("Cleared action preferences, returned to default")
            
            # Record rollback event
            self.telemetry.store_safety_event({
                'severity': 'warning',
                'event_type': 'policy_rollback',
                'description': 'Policy rolled back due to safety event',
                'action_taken': f'Reverted to version {self.policy_version}'
            })
    
    def save_state(self, path: Optional[Path] = None):
        """
        Save policy learner state.
        
        Args:
            path: Optional path to save to
        """
        if path is None:
            path = Path(self.config['paths']['policies_dir']) / 'policy_learner_state.pt'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'policy_version': self.policy_version,
            'last_update': self.last_update,
            'current_phase': self.current_phase,
            'performance_history': self.performance_history,
            'policy_network_state': self.policy_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        
        torch.save(state, path)
        self.logger.info(f"Policy learner state saved to {path}")
    
    def load_state(self, path: Optional[Path] = None):
        """
        Load policy learner state.
        
        Args:
            path: Optional path to load from
        """
        if path is None:
            path = Path(self.config['paths']['policies_dir']) / 'policy_learner_state.pt'
        
        if not path.exists():
            self.logger.warning(f"No saved state found at {path}")
            return
        
        state = torch.load(path)
        
        self.policy_version = state['policy_version']
        self.last_update = state['last_update']
        self.current_phase = state['current_phase']
        self.performance_history = state['performance_history']
        self.policy_network.load_state_dict(state['policy_network_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        
        # Restore the most recent action weights
        if self.performance_history:
            latest_policy = self.performance_history[-1]
            latest_weights = latest_policy.get('action_weights')
            if latest_weights:
                self.query_optimizer.set_action_preferences(latest_weights)
                self.logger.info(f"Restored action weights from version {self.policy_version}")
        
        self.logger.info(f"Policy learner state loaded from {path}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current policy learner status.
        
        Returns:
            Status dictionary
        """
        return {
            'enabled': self.enabled,
            'policy_version': self.policy_version,
            'last_update': self.last_update,
            'current_phase': self.current_phase,
            'updates_count': len(self.performance_history),
            'time_since_update': time.time() - self.last_update if self.last_update else None
        }