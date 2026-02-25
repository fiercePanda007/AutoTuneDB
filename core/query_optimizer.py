"""
DuckDB query optimizer driven by a Deep Q-Network (DQN).

This module chooses runtime DuckDB settings per query and learns from
execution latency feedback.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """Deep Q-Network for action-value estimation"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QueryOptimizer:
    """
    DQN-based query optimizer with DuckDB runtime control actions.
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, telemetry_collector):
        """
        Initialize query optimizer
        
        Args:
            config: System configuration
            db_manager: DatabaseManager instance
            telemetry_collector: TelemetryCollector instance
        """
        self.config = config
        self.db = db_manager
        self.telemetry = telemetry_collector
        self.logger = logging.getLogger(__name__)
        
        # Get level0 configuration
        self.level0_config = config.get('level0', {})
        self.learning_enabled = False
        
        # State and action spaces
        self.state_dim = 32  # Encoded state dimension
        self.actions = [
            {"name": "default", "setup_commands": [], "cleanup_commands": []},
            {"name": "threads_1", "setup_commands": ["SET threads TO 1"], "cleanup_commands": []},
            {"name": "threads_4", "setup_commands": ["SET threads TO 4"], "cleanup_commands": []},
            {"name": "threads_8", "setup_commands": ["SET threads TO 8"], "cleanup_commands": []},
            {"name": "mem_1gb", "setup_commands": ["SET memory_limit TO '1GB'"], "cleanup_commands": []},
            {"name": "mem_4gb", "setup_commands": ["SET memory_limit TO '4GB'"], "cleanup_commands": []},
            {"name": "mem_8gb", "setup_commands": ["SET memory_limit TO '8GB'"], "cleanup_commands": []},
            {
                "name": "aggressive",
                "setup_commands": ["SET threads TO 8", "SET memory_limit TO '8GB'"],
                "cleanup_commands": [],
            },
        ]
        self.action_dim = len(self.actions)
        
        self.action_preferences: Optional[Dict[str, float]] = None

        # Map action name -> index (needed to convert L1 policy to DQN action index)
        self.action_name_to_idx = {
            a['name']: i for i, a in enumerate(self.actions)
        }
        
        self.action_size = len(self.actions)
        self.state_size = self.state_dim
        self.query_type_action_overrides: Dict[str, str] = {}
        
        # DQN components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get hidden layer sizes from config
        hidden_layers = self.level0_config.get('hidden_layers', [128, 128])
        hidden_size = hidden_layers[0] if hidden_layers else 128
        
        self.policy_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Learning parameters
        self.learning_rate = self.level0_config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.gamma = self.level0_config.get('gamma', 0.95)
        self.epsilon = self.level0_config.get('epsilon_start', 1.0)
        self.epsilon_decay = self.level0_config.get('epsilon_decay', 0.990)
        self.epsilon_min = self.level0_config.get('epsilon_end', 0.05)
        
        # Experience replay
        self.memory = deque(maxlen=self.level0_config.get('buffer_size', 10000))
        self.batch_size = self.level0_config.get('batch_size', 64)
        
        # Target network update frequency
        self.target_update_freq = self.level0_config.get('target_update_freq', 100)
        
        # Training state
        self.steps = 0
        self.episodes = 0
        
        # Statistics
        self.stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'avg_loss': 0.0,
            'action_counts': {action['name']: 0 for action in self.actions}
        }
        
        self.logger.info("Query Optimizer (Level 0) initialized")
        
    def set_learning_enabled(self, enabled: bool):
        """Enable or disable learning"""
        self.learning_enabled = enabled
        self.logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
        
    def execute_query(self, query: str, query_type: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query with learned optimization
        
        This method provides compatibility with the existing interface
        
        Args:
            query: SQL query to execute
            query_type: Type of query
            state: Query state information
            
        Returns:
            Dictionary with execution results
        """
        # Use optimize_query for the actual work
        result = self.optimize_query(
            query,
            query_type=query_type,
            training=self.learning_enabled
        )
        
        # Format result to match expected interface
        return {
            'execution_time': result['execution_time'],
            'success': result['success'],
            'action': result['action'],
            'reward': result['reward'],
            'epsilon': result.get('epsilon'),
            'loss': result.get('loss'),
            'error': result.get('error')
        }
        
    def extract_features(self, query: str, query_type: Optional[str] = None) -> np.ndarray:
        """
        Extract numerical features from query
        
        Features include:
        - Query complexity indicators
        - Table and join counts
        - Estimated selectivity
        - Current system state
        """
        # Return a fixed-size feature vector
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        query_lower = query.lower()
        
        # Feature 0-3: Query type indicators
        features[0] = 1.0 if 'select' in query_lower else 0.0
        features[1] = 1.0 if 'join' in query_lower else 0.0
        features[2] = 1.0 if 'group by' in query_lower else 0.0
        features[3] = 1.0 if 'order by' in query_lower else 0.0
        
        # Feature 4-6: Complexity indicators
        features[4] = min(float(query_lower.count('join')) / 5.0, 1.0)
        features[5] = min(float(query_lower.count('where')) / 3.0, 1.0)
        features[6] = min(float(query_lower.count('and') + query_lower.count('or')) / 5.0, 1.0)
        
        # Feature 7-9: Aggregation indicators
        features[7] = 1.0 if any(agg in query_lower for agg in ['count(', 'sum(', 'avg(', 'max(', 'min(']) else 0.0
        features[8] = 1.0 if 'distinct' in query_lower else 0.0
        features[9] = 1.0 if '(' in query_lower else 0.0
        
        # Feature 10-14: Additional characteristics
        features[10] = min(float(len(query)) / 1000.0, 1.0)  # Query length (normalized)
        features[11] = min(float(query_lower.count('from')) / 5.0, 1.0)  # Table count
        features[12] = 1.0 if 'limit' in query_lower else 0.0
        features[13] = 1.0 if 'union' in query_lower else 0.0
        features[14] = 1.0 if 'subquery' in query_lower else 0.0
       
        # Query-type one-hot segment
        query_type_idx = {
            'select_simple': 15,
            'join_two_tables': 16,
            'join_multiple': 17,
            'aggregation': 18,
            'analytical': 19,
            'autodb_lookup': 20,
        }
        if query_type in query_type_idx:
            features[query_type_idx[query_type]] = 1.0

        # Deterministic runtime state features (avoid random noise in state).
        features[21] = min(max(float(self.epsilon), 0.0), 1.0)
        features[22] = min(float(len(self.memory)) / max(float(self.memory.maxlen), 1.0), 1.0)
        features[23] = min(float(self.episodes) / 10000.0, 1.0)
        
        return features

    def set_query_type_action_overrides(self, overrides: Optional[Dict[str, str]]):
        """
        Configure hard action overrides by query type.

        Supports a special key `__default__` used when query type is not found.
        """
        if not overrides:
            self.query_type_action_overrides = {}
            self.logger.info("Cleared query-type action overrides")
            return

        valid_names = set(self.action_name_to_idx.keys())
        cleaned: Dict[str, str] = {}
        for query_type, action_name in overrides.items():
            if action_name not in valid_names:
                self.logger.warning(
                    f"Ignoring invalid action override '{action_name}' for query type '{query_type}'"
                )
                continue
            cleaned[str(query_type)] = action_name

        self.query_type_action_overrides = cleaned
        self.logger.info(f"Set query-type action overrides: {cleaned}")

    def _resolve_override_action(self, query_type: Optional[str]) -> Optional[int]:
        if not self.query_type_action_overrides:
            return None

        action_name = None
        if query_type and query_type in self.query_type_action_overrides:
            action_name = self.query_type_action_overrides[query_type]
        elif '__default__' in self.query_type_action_overrides:
            action_name = self.query_type_action_overrides['__default__']

        if action_name is None:
            return None
        return self.action_name_to_idx.get(action_name)
        
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        query_type: Optional[str] = None
    ) -> int:
        override_action = self._resolve_override_action(query_type)
        if override_action is not None:
            self.stats['action_counts'][self.actions[override_action]['name']] += 1
            return override_action

        # Exploration
        if training and random.random() < self.epsilon:
            if self.action_preferences:
                # sample action index using preference distribution
                names = list(self.action_preferences.keys())
                probs = np.array([self.action_preferences[n] for n in names], dtype=np.float64)
                probs = probs / probs.sum()
                chosen_name = np.random.choice(names, p=probs)
                action = self.action_name_to_idx.get(chosen_name, random.randrange(self.action_size))
            else:
                action = random.randrange(self.action_size)
        else:
            # Exploitation (DQN)
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.policy_net(st)
                action = q.argmax().item()

        self.stats['action_counts'][self.actions[action]['name']] += 1
        return action
        
    def execute_with_action(self, query: str, action_idx: int) -> Tuple[float, bool, Optional[str]]:
        action = self.actions[action_idx]
        conn = None
        error_msg = None

        try:
            conn = self.db.get_connection()  # should return duckdb.DuckDBPyConnection

            # (Optional but recommended) reset to known defaults each run
            self._apply_duckdb_defaults(conn)

            # Apply action settings
            for cmd in action["setup_commands"]:
                self._apply_setting(conn, cmd)

            start_time = time.time()
            conn.execute(query).fetchall()
            execution_time = time.time() - start_time

            self.logger.info(f"DuckDB query executed with '{action['name']}' in {execution_time*1000:.2f}ms")
            return execution_time, True, None

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"DuckDB execution failed with '{action['name']}': {error_msg}")
            return 0.0, False, error_msg

        finally:
            if conn:
                self.db.return_connection(conn)

    def _apply_setting(self, conn, statement: str):
        """Best-effort setting application; skip unsupported parameters."""
        try:
            conn.execute(statement)
        except Exception as e:
            self.logger.debug(f"Skipping setting '{statement}': {e}")

    def _apply_duckdb_defaults(self, conn):
        # Pick defaults suitable for your environment/config.
        defaults = self.level0_config.get(
            "duckdb_defaults",
            {
                "threads": 4,
                "memory_limit": "2GB",
            },
        )

        threads = defaults.get("threads")
        if threads is not None:
            self._apply_setting(conn, f"SET threads TO {int(threads)}")

        memory_limit = defaults.get("memory_limit")
        if memory_limit:
            self._apply_setting(conn, f"SET memory_limit TO '{memory_limit}'")

    def set_action_preferences(self, weights: Optional[Dict[str, float]]):
        """
        weights: mapping action_name -> probability weight (should sum to ~1)
        If None: revert to pure epsilon-greedy DQN behavior.
        """
        self.action_preferences = weights
        if weights:
            self.logger.info(f"Action preferences updated: {weights}")
        else:
            self.logger.info("Action preferences cleared")

    def calculate_reward(self, execution_time: float, success: bool, baseline_time: float = 0.05) -> float:
        """
        Calculate reward for an execution
        
        Args:
            execution_time: Time taken to execute query (seconds)
            success: Whether execution succeeded
            baseline_time: Expected baseline time
            
        Returns:
            Reward value
        """
        if not success:
            return -10.0  # Large penalty for failures
        
        # Time-based reward
        time_reward = -20.0 * (execution_time / baseline_time - 1.0)
        
        # Bonus for fast execution
        if execution_time < baseline_time * 0.8:
            time_reward += 5.0
        
        # Cap rewards
        time_reward = max(-25.0, min(25.0, time_reward))
        
        return time_reward
        
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def train_step(self) -> float:
        """
        Perform one training step using experience replay
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Updated target network")
        
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.logger.debug(f"Epsilon decayed to {self.epsilon:.4f}")
            
    def optimize_query(
        self,
        query: str,
        query_type: Optional[str] = None,
        training: bool = True,
        baseline_time: float = 0.05
    ) -> Dict:
        """
        Main optimization loop for a query
        
        Args:
            query: SQL query to optimize
            training: Whether to train the network
            baseline_time: Expected baseline execution time
            
        Returns:
            Dictionary with execution results
        """
        # Extract features
        state = self.extract_features(query, query_type=query_type)
        
        # Select action
        action = self.select_action(state, training=training, query_type=query_type)
        
        # Execute query with action
        execution_time, success, error_msg = self.execute_with_action(query, action)
        
        # Calculate reward
        reward = self.calculate_reward(execution_time, success, baseline_time)
        
        # Get next state
        next_state = state
        done = True
        loss = None
        
        # Store experience if training
        if training:
            self.store_experience(state, action, reward, next_state, done)
            
            # Train if enough experiences
            loss = self.train_step()
            
            # Update statistics
            self.stats['episodes'] += 1
            self.stats['total_reward'] += reward
            self.stats['avg_loss'] = loss
            self.steps += 1
            self.episodes += 1
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Periodically update target network
            if self.episodes % self.target_update_freq == 0:
                self.update_target_network()
        
        return {
            'execution_time': execution_time,
            'success': success,
            'error': error_msg,
            'action': self.actions[action]['name'],
            'reward': reward,
            'epsilon': self.epsilon,
            'loss': loss
        }
        
    def get_stats(self) -> Dict:
        """Return current statistics"""
        return self.stats.copy()
        
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': self.stats
        }, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.stats = checkpoint['stats']
        self.logger.info(f"Model loaded from {path}")
