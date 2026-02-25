import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any


class DQNNetwork(nn.Module):
    """Deep Q-Network for query optimization decisions."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [128, 128],
        activation: str = 'relu'
    ):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: List of hidden layer sizes
            activation: Activation function name
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.argmax(dim=1).item()


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    """Neural network for policy learning."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        """
        Initialize policy network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class MetaLearnerChromosome:
    """
    Chromosome for genetic algorithm-based meta-learning.
    Represents a set of hyperparameters and architecture choices.
    """
    
    def __init__(self, genes: Dict[str, Any]):
        """
        Initialize chromosome.
        
        Args:
            genes: Dictionary of hyperparameter values
        """
        self.genes = genes
        self.fitness = 0.0
        
    def mutate(self, mutation_rate: float, bounds: Dict[str, Tuple]):
        """
        Mutate genes randomly.
        
        Args:
            mutation_rate: Probability of mutation
            bounds: Valid ranges for each gene
        """
        for gene_name, value in self.genes.items():
            if np.random.random() < mutation_rate:
                if gene_name in bounds:
                    low, high = bounds[gene_name]
                    if isinstance(value, float):
                        # Add gaussian noise
                        noise = np.random.normal(0, (high - low) * 0.1)
                        self.genes[gene_name] = np.clip(value + noise, low, high)
                    elif isinstance(value, int):
                        # Random integer in range
                        self.genes[gene_name] = np.random.randint(low, high + 1)
    
    @staticmethod
    def crossover(
        parent1: 'MetaLearnerChromosome',
        parent2: 'MetaLearnerChromosome'
    ) -> 'MetaLearnerChromosome':
        """
        Create offspring through crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            New chromosome
        """
        child_genes = {}
        
        for gene_name in parent1.genes.keys():
            # Randomly choose from parents
            if np.random.random() < 0.5:
                child_genes[gene_name] = parent1.genes[gene_name]
            else:
                child_genes[gene_name] = parent2.genes[gene_name]
        
        return MetaLearnerChromosome(child_genes)
    
    def clone(self) -> 'MetaLearnerChromosome':
        """Create a copy of this chromosome."""
        return MetaLearnerChromosome(self.genes.copy())


class StateEncoder:
    """Encodes database state into feature vectors."""
    
    def __init__(self, feature_dim: int = 32):
        """
        Initialize state encoder.
        
        Args:
            feature_dim: Dimension of encoded state
        """
        self.feature_dim = feature_dim
        
    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Encode state dictionary into feature vector.
        
        Args:
            state: State dictionary
            
        Returns:
            Encoded feature vector
        """
        features = []
        
        # Cache hit rate
        features.append(state.get('cache_hit_rate', 0.0))
        
        # Connection count (normalized)
        conn_count = state.get('connection_count', 0)
        features.append(min(conn_count / 100.0, 1.0))
        
        # Load average
        load = state.get('load_average', {})
        features.append(load.get('cpu_percent', 0.0) / 100.0)
        features.append(load.get('memory_percent', 0.0) / 100.0)
        
        # Table sizes (take top 5, normalized)
        table_sizes = state.get('table_sizes', {})
        if table_sizes:
            sorted_sizes = sorted(table_sizes.values(), reverse=True)[:5]
            max_size = max(sorted_sizes) if sorted_sizes else 1
            features.extend([s / max_size for s in sorted_sizes])
            # Pad if less than 5
            features.extend([0.0] * (5 - len(sorted_sizes)))
        else:
            features.extend([0.0] * 5)
        
        # Index usage (take top 5, normalized)
        index_usage = state.get('index_usage', {})
        if index_usage:
            sorted_usage = sorted(index_usage.values(), reverse=True)[:5]
            max_usage = max(sorted_usage) if sorted_usage else 1
            features.extend([u / max_usage for u in sorted_usage])
            features.extend([0.0] * (5 - len(sorted_usage)))
        else:
            features.extend([0.0] * 5)
        
        # Time of day (cyclical encoding)
        import datetime
        now = datetime.datetime.now()
        hour_angle = 2 * np.pi * now.hour / 24
        features.append(np.sin(hour_angle))
        features.append(np.cos(hour_angle))
        
        # Day of week (cyclical encoding)
        day_angle = 2 * np.pi * now.weekday() / 7
        features.append(np.sin(day_angle))
        features.append(np.cos(day_angle))
        
        # Pad or truncate to feature_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
        
        return features