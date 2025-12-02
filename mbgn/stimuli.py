"""
Stimulus generation for MBGN experiments.

Provides various stimulus types:
- Binary patterns (abstract, easy to analyze)
- Colored shapes (more interpretable, similar to bee experiments)
- Parametric stimuli (controlled similarity)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Stimulus:
    """A single stimulus with its features and metadata."""
    vector: np.ndarray  # The input vector representation
    name: str  # Human-readable identifier
    category: str  # 'training' or 'transfer'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Stimulus):
            return self.name == other.name
        return False


class StimulusGenerator:
    """
    Generates stimuli for MBGN experiments.

    Supports multiple stimulus types:
    - 'binary': Random binary patterns
    - 'gaussian': Gaussian random vectors
    - 'shapes': Simple colored shape representations
    """

    def __init__(
        self,
        n_dims: int = 50,
        stimulus_type: str = 'binary',
        seed: Optional[int] = None
    ):
        """
        Initialize the stimulus generator.

        Args:
            n_dims: Dimensionality of stimulus vectors
            stimulus_type: Type of stimuli to generate ('binary', 'gaussian', 'shapes')
            seed: Random seed for reproducibility
        """
        self.n_dims = n_dims
        self.stimulus_type = stimulus_type
        self.rng = np.random.default_rng(seed)
        self._generated_stimuli: Dict[str, Stimulus] = {}

    def generate_binary_pattern(
        self,
        density: float = 0.5,
        name: Optional[str] = None,
        category: str = 'training'
    ) -> Stimulus:
        """
        Generate a random binary pattern.

        Args:
            density: Fraction of bits that are 1 (default 0.5)
            name: Identifier for the stimulus
            category: 'training' or 'transfer'

        Returns:
            Stimulus object with binary vector
        """
        vector = (self.rng.random(self.n_dims) < density).astype(np.float32)

        if name is None:
            name = f"binary_{len(self._generated_stimuli)}"

        stimulus = Stimulus(vector=vector, name=name, category=category)
        self._generated_stimuli[name] = stimulus
        return stimulus

    def generate_gaussian_pattern(
        self,
        mean: float = 0.5,
        std: float = 0.2,
        name: Optional[str] = None,
        category: str = 'training'
    ) -> Stimulus:
        """
        Generate a Gaussian random pattern, clipped to [0, 1].

        Args:
            mean: Mean of Gaussian distribution
            std: Standard deviation
            name: Identifier for the stimulus
            category: 'training' or 'transfer'

        Returns:
            Stimulus object with continuous vector
        """
        vector = self.rng.normal(mean, std, self.n_dims)
        vector = np.clip(vector, 0, 1).astype(np.float32)

        if name is None:
            name = f"gaussian_{len(self._generated_stimuli)}"

        stimulus = Stimulus(vector=vector, name=name, category=category)
        self._generated_stimuli[name] = stimulus
        return stimulus

    def generate_sparse_pattern(
        self,
        sparsity: float = 0.2,
        name: Optional[str] = None,
        category: str = 'training'
    ) -> Stimulus:
        """
        Generate a sparse continuous pattern (few non-zero values).

        This mimics sensory inputs where only a subset of receptors
        are strongly activated.

        Args:
            sparsity: Fraction of dimensions that are non-zero
            name: Identifier for the stimulus
            category: 'training' or 'transfer'

        Returns:
            Stimulus object with sparse continuous vector
        """
        vector = np.zeros(self.n_dims, dtype=np.float32)
        n_active = int(self.n_dims * sparsity)
        active_indices = self.rng.choice(self.n_dims, n_active, replace=False)
        vector[active_indices] = self.rng.uniform(0.5, 1.0, n_active).astype(np.float32)

        if name is None:
            name = f"sparse_{len(self._generated_stimuli)}"

        stimulus = Stimulus(vector=vector, name=name, category=category)
        self._generated_stimuli[name] = stimulus
        return stimulus

    def generate_normalized_pattern(
        self,
        name: Optional[str] = None,
        category: str = 'training'
    ) -> Stimulus:
        """
        Generate a unit-normalized random pattern.

        All stimuli have the same L2 norm, which controls for
        differences in overall activation magnitude.

        Args:
            name: Identifier for the stimulus
            category: 'training' or 'transfer'

        Returns:
            Stimulus object with normalized vector
        """
        vector = self.rng.normal(0, 1, self.n_dims)
        vector = vector / np.linalg.norm(vector)
        # Shift to positive and scale
        vector = (vector - vector.min()) / (vector.max() - vector.min())
        vector = vector.astype(np.float32)

        if name is None:
            name = f"normalized_{len(self._generated_stimuli)}"

        stimulus = Stimulus(vector=vector, name=name, category=category)
        self._generated_stimuli[name] = stimulus
        return stimulus

    def generate_correlated_pattern(
        self,
        base_pattern: Optional[np.ndarray] = None,
        correlation: float = 0.3,
        name: Optional[str] = None,
        category: str = 'training'
    ) -> Stimulus:
        """
        Generate a pattern correlated with a base pattern.

        Useful for testing how the model handles similar stimuli.

        Args:
            base_pattern: Pattern to correlate with (random if None)
            correlation: Target correlation (0 = independent, 1 = identical)
            name: Identifier for the stimulus
            category: 'training' or 'transfer'

        Returns:
            Stimulus object
        """
        if base_pattern is None:
            base_pattern = self.rng.random(self.n_dims)

        # Generate correlated pattern: x' = r*x + sqrt(1-r^2)*noise
        noise = self.rng.random(self.n_dims)
        vector = correlation * base_pattern + np.sqrt(1 - correlation**2) * noise
        vector = np.clip(vector, 0, 1).astype(np.float32)

        if name is None:
            name = f"correlated_{len(self._generated_stimuli)}"

        stimulus = Stimulus(vector=vector, name=name, category=category)
        self._generated_stimuli[name] = stimulus
        return stimulus

    def generate_training_set(
        self,
        n_stimuli: int = 4,
        **kwargs
    ) -> List[Stimulus]:
        """
        Generate a set of training stimuli.

        Args:
            n_stimuli: Number of stimuli to generate
            **kwargs: Additional arguments for pattern generation

        Returns:
            List of Stimulus objects
        """
        stimuli = []
        for i in range(n_stimuli):
            name = f"train_{chr(65 + i)}"  # A, B, C, D, ...
            stim = self._generate_pattern(name, 'training', **kwargs)
            stimuli.append(stim)

        return stimuli

    def generate_transfer_set(
        self,
        n_stimuli: int = 4,
        **kwargs
    ) -> List[Stimulus]:
        """
        Generate a set of transfer (novel) stimuli.

        Args:
            n_stimuli: Number of stimuli to generate
            **kwargs: Additional arguments for pattern generation

        Returns:
            List of Stimulus objects
        """
        stimuli = []
        for i in range(n_stimuli):
            # Start from E, F, G, H, ... for transfer stimuli
            name = f"transfer_{chr(69 + i)}"
            stim = self._generate_pattern(name, 'transfer', **kwargs)
            stimuli.append(stim)

        return stimuli

    def _generate_pattern(
        self,
        name: str,
        category: str,
        **kwargs
    ) -> Stimulus:
        """Generate a pattern based on current stimulus_type."""
        if self.stimulus_type == 'binary':
            return self.generate_binary_pattern(name=name, category=category, **kwargs)
        elif self.stimulus_type == 'gaussian':
            return self.generate_gaussian_pattern(name=name, category=category, **kwargs)
        elif self.stimulus_type == 'sparse':
            return self.generate_sparse_pattern(name=name, category=category, **kwargs)
        elif self.stimulus_type == 'normalized':
            return self.generate_normalized_pattern(name=name, category=category)
        else:
            raise ValueError(f"Unknown stimulus type: {self.stimulus_type}")

    def compute_similarity(
        self,
        stim1: Stimulus,
        stim2: Stimulus,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two stimuli.

        Args:
            stim1: First stimulus
            stim2: Second stimulus
            metric: Similarity metric ('cosine', 'jaccard', 'euclidean')

        Returns:
            Similarity value (higher = more similar)
        """
        v1, v2 = stim1.vector, stim2.vector

        if metric == 'cosine':
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))

        elif metric == 'jaccard':
            # For binary patterns
            intersection = np.sum(v1 * v2)
            union = np.sum(np.maximum(v1, v2))
            if union == 0:
                return 0.0
            return float(intersection / union)

        elif metric == 'euclidean':
            # Convert distance to similarity
            dist = np.linalg.norm(v1 - v2)
            return float(1.0 / (1.0 + dist))

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_all_stimuli(self) -> Dict[str, Stimulus]:
        """Return all generated stimuli."""
        return self._generated_stimuli.copy()

    def reset(self):
        """Clear all generated stimuli."""
        self._generated_stimuli.clear()


def create_experiment_stimuli(
    n_training: int = 4,
    n_transfer: int = 4,
    n_dims: int = 50,
    stimulus_type: str = 'binary',
    seed: Optional[int] = 42
) -> Tuple[List[Stimulus], List[Stimulus]]:
    """
    Convenience function to create training and transfer stimulus sets.

    Args:
        n_training: Number of training stimuli
        n_transfer: Number of transfer stimuli
        n_dims: Dimensionality of stimulus vectors
        stimulus_type: Type of stimuli
        seed: Random seed

    Returns:
        Tuple of (training_stimuli, transfer_stimuli)
    """
    generator = StimulusGenerator(
        n_dims=n_dims,
        stimulus_type=stimulus_type,
        seed=seed
    )

    training = generator.generate_training_set(n_training)
    transfer = generator.generate_transfer_set(n_transfer)

    return training, transfer
