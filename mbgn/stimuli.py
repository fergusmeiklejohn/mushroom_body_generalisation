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

            if self.stimulus_type == 'binary':
                stim = self.generate_binary_pattern(
                    name=name, category='training', **kwargs
                )
            elif self.stimulus_type == 'gaussian':
                stim = self.generate_gaussian_pattern(
                    name=name, category='training', **kwargs
                )
            else:
                raise ValueError(f"Unknown stimulus type: {self.stimulus_type}")

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

            if self.stimulus_type == 'binary':
                stim = self.generate_binary_pattern(
                    name=name, category='transfer', **kwargs
                )
            elif self.stimulus_type == 'gaussian':
                stim = self.generate_gaussian_pattern(
                    name=name, category='transfer', **kwargs
                )
            else:
                raise ValueError(f"Unknown stimulus type: {self.stimulus_type}")

            stimuli.append(stim)

        return stimuli

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
