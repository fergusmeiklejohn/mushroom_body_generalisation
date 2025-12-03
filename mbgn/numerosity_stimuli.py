"""
Numerosity stimulus generation for MBGN Phase 2 experiments.

Provides stimuli that vary in number of elements for testing whether MBGN
can learn relational concepts beyond same/different.

Three stimulus types:
- Sparse binary: Most controlled, numerosity = number of active units
- Binary with element size: Controlled density, element_size * n_elements active
- 2D Gaussian blobs: More naturalistic, visual interpretation possible
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class NumerosityStimulus:
    """A stimulus with a specific number of elements."""
    vector: np.ndarray  # The input vector representation
    n_elements: int  # Number of elements in the stimulus
    element_type: str  # 'A', 'B', etc. for transfer testing
    seed: Optional[int] = None  # Seed used to generate this specific stimulus

    def __hash__(self):
        return hash((self.n_elements, self.element_type, self.seed))

    def __eq__(self, other):
        if isinstance(other, NumerosityStimulus):
            return (self.n_elements == other.n_elements and
                    self.element_type == other.element_type and
                    self.seed == other.seed)
        return False


@dataclass
class NumerosityStimulusSet:
    """Collection of numerosity stimuli with training/transfer splits."""
    training_numerosities: List[int]  # e.g., [2, 3, 5, 6]
    transfer_numerosities: List[int]  # e.g., [4, 7]
    training_element_type: str  # e.g., 'A'
    transfer_element_type: str  # e.g., 'B'
    n_input: int
    stimuli: Dict[Tuple[int, str], List[NumerosityStimulus]] = field(default_factory=dict)


class NumerosityStimulusGenerator:
    """
    Generates numerosity stimuli for MBGN experiments.

    Key design principle: numerosity (n_elements) should correlate with
    total input activity, so the aggregate pathway can learn to compare.
    """

    def __init__(
        self,
        n_input: int = 50,
        stimulus_type: str = 'sparse',
        seed: Optional[int] = None
    ):
        """
        Initialize the numerosity stimulus generator.

        Args:
            n_input: Dimensionality of stimulus vectors (matches MBGN input)
            stimulus_type: 'sparse', 'binary', or 'gaussian_2d'
            seed: Random seed for reproducibility
        """
        self.n_input = n_input
        self.stimulus_type = stimulus_type
        self.rng = np.random.default_rng(seed)
        self._base_seed = seed

    def make_sparse_stimulus(
        self,
        n_elements: int,
        element_type: str = 'A',
        instance_seed: Optional[int] = None
    ) -> NumerosityStimulus:
        """
        Create stimulus with exactly n_elements active input units.

        This is the most controlled stimulus type:
        - numerosity = number of active inputs
        - No confounds with element size or spatial arrangement

        Args:
            n_elements: Number of elements (active units) in stimulus
            element_type: Type identifier for transfer testing
            instance_seed: Specific seed for this stimulus instance

        Returns:
            NumerosityStimulus object
        """
        if n_elements > self.n_input:
            raise ValueError(f"n_elements ({n_elements}) > n_input ({self.n_input})")

        # Use instance seed if provided, else use generator's rng
        if instance_seed is not None:
            rng = np.random.default_rng(instance_seed)
        else:
            rng = self.rng

        vector = np.zeros(self.n_input, dtype=np.float32)
        active_indices = rng.choice(self.n_input, n_elements, replace=False)
        vector[active_indices] = 1.0

        return NumerosityStimulus(
            vector=vector,
            n_elements=n_elements,
            element_type=element_type,
            seed=instance_seed
        )

    def make_binary_element_stimulus(
        self,
        n_elements: int,
        element_size: int = 3,
        element_type: str = 'A',
        instance_seed: Optional[int] = None
    ) -> NumerosityStimulus:
        """
        Create stimulus with n_elements "regions", each activating element_size units.

        Total active units = n_elements * element_size
        This allows for more biologically realistic stimuli where each
        "element" has some spatial extent.

        Args:
            n_elements: Number of elements (active regions)
            element_size: Number of units per element
            element_type: Type identifier for transfer testing
            instance_seed: Specific seed for this stimulus instance

        Returns:
            NumerosityStimulus object
        """
        n_regions = self.n_input // element_size
        if n_elements > n_regions:
            raise ValueError(f"n_elements ({n_elements}) > available regions ({n_regions})")

        if instance_seed is not None:
            rng = np.random.default_rng(instance_seed)
        else:
            rng = self.rng

        vector = np.zeros(self.n_input, dtype=np.float32)

        # Choose non-overlapping regions
        region_indices = rng.choice(n_regions, n_elements, replace=False)

        for region_idx in region_indices:
            start = region_idx * element_size
            end = start + element_size
            vector[start:end] = 1.0

        return NumerosityStimulus(
            vector=vector,
            n_elements=n_elements,
            element_type=element_type,
            seed=instance_seed
        )

    def make_gaussian_2d_stimulus(
        self,
        n_elements: int,
        grid_size: int = 7,  # 7x7 = 49, close to default n_input=50
        blob_sigma: float = 0.8,
        element_type: str = 'A',
        instance_seed: Optional[int] = None
    ) -> NumerosityStimulus:
        """
        Create a 2D image with n_elements Gaussian blobs, then flatten.

        This creates more naturalistic stimuli that could represent
        visual scenes with discrete objects.

        Args:
            n_elements: Number of Gaussian blobs
            grid_size: Size of 2D grid (grid_size x grid_size)
            blob_sigma: Standard deviation of Gaussian blobs
            element_type: Type identifier for transfer testing
            instance_seed: Specific seed for this stimulus instance

        Returns:
            NumerosityStimulus object (vector is flattened grid)
        """
        if instance_seed is not None:
            rng = np.random.default_rng(instance_seed)
        else:
            rng = self.rng

        image = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Generate non-overlapping positions for blobs
        positions = self._sample_blob_positions(n_elements, grid_size, blob_sigma, rng)

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:grid_size, 0:grid_size]

        for (cx, cy) in positions:
            # Add Gaussian blob at (cx, cy)
            blob = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * blob_sigma**2))
            image += blob

        # Clip to [0, 1] and flatten
        image = np.clip(image, 0, 1)
        vector = image.flatten().astype(np.float32)

        # Pad if necessary to match n_input
        if len(vector) < self.n_input:
            vector = np.pad(vector, (0, self.n_input - len(vector)))
        elif len(vector) > self.n_input:
            vector = vector[:self.n_input]

        return NumerosityStimulus(
            vector=vector,
            n_elements=n_elements,
            element_type=element_type,
            seed=instance_seed
        )

    def _sample_blob_positions(
        self,
        n_elements: int,
        grid_size: int,
        blob_sigma: float,
        rng: np.random.Generator
    ) -> List[Tuple[float, float]]:
        """
        Sample non-overlapping positions for Gaussian blobs.

        Uses rejection sampling to ensure blobs don't overlap too much.
        """
        min_distance = blob_sigma * 2  # Minimum distance between blob centers
        positions = []
        max_attempts = 1000

        for _ in range(n_elements):
            for attempt in range(max_attempts):
                # Sample random position (with margin from edges)
                margin = blob_sigma
                cx = rng.uniform(margin, grid_size - 1 - margin)
                cy = rng.uniform(margin, grid_size - 1 - margin)

                # Check distance from existing positions
                valid = True
                for (px, py) in positions:
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    if dist < min_distance:
                        valid = False
                        break

                if valid:
                    positions.append((cx, cy))
                    break
            else:
                # If we couldn't find a valid position, use random anyway
                cx = rng.uniform(0, grid_size - 1)
                cy = rng.uniform(0, grid_size - 1)
                positions.append((cx, cy))

        return positions

    def make_stimulus(
        self,
        n_elements: int,
        element_type: str = 'A',
        instance_seed: Optional[int] = None,
        **kwargs
    ) -> NumerosityStimulus:
        """
        Generate a stimulus using the configured stimulus type.

        Args:
            n_elements: Number of elements
            element_type: Type identifier
            instance_seed: Specific seed for reproducibility
            **kwargs: Additional arguments for specific stimulus types

        Returns:
            NumerosityStimulus object
        """
        if self.stimulus_type == 'sparse':
            return self.make_sparse_stimulus(n_elements, element_type, instance_seed)
        elif self.stimulus_type == 'binary':
            element_size = kwargs.get('element_size', 3)
            return self.make_binary_element_stimulus(
                n_elements, element_size, element_type, instance_seed
            )
        elif self.stimulus_type == 'gaussian_2d':
            grid_size = kwargs.get('grid_size', 7)
            blob_sigma = kwargs.get('blob_sigma', 0.8)
            return self.make_gaussian_2d_stimulus(
                n_elements, grid_size, blob_sigma, element_type, instance_seed
            )
        else:
            raise ValueError(f"Unknown stimulus type: {self.stimulus_type}")

    def generate_stimulus_set(
        self,
        training_numerosities: List[int] = [2, 3, 5, 6],
        transfer_numerosities: List[int] = [4, 7],
        training_element_type: str = 'A',
        transfer_element_type: str = 'B',
        n_instances_per_condition: int = 10,
        **kwargs
    ) -> NumerosityStimulusSet:
        """
        Generate a complete set of training and transfer stimuli.

        Creates multiple instances of each numerosity/element_type combination
        for robust training and testing.

        Args:
            training_numerosities: Numerosities for training
            transfer_numerosities: Numerosities held out for transfer
            training_element_type: Element type for training
            transfer_element_type: Novel element type for transfer
            n_instances_per_condition: Number of stimulus instances per condition
            **kwargs: Additional arguments for stimulus generation

        Returns:
            NumerosityStimulusSet with all stimuli organized by condition
        """
        stimulus_set = NumerosityStimulusSet(
            training_numerosities=training_numerosities,
            transfer_numerosities=transfer_numerosities,
            training_element_type=training_element_type,
            transfer_element_type=transfer_element_type,
            n_input=self.n_input,
            stimuli={}
        )

        # Generate training stimuli (training numerosities, training element type)
        for n in training_numerosities:
            key = (n, training_element_type)
            stimulus_set.stimuli[key] = []
            for i in range(n_instances_per_condition):
                # Use deterministic seeds for reproducibility
                seed = hash((n, training_element_type, i, self._base_seed)) % (2**31)
                stim = self.make_stimulus(n, training_element_type, seed, **kwargs)
                stimulus_set.stimuli[key].append(stim)

        # Generate transfer stimuli (novel numerosities, training element type)
        for n in transfer_numerosities:
            key = (n, training_element_type)
            stimulus_set.stimuli[key] = []
            for i in range(n_instances_per_condition):
                seed = hash((n, training_element_type, i, self._base_seed)) % (2**31)
                stim = self.make_stimulus(n, training_element_type, seed, **kwargs)
                stimulus_set.stimuli[key].append(stim)

        # Generate transfer stimuli (training numerosities, novel element type)
        for n in training_numerosities:
            key = (n, transfer_element_type)
            stimulus_set.stimuli[key] = []
            for i in range(n_instances_per_condition):
                seed = hash((n, transfer_element_type, i, self._base_seed)) % (2**31)
                stim = self.make_stimulus(n, transfer_element_type, seed, **kwargs)
                stimulus_set.stimuli[key].append(stim)

        # Generate full transfer stimuli (novel numerosities, novel element type)
        for n in transfer_numerosities:
            key = (n, transfer_element_type)
            stimulus_set.stimuli[key] = []
            for i in range(n_instances_per_condition):
                seed = hash((n, transfer_element_type, i, self._base_seed)) % (2**31)
                stim = self.make_stimulus(n, transfer_element_type, seed, **kwargs)
                stimulus_set.stimuli[key].append(stim)

        return stimulus_set

    def get_stimulus_pair(
        self,
        n1: int,
        n2: int,
        element_type: str = 'A',
        instance_seed: Optional[int] = None
    ) -> Tuple[NumerosityStimulus, NumerosityStimulus]:
        """
        Generate a pair of stimuli with different numerosities.

        Useful for single trial generation.

        Args:
            n1: Number of elements in first stimulus
            n2: Number of elements in second stimulus
            element_type: Element type for both stimuli
            instance_seed: Base seed (stimuli get seed and seed+1)

        Returns:
            Tuple of (stimulus_1, stimulus_2)
        """
        seed1 = instance_seed
        seed2 = instance_seed + 1 if instance_seed is not None else None

        stim1 = self.make_stimulus(n1, element_type, seed1)
        stim2 = self.make_stimulus(n2, element_type, seed2)

        return stim1, stim2


def verify_numerosity_correlation(
    generator: NumerosityStimulusGenerator,
    numerosities: List[int] = [2, 3, 4, 5, 6, 7, 8],
    n_samples: int = 50
) -> Dict[str, float]:
    """
    Verify that stimulus total activity correlates with numerosity.

    This is a sanity check before running experiments to ensure
    the aggregate pathway has a signal to learn from.

    Args:
        generator: NumerosityStimulusGenerator instance
        numerosities: List of numerosities to test
        n_samples: Number of samples per numerosity

    Returns:
        Dictionary with correlation statistics
    """
    activities = []
    nums = []

    for n in numerosities:
        for _ in range(n_samples):
            stim = generator.make_stimulus(n)
            total_activity = stim.vector.sum()
            activities.append(total_activity)
            nums.append(n)

    # Compute correlation
    activities = np.array(activities)
    nums = np.array(nums)
    correlation = np.corrcoef(nums, activities)[0, 1]

    # Compute mean activity per numerosity
    mean_by_num = {n: np.mean(activities[nums == n]) for n in numerosities}
    std_by_num = {n: np.std(activities[nums == n]) for n in numerosities}

    return {
        'correlation': correlation,
        'mean_by_numerosity': mean_by_num,
        'std_by_numerosity': std_by_num,
        'overall_mean': activities.mean(),
        'overall_std': activities.std()
    }


def create_numerosity_stimuli(
    training_numerosities: List[int] = [2, 3, 5, 6],
    transfer_numerosities: List[int] = [4, 7],
    n_input: int = 50,
    stimulus_type: str = 'sparse',
    seed: int = 42,
    n_instances: int = 10
) -> NumerosityStimulusSet:
    """
    Convenience function to create a standard numerosity stimulus set.

    Args:
        training_numerosities: Numerosities for training
        transfer_numerosities: Numerosities for transfer testing
        n_input: Input dimensionality
        stimulus_type: Type of stimuli ('sparse', 'binary', 'gaussian_2d')
        seed: Random seed
        n_instances: Instances per condition

    Returns:
        NumerosityStimulusSet ready for experiments
    """
    generator = NumerosityStimulusGenerator(
        n_input=n_input,
        stimulus_type=stimulus_type,
        seed=seed
    )

    return generator.generate_stimulus_set(
        training_numerosities=training_numerosities,
        transfer_numerosities=transfer_numerosities,
        n_instances_per_condition=n_instances
    )
