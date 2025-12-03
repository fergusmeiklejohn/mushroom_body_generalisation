"""
Magnitude stimuli for MBGN Phase 2 extension.

Generates stimuli that vary in magnitude (intensity/size) rather than count.
This tests whether the aggregate pathway can learn another relational concept
beyond same/different and numerosity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MagnitudeStimulus:
    """A stimulus with a specific magnitude."""
    pattern: np.ndarray
    magnitude: float  # The magnitude value (e.g., 0.2, 0.5, 0.8)
    stimulus_type: str  # e.g., 'A', 'B' for transfer


class MagnitudeStimulusGenerator:
    """
    Generate stimuli with varying magnitudes.

    Unlike numerosity (varying count), magnitude stimuli have:
    - Fixed number of active elements
    - Varying intensity/activation levels

    This tests whether the aggregate pathway can compare
    total activity when element count is held constant.
    """

    def __init__(
        self,
        n_input: int = 50,
        n_active: int = 10,  # Fixed number of active elements
        magnitude_type: str = 'intensity',  # 'intensity' or 'spread'
        seed: Optional[int] = None
    ):
        """
        Args:
            n_input: Dimension of input vector
            n_active: Number of active elements (constant)
            magnitude_type: How magnitude is encoded
                - 'intensity': Vary activation strength
                - 'spread': Vary how many units are active
            seed: Random seed
        """
        self.n_input = n_input
        self.n_active = n_active
        self.magnitude_type = magnitude_type
        self.rng = np.random.RandomState(seed)

        # Cache patterns for consistent stimulus types
        self._pattern_cache = {}

    def generate(
        self,
        magnitude: float,
        stimulus_type: str = 'A'
    ) -> MagnitudeStimulus:
        """
        Generate a stimulus with the given magnitude.

        Args:
            magnitude: Value between 0 and 1 indicating intensity
            stimulus_type: 'A' or 'B' for different pattern sets

        Returns:
            MagnitudeStimulus
        """
        # Get or create base pattern for this type
        if stimulus_type not in self._pattern_cache:
            self._pattern_cache[stimulus_type] = self.rng.choice(
                self.n_input, self.n_active, replace=False
            )

        pattern_indices = self._pattern_cache[stimulus_type]

        # Create stimulus based on magnitude type
        if self.magnitude_type == 'intensity':
            # All active elements have the same magnitude
            stimulus = np.zeros(self.n_input)
            stimulus[pattern_indices] = magnitude

        elif self.magnitude_type == 'spread':
            # Magnitude determines how many of the base pattern are active
            n_to_activate = max(1, int(self.n_active * magnitude))
            active_indices = pattern_indices[:n_to_activate]
            stimulus = np.zeros(self.n_input)
            stimulus[active_indices] = 1.0

        else:
            raise ValueError(f"Unknown magnitude_type: {self.magnitude_type}")

        return MagnitudeStimulus(
            pattern=stimulus,
            magnitude=magnitude,
            stimulus_type=stimulus_type
        )

    def generate_random(
        self,
        magnitude: float,
        stimulus_type: str = 'A'
    ) -> MagnitudeStimulus:
        """
        Generate stimulus with random pattern (not cached).

        Useful for transfer testing with novel patterns.
        """
        # Random pattern each time
        pattern_indices = self.rng.choice(
            self.n_input, self.n_active, replace=False
        )

        if self.magnitude_type == 'intensity':
            stimulus = np.zeros(self.n_input)
            stimulus[pattern_indices] = magnitude
        else:
            n_to_activate = max(1, int(self.n_active * magnitude))
            active_indices = pattern_indices[:n_to_activate]
            stimulus = np.zeros(self.n_input)
            stimulus[active_indices] = 1.0

        return MagnitudeStimulus(
            pattern=stimulus,
            magnitude=magnitude,
            stimulus_type=stimulus_type
        )

    def clear_cache(self):
        """Clear pattern cache for new experiment."""
        self._pattern_cache = {}


def verify_magnitude_correlation(
    generator: MagnitudeStimulusGenerator,
    magnitudes: List[float],
    n_samples: int = 100
) -> Dict:
    """
    Verify that total stimulus activity correlates with magnitude.

    Args:
        generator: Magnitude stimulus generator
        magnitudes: List of magnitudes to test
        n_samples: Samples per magnitude

    Returns:
        Dict with correlation and statistics
    """
    activities = {m: [] for m in magnitudes}

    for mag in magnitudes:
        for _ in range(n_samples):
            stim = generator.generate_random(mag)
            total_activity = np.sum(stim.pattern)
            activities[mag].append(total_activity)

    # Compute correlation
    all_mags = []
    all_acts = []
    for mag in magnitudes:
        all_mags.extend([mag] * len(activities[mag]))
        all_acts.extend(activities[mag])

    correlation = np.corrcoef(all_mags, all_acts)[0, 1]

    mean_by_mag = {m: np.mean(activities[m]) for m in magnitudes}
    std_by_mag = {m: np.std(activities[m]) for m in magnitudes}

    return {
        'correlation': correlation,
        'mean_by_magnitude': mean_by_mag,
        'std_by_magnitude': std_by_mag,
        'activities': activities
    }
