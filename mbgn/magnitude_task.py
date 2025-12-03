"""
Magnitude comparison task for MBGN.

Tests whether MBGN can learn to compare stimuli by magnitude (intensity).
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .magnitude_stimuli import MagnitudeStimulusGenerator, MagnitudeStimulus
from .model import MBGN


class MagnitudeTaskType(Enum):
    """Type of magnitude task."""
    CHOOSE_BRIGHTER = "choose_brighter"  # Choose higher magnitude
    CHOOSE_DIMMER = "choose_dimmer"      # Choose lower magnitude


@dataclass
class MagnitudeTrialResult:
    """Result of a single magnitude comparison trial."""
    stim_a: MagnitudeStimulus
    stim_b: MagnitudeStimulus
    correct_choice: str  # 'A' or 'B'
    model_choice: str
    correct: bool
    aggregate_a: float
    aggregate_b: float
    magnitude_diff: float


class MagnitudeTask:
    """
    Magnitude comparison task.

    On each trial:
    1. Present stimulus A with magnitude M_A
    2. Present stimulus B with magnitude M_B
    3. Choose based on task type (brighter or dimmer)
    """

    def __init__(
        self,
        magnitudes: List[float],
        stimulus_generator: MagnitudeStimulusGenerator,
        task_type: MagnitudeTaskType = MagnitudeTaskType.CHOOSE_BRIGHTER,
        seed: Optional[int] = None
    ):
        """
        Args:
            magnitudes: List of magnitude levels to use
            stimulus_generator: Generator for magnitude stimuli
            task_type: Whether to choose brighter or dimmer
            seed: Random seed
        """
        self.magnitudes = sorted(magnitudes)
        self.stimulus_generator = stimulus_generator
        self.task_type = task_type
        self.rng = np.random.RandomState(seed)

    def generate_trial(
        self,
        stimulus_type: str = 'A',
        use_random_patterns: bool = False
    ) -> Tuple[MagnitudeStimulus, MagnitudeStimulus, str]:
        """
        Generate a trial with two different magnitudes.

        Returns:
            (stim_a, stim_b, correct_choice)
        """
        # Select two different magnitudes
        mag_a, mag_b = self.rng.choice(
            self.magnitudes, size=2, replace=False
        )

        # Generate stimuli
        gen_func = (self.stimulus_generator.generate_random
                    if use_random_patterns
                    else self.stimulus_generator.generate)

        stim_a = gen_func(mag_a, stimulus_type)
        stim_b = gen_func(mag_b, stimulus_type)

        # Determine correct choice
        if self.task_type == MagnitudeTaskType.CHOOSE_BRIGHTER:
            correct_choice = 'A' if mag_a > mag_b else 'B'
        else:  # CHOOSE_DIMMER
            correct_choice = 'A' if mag_a < mag_b else 'B'

        return stim_a, stim_b, correct_choice


class MagnitudeTaskRunner:
    """
    Run magnitude comparison task with MBGN.
    """

    def __init__(
        self,
        model: MBGN,
        task: MagnitudeTask,
        use_accommodation: bool = False,
        seed: Optional[int] = None
    ):
        """
        Args:
            model: MBGN model
            task: Magnitude task
            use_accommodation: Whether to use accommodation
            seed: Random seed
        """
        self.model = model
        self.task = task
        self.use_accommodation = use_accommodation
        self.rng = np.random.RandomState(seed)

    def run_trial(
        self,
        stim_a: MagnitudeStimulus,
        stim_b: MagnitudeStimulus,
        correct_choice: str,
        learn: bool = True
    ) -> MagnitudeTrialResult:
        """
        Run a single trial.

        Args:
            stim_a, stim_b: Stimuli to compare
            correct_choice: Expected answer ('A' or 'B')
            learn: Whether to update weights

        Returns:
            MagnitudeTrialResult
        """
        # Reset accommodation for clean comparison
        if not self.use_accommodation:
            self.model.reset_accommodation()

        # Present both stimuli and get aggregates
        result_a = self.model.forward(
            stim_a.pattern,
            update_accommodation=self.use_accommodation
        )
        result_b = self.model.forward(
            stim_b.pattern,
            update_accommodation=self.use_accommodation
        )

        agg_a = result_a.aggregate_activity
        agg_b = result_b.aggregate_activity

        # Model chooses based on aggregate comparison
        # For CHOOSE_BRIGHTER: higher aggregate = brighter
        if self.task.task_type == MagnitudeTaskType.CHOOSE_BRIGHTER:
            model_choice = 'A' if agg_a > agg_b else 'B'
        else:  # CHOOSE_DIMMER
            model_choice = 'A' if agg_a < agg_b else 'B'

        correct = (model_choice == correct_choice)

        # Learning (optional)
        if learn and hasattr(self.model, 'learn_from_feedback'):
            self.model.learn_from_feedback(correct)

        return MagnitudeTrialResult(
            stim_a=stim_a,
            stim_b=stim_b,
            correct_choice=correct_choice,
            model_choice=model_choice,
            correct=correct,
            aggregate_a=agg_a,
            aggregate_b=agg_b,
            magnitude_diff=abs(stim_a.magnitude - stim_b.magnitude)
        )

    def run_block(
        self,
        n_trials: int,
        learn: bool = True,
        stimulus_type: str = 'A',
        use_random_patterns: bool = False
    ) -> List[MagnitudeTrialResult]:
        """Run a block of trials."""
        results = []
        for _ in range(n_trials):
            stim_a, stim_b, correct = self.task.generate_trial(
                stimulus_type=stimulus_type,
                use_random_patterns=use_random_patterns
            )
            result = self.run_trial(stim_a, stim_b, correct, learn=learn)
            results.append(result)
        return results

    @staticmethod
    def compute_accuracy(results: List[MagnitudeTrialResult]) -> float:
        """Compute accuracy from results."""
        if not results:
            return 0.0
        return sum(r.correct for r in results) / len(results)

    @staticmethod
    def compute_accuracy_by_diff(
        results: List[MagnitudeTrialResult]
    ) -> Dict[float, float]:
        """Compute accuracy by magnitude difference."""
        by_diff = {}
        for r in results:
            diff = round(r.magnitude_diff, 2)
            if diff not in by_diff:
                by_diff[diff] = []
            by_diff[diff].append(r.correct)

        return {d: np.mean(v) for d, v in by_diff.items()}
