"""
Numerosity task definitions for MBGN Phase 2 experiments.

Implements numerosity comparison tasks:
- "Choose More": Select stimulus with more elements
- "Choose Fewer": Select stimulus with fewer elements
- "Match Numerosity": Match sample numerosity (combines same/different with numerosity)

Key difference from DMTS/DNMTS:
- This is 2AFC comparison (A vs B), not match-to-sample
- No sample phase in basic version - direct comparison
- Compare aggregates rather than threshold-based GO/NOGO
- Accommodation should be disabled for clean numerosity signal
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from .numerosity_stimuli import NumerosityStimulus, NumerosityStimulusGenerator


class NumerosityTaskType(Enum):
    """Type of numerosity task."""
    CHOOSE_MORE = auto()      # Reward for choosing stimulus with more elements
    CHOOSE_FEWER = auto()     # Reward for choosing stimulus with fewer elements
    MATCH_NUMEROSITY = auto()  # Reward for matching sample's numerosity


@dataclass
class NumerosityTrialConfig:
    """Configuration for numerosity trial timing and structure."""
    # Timing
    inter_stimulus_interval: float = 0.1  # Brief delay between A and B
    inter_trial_interval: float = 1.0     # Time between trials

    # Accommodation handling
    use_accommodation: bool = False  # Whether to use accommodation mechanism

    # Stimulus presentation order
    order: str = 'random'  # 'random', 'more_first', 'fewer_first'


@dataclass
class NumerosityTrial:
    """A single numerosity comparison trial."""
    stim_a: NumerosityStimulus  # First stimulus
    stim_b: NumerosityStimulus  # Second stimulus
    n_a: int  # Number of elements in A
    n_b: int  # Number of elements in B
    correct_choice: str  # 'A' or 'B' (correct answer)
    trial_number: int


@dataclass
class NumerosityTrialResult:
    """Results from running a numerosity trial."""
    trial: NumerosityTrial
    choice: str  # 'A' or 'B'
    correct: bool
    reward: float
    aggregate_a: float  # Aggregate activity for stimulus A
    aggregate_b: float  # Aggregate activity for stimulus B
    aggregate_diff: float  # A - B
    response_info: Dict[str, Any] = field(default_factory=dict)


class NumerosityTask:
    """
    Numerosity comparison task.

    Presents two stimuli (A and B) with different numbers of elements.
    Agent must choose the one with more (or fewer) elements.
    """

    def __init__(
        self,
        numerosities: List[int],
        stimulus_generator: NumerosityStimulusGenerator,
        task_type: NumerosityTaskType = NumerosityTaskType.CHOOSE_MORE,
        config: Optional[NumerosityTrialConfig] = None,
        element_type: str = 'A',
        seed: Optional[int] = None
    ):
        """
        Initialize numerosity task.

        Args:
            numerosities: List of valid numerosities to use (e.g., [2, 3, 5, 6])
            stimulus_generator: Generator for creating stimuli
            task_type: CHOOSE_MORE, CHOOSE_FEWER, or MATCH_NUMEROSITY
            config: Trial configuration
            element_type: Element type to use for stimuli
            seed: Random seed
        """
        self.numerosities = numerosities
        self.stimulus_generator = stimulus_generator
        self.task_type = task_type
        self.config = config or NumerosityTrialConfig()
        self.element_type = element_type
        self.rng = np.random.default_rng(seed)
        self.trial_count = 0

    def generate_trial(self) -> NumerosityTrial:
        """Generate a single numerosity comparison trial."""
        # Select two different numerosities
        n_a, n_b = self.rng.choice(self.numerosities, size=2, replace=False)

        # Generate stimuli with unique seeds
        seed_base = self.rng.integers(0, 2**30)
        stim_a = self.stimulus_generator.make_stimulus(
            n_a, self.element_type, instance_seed=seed_base
        )
        stim_b = self.stimulus_generator.make_stimulus(
            n_b, self.element_type, instance_seed=seed_base + 1
        )

        # Optionally swap based on presentation order config
        if self.config.order == 'more_first':
            if n_b > n_a:
                stim_a, stim_b = stim_b, stim_a
                n_a, n_b = n_b, n_a
        elif self.config.order == 'fewer_first':
            if n_a > n_b:
                stim_a, stim_b = stim_b, stim_a
                n_a, n_b = n_b, n_a
        # else 'random': keep as generated

        # Determine correct choice
        if self.task_type == NumerosityTaskType.CHOOSE_MORE:
            correct_choice = 'A' if n_a > n_b else 'B'
        elif self.task_type == NumerosityTaskType.CHOOSE_FEWER:
            correct_choice = 'A' if n_a < n_b else 'B'
        else:  # MATCH_NUMEROSITY handled separately
            correct_choice = 'A'  # Placeholder

        trial = NumerosityTrial(
            stim_a=stim_a,
            stim_b=stim_b,
            n_a=n_a,
            n_b=n_b,
            correct_choice=correct_choice,
            trial_number=self.trial_count
        )

        self.trial_count += 1
        return trial

    def generate_trial_sequence(
        self,
        n_trials: int,
        balanced: bool = True
    ) -> List[NumerosityTrial]:
        """
        Generate a sequence of trials.

        Args:
            n_trials: Number of trials to generate
            balanced: If True, balance numerosity pairs

        Returns:
            List of NumerosityTrial objects
        """
        if balanced:
            return self._generate_balanced_sequence(n_trials)
        else:
            return [self.generate_trial() for _ in range(n_trials)]

    def _generate_balanced_sequence(self, n_trials: int) -> List[NumerosityTrial]:
        """Generate a balanced sequence covering all numerosity pairs."""
        trials = []

        # Generate all valid pairs (n_a < n_b to avoid duplicates)
        pairs = []
        for i, n_a in enumerate(self.numerosities):
            for n_b in self.numerosities[i+1:]:
                pairs.append((n_a, n_b))

        # We'll cycle through pairs, presenting each in both orders
        extended_pairs = []
        for (n_a, n_b) in pairs:
            extended_pairs.append((n_a, n_b))
            extended_pairs.append((n_b, n_a))

        # Shuffle and repeat to fill n_trials
        self.rng.shuffle(extended_pairs)
        pair_cycle = extended_pairs * ((n_trials // len(extended_pairs)) + 1)

        for i in range(n_trials):
            n_a, n_b = pair_cycle[i]

            seed_base = self.rng.integers(0, 2**30)
            stim_a = self.stimulus_generator.make_stimulus(
                n_a, self.element_type, instance_seed=seed_base
            )
            stim_b = self.stimulus_generator.make_stimulus(
                n_b, self.element_type, instance_seed=seed_base + 1
            )

            # Determine correct choice
            if self.task_type == NumerosityTaskType.CHOOSE_MORE:
                correct_choice = 'A' if n_a > n_b else 'B'
            elif self.task_type == NumerosityTaskType.CHOOSE_FEWER:
                correct_choice = 'A' if n_a < n_b else 'B'
            else:
                correct_choice = 'A'  # Placeholder for MATCH_NUMEROSITY

            trial = NumerosityTrial(
                stim_a=stim_a,
                stim_b=stim_b,
                n_a=n_a,
                n_b=n_b,
                correct_choice=correct_choice,
                trial_number=self.trial_count
            )
            self.trial_count += 1
            trials.append(trial)

        return trials

    def generate_specific_pair_trial(
        self,
        n_a: int,
        n_b: int
    ) -> NumerosityTrial:
        """
        Generate a trial with specific numerosities.

        Useful for testing specific comparisons.
        """
        seed_base = self.rng.integers(0, 2**30)
        stim_a = self.stimulus_generator.make_stimulus(
            n_a, self.element_type, instance_seed=seed_base
        )
        stim_b = self.stimulus_generator.make_stimulus(
            n_b, self.element_type, instance_seed=seed_base + 1
        )

        if self.task_type == NumerosityTaskType.CHOOSE_MORE:
            correct_choice = 'A' if n_a > n_b else 'B'
        elif self.task_type == NumerosityTaskType.CHOOSE_FEWER:
            correct_choice = 'A' if n_a < n_b else 'B'
        else:
            correct_choice = 'A'

        return NumerosityTrial(
            stim_a=stim_a,
            stim_b=stim_b,
            n_a=n_a,
            n_b=n_b,
            correct_choice=correct_choice,
            trial_number=self.trial_count
        )

    def get_reward(self, choice: str, trial: NumerosityTrial) -> float:
        """
        Determine reward based on choice.

        Args:
            choice: 'A' or 'B'
            trial: The trial being evaluated

        Returns:
            Reward value (1.0 for correct, 0.0 for incorrect)
        """
        return 1.0 if choice == trial.correct_choice else 0.0

    def is_correct(self, choice: str, trial: NumerosityTrial) -> bool:
        """Check if choice was correct."""
        return choice == trial.correct_choice

    def reset(self):
        """Reset trial counter."""
        self.trial_count = 0


class NumerosityTaskRunner:
    """
    Runs numerosity trials and manages model-task interaction.

    Key difference from TaskRunner for DMTS:
    - Presents two stimuli and compares aggregates
    - Decision based on aggregate comparison, not threshold
    - Accommodation typically disabled for numerosity
    """

    def __init__(
        self,
        model,
        task: NumerosityTask,
        use_accommodation: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize numerosity task runner.

        Args:
            model: MBGN model instance
            task: NumerosityTask instance
            use_accommodation: Whether to use accommodation (default False)
            seed: Random seed for tie-breaking
        """
        self.model = model
        self.task = task
        self.use_accommodation = use_accommodation
        self.rng = np.random.default_rng(seed)

    def run_trial(
        self,
        trial: NumerosityTrial,
        learn: bool = True
    ) -> NumerosityTrialResult:
        """
        Run a single numerosity comparison trial.

        Protocol:
        1. Reset accommodation (if using)
        2. Present stimulus A, record aggregate
        3. (Optional) Brief delay
        4. Present stimulus B, record aggregate
        5. Choose based on aggregate comparison
        6. Update weights based on reward

        Args:
            trial: The trial to run
            learn: Whether to update weights

        Returns:
            NumerosityTrialResult
        """
        # Reset state at trial start
        if self.use_accommodation:
            self.model.reset_accommodation()
        else:
            # Ensure accommodation is zeroed for clean comparison
            self.model.reset_accommodation()

        self.model.clear_sample_reference()

        # Present stimulus A
        result_a = self.model.forward(
            trial.stim_a.vector,
            update_accommodation=self.use_accommodation
        )
        aggregate_a = result_a.aggregate_activity

        # Optional delay (with accommodation decay if enabled)
        if self.use_accommodation and self.task.config.inter_stimulus_interval > 0:
            self.model.decay_accommodation(self.task.config.inter_stimulus_interval)

        # Present stimulus B
        result_b = self.model.forward(
            trial.stim_b.vector,
            update_accommodation=self.use_accommodation
        )
        aggregate_b = result_b.aggregate_activity

        # Make decision based on aggregate comparison
        aggregate_diff = aggregate_a - aggregate_b

        if self.task.task_type == NumerosityTaskType.CHOOSE_MORE:
            # Higher aggregate should indicate more elements
            # Choose A if aggregate_a > aggregate_b (more elements in A)
            if aggregate_a > aggregate_b:
                choice = 'A'
            elif aggregate_b > aggregate_a:
                choice = 'B'
            else:
                # Tie-breaker
                choice = 'A' if self.rng.random() < 0.5 else 'B'
        elif self.task.task_type == NumerosityTaskType.CHOOSE_FEWER:
            # Lower aggregate should indicate fewer elements
            if aggregate_a < aggregate_b:
                choice = 'A'
            elif aggregate_b < aggregate_a:
                choice = 'B'
            else:
                choice = 'A' if self.rng.random() < 0.5 else 'B'
        else:
            # Default to comparing aggregates
            choice = 'A' if aggregate_a > aggregate_b else 'B'

        # Evaluate choice
        correct = self.task.is_correct(choice, trial)
        reward = self.task.get_reward(choice, trial)

        # Learning update
        if learn:
            self._update_weights(
                choice=choice,
                correct=correct,
                reward=reward,
                aggregate_a=aggregate_a,
                aggregate_b=aggregate_b,
                trial=trial,
                result_chosen=result_a if choice == 'A' else result_b
            )

        return NumerosityTrialResult(
            trial=trial,
            choice=choice,
            correct=correct,
            reward=reward,
            aggregate_a=aggregate_a,
            aggregate_b=aggregate_b,
            aggregate_diff=aggregate_diff,
            response_info={
                'sparse_rep_a': result_a.sparse_rep,
                'sparse_rep_b': result_b.sparse_rep,
                'out_combined_a': result_a.out_combined,
                'out_combined_b': result_b.out_combined
            }
        )

    def _update_weights(
        self,
        choice: str,
        correct: bool,
        reward: float,
        aggregate_a: float,
        aggregate_b: float,
        trial: NumerosityTrial,
        result_chosen
    ):
        """
        Update weights based on numerosity comparison outcome.

        Learning rule for numerosity:
        - Reinforce the aggregate pathway to strengthen the correlation
          between aggregate activity and numerosity
        - If correct: reinforce the current aggregate-to-choice mapping
        - If incorrect: weaken the current mapping

        This differs from same/different where we learn a threshold.
        Here we learn that aggregate correlates with numerosity.
        """
        # Use the standard weight update with the chosen stimulus's result
        # The reward signal will reinforce correct aggregate comparisons
        self.model.update_weights(
            reward=reward,
            result=result_chosen,
            learn_specific=False,  # Don't learn specific patterns for numerosity
            learn_aggregate=True   # Learn aggregate relationship
        )

    def run_block(
        self,
        n_trials: int,
        learn: bool = True,
        balanced: bool = True
    ) -> List[NumerosityTrialResult]:
        """
        Run a block of numerosity trials.

        Args:
            n_trials: Number of trials
            learn: Whether to update weights
            balanced: Whether to balance numerosity pairs

        Returns:
            List of NumerosityTrialResult
        """
        trials = self.task.generate_trial_sequence(n_trials, balanced=balanced)
        results = []

        for trial in trials:
            result = self.run_trial(trial, learn=learn)
            results.append(result)

        return results

    def compute_accuracy(self, results: List[NumerosityTrialResult]) -> float:
        """Compute accuracy from results."""
        if not results:
            return 0.0
        correct_count = sum(1 for r in results if r.correct)
        return correct_count / len(results)

    def compute_accuracy_by_distance(
        self,
        results: List[NumerosityTrialResult]
    ) -> Dict[int, float]:
        """
        Compute accuracy as a function of numerical distance.

        The numerical distance effect predicts that accuracy should
        increase with |n_a - n_b|.

        Returns:
            Dictionary mapping distance to accuracy
        """
        by_distance: Dict[int, List[bool]] = {}

        for result in results:
            distance = abs(result.trial.n_a - result.trial.n_b)
            if distance not in by_distance:
                by_distance[distance] = []
            by_distance[distance].append(result.correct)

        return {
            dist: np.mean(correct_list)
            for dist, correct_list in by_distance.items()
        }

    def compute_accuracy_by_pair(
        self,
        results: List[NumerosityTrialResult]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute accuracy for each numerosity pair.

        Returns:
            Dictionary mapping (n_a, n_b) pairs to accuracy
        """
        by_pair: Dict[Tuple[int, int], List[bool]] = {}

        for result in results:
            # Normalize pair order (smaller first)
            pair = tuple(sorted([result.trial.n_a, result.trial.n_b]))
            if pair not in by_pair:
                by_pair[pair] = []
            by_pair[pair].append(result.correct)

        return {
            pair: np.mean(correct_list)
            for pair, correct_list in by_pair.items()
        }


class NumerosityExperimentRunner:
    """
    Runs complete numerosity experiments with training and transfer.
    """

    def __init__(
        self,
        model,
        training_numerosities: List[int],
        transfer_numerosities: List[int],
        stimulus_generator: NumerosityStimulusGenerator,
        task_type: NumerosityTaskType = NumerosityTaskType.CHOOSE_MORE,
        training_element_type: str = 'A',
        transfer_element_type: str = 'B',
        use_accommodation: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize experiment runner.

        Args:
            model: MBGN model instance
            training_numerosities: Numerosities for training (e.g., [2, 3, 5, 6])
            transfer_numerosities: Novel numerosities for transfer (e.g., [4, 7])
            stimulus_generator: Generator for stimuli
            task_type: CHOOSE_MORE or CHOOSE_FEWER
            training_element_type: Element type for training stimuli
            transfer_element_type: Novel element type for transfer
            use_accommodation: Whether to use accommodation
            seed: Random seed
        """
        self.model = model
        self.training_numerosities = training_numerosities
        self.transfer_numerosities = transfer_numerosities
        self.stimulus_generator = stimulus_generator
        self.task_type = task_type
        self.training_element_type = training_element_type
        self.transfer_element_type = transfer_element_type
        self.use_accommodation = use_accommodation
        self.seed = seed

        # Create tasks for different conditions
        self._create_tasks()

    def _create_tasks(self):
        """Create tasks for all experimental conditions."""
        seed = self.seed

        # Training task
        self.training_task = NumerosityTask(
            numerosities=self.training_numerosities,
            stimulus_generator=self.stimulus_generator,
            task_type=self.task_type,
            element_type=self.training_element_type,
            seed=seed
        )

        # Transfer: novel numerosities, same element type
        self.transfer_novel_num_task = NumerosityTask(
            numerosities=self.transfer_numerosities + self.training_numerosities,
            stimulus_generator=self.stimulus_generator,
            task_type=self.task_type,
            element_type=self.training_element_type,
            seed=seed + 1000 if seed else None
        )

        # Transfer: training numerosities, novel element type
        self.transfer_novel_type_task = NumerosityTask(
            numerosities=self.training_numerosities,
            stimulus_generator=self.stimulus_generator,
            task_type=self.task_type,
            element_type=self.transfer_element_type,
            seed=seed + 2000 if seed else None
        )

        # Full transfer: novel numerosities AND novel element type
        self.transfer_full_task = NumerosityTask(
            numerosities=self.transfer_numerosities,
            stimulus_generator=self.stimulus_generator,
            task_type=self.task_type,
            element_type=self.transfer_element_type,
            seed=seed + 3000 if seed else None
        )

    def run_experiment(
        self,
        n_training: int = 100,
        n_transfer: int = 40,
        block_size: int = 20
    ) -> Dict[str, Any]:
        """
        Run complete numerosity experiment.

        Args:
            n_training: Number of training trials
            n_transfer: Number of transfer trials per condition
            block_size: Size of blocks for accuracy tracking

        Returns:
            Dictionary with all results
        """
        results = {
            'training_results': [],
            'training_accuracy_by_block': [],
            'transfer_novel_num': [],
            'transfer_novel_type': [],
            'transfer_full': [],
            'transfer_novel_num_accuracy': 0.0,
            'transfer_novel_type_accuracy': 0.0,
            'transfer_full_accuracy': 0.0,
        }

        # Training phase
        training_runner = NumerosityTaskRunner(
            self.model,
            self.training_task,
            use_accommodation=self.use_accommodation,
            seed=self.seed
        )

        for block_start in range(0, n_training, block_size):
            block_end = min(block_start + block_size, n_training)
            block_trials = block_end - block_start

            block_results = training_runner.run_block(
                block_trials, learn=True, balanced=True
            )
            results['training_results'].extend(block_results)
            block_accuracy = training_runner.compute_accuracy(block_results)
            results['training_accuracy_by_block'].append(block_accuracy)

        results['training_accuracy'] = training_runner.compute_accuracy(
            results['training_results']
        )

        # Transfer tests (no learning)
        # Novel numerosities
        runner_novel_num = NumerosityTaskRunner(
            self.model,
            self.transfer_novel_num_task,
            use_accommodation=self.use_accommodation,
            seed=self.seed + 100 if self.seed else None
        )
        results['transfer_novel_num'] = runner_novel_num.run_block(
            n_transfer, learn=False, balanced=True
        )
        results['transfer_novel_num_accuracy'] = runner_novel_num.compute_accuracy(
            results['transfer_novel_num']
        )

        # Novel element type
        runner_novel_type = NumerosityTaskRunner(
            self.model,
            self.transfer_novel_type_task,
            use_accommodation=self.use_accommodation,
            seed=self.seed + 200 if self.seed else None
        )
        results['transfer_novel_type'] = runner_novel_type.run_block(
            n_transfer, learn=False, balanced=True
        )
        results['transfer_novel_type_accuracy'] = runner_novel_type.compute_accuracy(
            results['transfer_novel_type']
        )

        # Full transfer
        runner_full = NumerosityTaskRunner(
            self.model,
            self.transfer_full_task,
            use_accommodation=self.use_accommodation,
            seed=self.seed + 300 if self.seed else None
        )
        results['transfer_full'] = runner_full.run_block(
            n_transfer, learn=False, balanced=True
        )
        results['transfer_full_accuracy'] = runner_full.compute_accuracy(
            results['transfer_full']
        )

        # Analyze numerical distance effect
        results['distance_effect_training'] = training_runner.compute_accuracy_by_distance(
            results['training_results']
        )
        results['distance_effect_transfer'] = runner_novel_num.compute_accuracy_by_distance(
            results['transfer_novel_num']
        )

        return results
