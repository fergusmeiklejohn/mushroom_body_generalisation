"""
Task definitions for MBGN experiments.

Implements Delayed Match-to-Sample (DMTS) and Delayed Non-Match-to-Sample (DNMTS)
tasks following experimental protocols used with bees.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from .stimuli import Stimulus


class TaskType(Enum):
    """Type of matching task."""
    DMTS = auto()  # Delayed Match to Sample (reward for same)
    DNMTS = auto()  # Delayed Non-Match to Sample (reward for different)


class TrialPhase(Enum):
    """Phase within a trial."""
    SAMPLE = auto()  # Sample stimulus presented
    DELAY = auto()  # Delay period (accommodation decays)
    CHOICE_1 = auto()  # First choice stimulus
    CHOICE_2 = auto()  # Second choice stimulus (if first was NOGO)
    OUTCOME = auto()  # Trial outcome (reward/no reward)


@dataclass
class TrialConfig:
    """Configuration for trial timing and structure."""
    delay_duration: float = 1.0  # Delay between sample and choice (seconds)
    inter_trial_interval: float = 5.0  # Time between trials (seconds)
    choice_order: str = 'random'  # 'random', 'match_first', 'nonmatch_first'


@dataclass
class Trial:
    """A single trial in the task."""
    sample: Stimulus  # The sample stimulus
    match: Stimulus  # The matching stimulus (same as sample)
    nonmatch: Stimulus  # The non-matching stimulus (different from sample)
    first_choice: Stimulus  # First choice presented
    second_choice: Stimulus  # Second choice presented
    match_is_first: bool  # Whether match is the first choice
    trial_number: int  # Trial index


@dataclass
class TrialResult:
    """Results from running a trial."""
    trial: Trial
    chose_first: bool  # Did agent choose the first option?
    chose_match: bool  # Did agent choose the matching stimulus?
    correct: bool  # Was the choice correct for this task type?
    reward: float  # Reward received
    sample_response: Dict[str, Any]  # Network response to sample
    choice_response: Dict[str, Any]  # Network response to chosen stimulus
    aggregate_before: float  # Aggregate activity after sample
    aggregate_choice: float  # Aggregate activity at choice


class BaseTask:
    """Base class for matching tasks."""

    def __init__(
        self,
        stimuli: List[Stimulus],
        task_type: TaskType,
        config: Optional[TrialConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the task.

        Args:
            stimuli: List of stimuli to use
            task_type: DMTS or DNMTS
            config: Trial configuration
            seed: Random seed
        """
        self.stimuli = stimuli
        self.task_type = task_type
        self.config = config or TrialConfig()
        self.rng = np.random.default_rng(seed)
        self.trial_count = 0

    def generate_trial(self) -> Trial:
        """Generate a single trial."""
        # Select sample and non-match randomly
        indices = self.rng.choice(len(self.stimuli), size=2, replace=False)
        sample = self.stimuli[indices[0]]
        nonmatch = self.stimuli[indices[1]]

        # The match is the same as the sample
        match = sample

        # Determine order of choices
        if self.config.choice_order == 'random':
            match_is_first = self.rng.random() < 0.5
        elif self.config.choice_order == 'match_first':
            match_is_first = True
        else:  # nonmatch_first
            match_is_first = False

        if match_is_first:
            first_choice = match
            second_choice = nonmatch
        else:
            first_choice = nonmatch
            second_choice = match

        trial = Trial(
            sample=sample,
            match=match,
            nonmatch=nonmatch,
            first_choice=first_choice,
            second_choice=second_choice,
            match_is_first=match_is_first,
            trial_number=self.trial_count
        )

        self.trial_count += 1
        return trial

    def generate_trial_sequence(self, n_trials: int) -> List[Trial]:
        """
        Generate a balanced sequence of trials.

        Args:
            n_trials: Number of trials to generate

        Returns:
            List of Trial objects
        """
        trials = []

        # Ensure balanced presentation of stimuli and positions
        for _ in range(n_trials):
            trials.append(self.generate_trial())

        return trials

    def get_reward(self, chose_match: bool) -> float:
        """
        Determine reward based on choice and task type.

        Args:
            chose_match: Whether the agent chose the matching stimulus

        Returns:
            Reward value (1.0 for correct, 0.0 for incorrect)
        """
        if self.task_type == TaskType.DMTS:
            # DMTS: reward for choosing match
            return 1.0 if chose_match else 0.0
        else:
            # DNMTS: reward for choosing non-match
            return 1.0 if not chose_match else 0.0

    def is_correct(self, chose_match: bool) -> bool:
        """Check if the choice was correct for this task type."""
        if self.task_type == TaskType.DMTS:
            return chose_match
        else:
            return not chose_match

    def reset(self):
        """Reset trial counter."""
        self.trial_count = 0


class DMTSTask(BaseTask):
    """Delayed Match-to-Sample task."""

    def __init__(
        self,
        stimuli: List[Stimulus],
        config: Optional[TrialConfig] = None,
        seed: Optional[int] = None
    ):
        super().__init__(stimuli, TaskType.DMTS, config, seed)


class DNMTSTask(BaseTask):
    """Delayed Non-Match-to-Sample task."""

    def __init__(
        self,
        stimuli: List[Stimulus],
        config: Optional[TrialConfig] = None,
        seed: Optional[int] = None
    ):
        super().__init__(stimuli, TaskType.DNMTS, config, seed)


class TaskRunner:
    """
    Runs trials and manages the interaction between model and task.
    """

    def __init__(self, model, task: BaseTask, seed: Optional[int] = None):
        """
        Initialize the task runner.

        Args:
            model: MBGN model instance
            task: Task instance (DMTS or DNMTS)
            seed: Random seed for forced choice decisions
        """
        self.model = model
        self.task = task
        # Separate RNG for forced choice to avoid corrupting trial generation
        self.forced_choice_rng = np.random.default_rng(seed)

    def run_trial(
        self,
        trial: Trial,
        learn: bool = True
    ) -> TrialResult:
        """
        Run a single trial.

        Args:
            trial: The trial to run
            learn: Whether to update weights based on outcome

        Returns:
            TrialResult with all trial information
        """
        # IMPORTANT: Reset accommodation at start of each trial
        # The same/different signal should only depend on whether the
        # stimulus was seen WITHIN THIS TRIAL (sample phase), not from
        # previous trials. This matches the biological paradigm where
        # inter-trial intervals are long enough for full recovery.
        self.model.reset_accommodation()
        self.model.clear_sample_reference()

        # === Sample Phase ===
        sample_result = self.model.forward(trial.sample.vector, update_accommodation=True)
        aggregate_after_sample = sample_result.aggregate_activity

        # Set sample's aggregate as reference for relative comparison
        self.model.set_sample_reference(sample_result.aggregate_activity)

        # === Delay Phase ===
        self.model.decay_accommodation(self.task.config.delay_duration)

        # === Choice Phase ===
        # Present first choice
        choice1_result = self.model.forward(
            trial.first_choice.vector,
            update_accommodation=True
        )

        if choice1_result.decision:
            # Agent chose first option
            chose_first = True
            chose_match = trial.match_is_first
            choice_result = choice1_result
        else:
            # Present second choice
            choice2_result = self.model.forward(
                trial.second_choice.vector,
                update_accommodation=True
            )

            if choice2_result.decision:
                # Agent chose second option
                chose_first = False
                chose_match = not trial.match_is_first
                choice_result = choice2_result
            else:
                # Neither chosen - forced choice (random)
                # Use dedicated RNG to avoid corrupting trial generation RNG
                chose_first = self.forced_choice_rng.random() < 0.5
                chose_match = trial.match_is_first if chose_first else not trial.match_is_first
                choice_result = choice1_result if chose_first else choice2_result

        # === Outcome Phase ===
        reward = self.task.get_reward(chose_match)
        correct = self.task.is_correct(chose_match)

        # Learn from outcome
        if learn:
            self.model.update_weights(reward, choice_result)

        return TrialResult(
            trial=trial,
            chose_first=chose_first,
            chose_match=chose_match,
            correct=correct,
            reward=reward,
            sample_response={
                'sparse_rep': sample_result.sparse_rep,
                'aggregate': sample_result.aggregate_activity,
                'out_combined': sample_result.out_combined
            },
            choice_response={
                'sparse_rep': choice_result.sparse_rep,
                'aggregate': choice_result.aggregate_activity,
                'out_combined': choice_result.out_combined
            },
            aggregate_before=aggregate_after_sample,
            aggregate_choice=choice_result.aggregate_activity
        )

    def run_block(
        self,
        n_trials: int,
        learn: bool = True
    ) -> List[TrialResult]:
        """
        Run a block of trials.

        Args:
            n_trials: Number of trials in block
            learn: Whether to update weights

        Returns:
            List of TrialResult objects
        """
        trials = self.task.generate_trial_sequence(n_trials)
        results = []

        for trial in trials:
            result = self.run_trial(trial, learn=learn)
            results.append(result)

        return results

    def compute_accuracy(self, results: List[TrialResult]) -> float:
        """Compute accuracy from a list of trial results."""
        if not results:
            return 0.0
        correct_count = sum(1 for r in results if r.correct)
        return correct_count / len(results)


class ExperimentRunner:
    """
    Runs complete experiments with training and transfer testing.
    """

    def __init__(
        self,
        model,
        training_stimuli: List[Stimulus],
        transfer_stimuli: List[Stimulus],
        task_type: TaskType = TaskType.DMTS,
        seed: Optional[int] = None
    ):
        """
        Initialize experiment runner.

        Args:
            model: MBGN model instance
            training_stimuli: Stimuli for training phase
            transfer_stimuli: Novel stimuli for transfer testing
            task_type: DMTS or DNMTS
            seed: Random seed
        """
        self.model = model
        self.training_stimuli = training_stimuli
        self.transfer_stimuli = transfer_stimuli
        self.task_type = task_type
        self.seed = seed

        # Create tasks
        if task_type == TaskType.DMTS:
            self.training_task = DMTSTask(training_stimuli, seed=seed)
            self.transfer_task = DMTSTask(transfer_stimuli, seed=seed)
        else:
            self.training_task = DNMTSTask(training_stimuli, seed=seed)
            self.transfer_task = DNMTSTask(transfer_stimuli, seed=seed)

        # Create runners with different seeds for forced choice RNG
        self.training_runner = TaskRunner(model, self.training_task, seed=seed)
        self.transfer_runner = TaskRunner(model, self.transfer_task, seed=seed + 1000 if seed else None)

    def run_experiment(
        self,
        n_familiarization: int = 10,
        n_training: int = 60,
        n_transfer: int = 20,
        block_size: int = 10
    ) -> Dict[str, Any]:
        """
        Run a complete experiment.

        Args:
            n_familiarization: Trials for familiarization (no learning)
            n_training: Trials for training
            n_transfer: Trials for transfer test
            block_size: Size of blocks for computing accuracy

        Returns:
            Dictionary with all results
        """
        results = {
            'familiarization': [],
            'training': [],
            'transfer': [],
            'training_accuracy_by_block': [],
            'transfer_accuracy': 0.0
        }

        # === Familiarization Phase ===
        if n_familiarization > 0:
            self.model.reset_accommodation()
            fam_results = self.training_runner.run_block(
                n_familiarization, learn=False
            )
            results['familiarization'] = fam_results

        # === Training Phase ===
        self.model.reset_accommodation()
        training_results = []
        for block_start in range(0, n_training, block_size):
            block_end = min(block_start + block_size, n_training)
            block_trials = block_end - block_start

            block_results = self.training_runner.run_block(
                block_trials, learn=True
            )
            training_results.extend(block_results)

            # Compute block accuracy
            block_accuracy = self.training_runner.compute_accuracy(block_results)
            results['training_accuracy_by_block'].append(block_accuracy)

        results['training'] = training_results

        # === Transfer Phase ===
        # Don't reset accommodation - carry over from training
        transfer_results = self.transfer_runner.run_block(
            n_transfer, learn=False
        )
        results['transfer'] = transfer_results
        results['transfer_accuracy'] = self.transfer_runner.compute_accuracy(
            transfer_results
        )

        # Compute overall training accuracy
        results['training_accuracy'] = self.training_runner.compute_accuracy(
            training_results
        )

        return results
