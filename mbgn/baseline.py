"""
MLP Baseline for comparison with MBGN.

Implements a standard feedforward network trained with backpropagation
to demonstrate that the MBGN architecture specifically enables transfer,
not just any neural network.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .stimuli import Stimulus


@dataclass
class MLPConfig:
    """Configuration for MLP baseline."""
    n_input: int = 50
    n_hidden: int = 100  # Single hidden layer
    n_output: int = 1
    learning_rate: float = 0.01
    seed: Optional[int] = None


class MLPBaseline:
    """
    Standard MLP baseline for same/different task.

    Architecture: Input → Hidden (ReLU) → Output (Sigmoid)
    Training: Supervised learning with binary cross-entropy loss

    The MLP receives concatenated [sample, choice] as input and
    outputs probability of "GO" (match for DMTS, non-match for DNMTS).
    """

    def __init__(self, config: MLPConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Input is concatenated [sample, choice]
        self.n_input_total = config.n_input * 2

        # Xavier initialization
        self.W1 = self.rng.normal(
            0, np.sqrt(2.0 / self.n_input_total),
            (config.n_hidden, self.n_input_total)
        ).astype(np.float32)
        self.b1 = np.zeros(config.n_hidden, dtype=np.float32)

        self.W2 = self.rng.normal(
            0, np.sqrt(2.0 / config.n_hidden),
            (config.n_output, config.n_hidden)
        ).astype(np.float32)
        self.b2 = np.zeros(config.n_output, dtype=np.float32)

    def forward(self, sample: np.ndarray, choice: np.ndarray) -> Tuple[float, Dict]:
        """
        Forward pass.

        Args:
            sample: Sample stimulus vector
            choice: Choice stimulus vector

        Returns:
            Tuple of (output probability, cache for backprop)
        """
        # Concatenate inputs
        x = np.concatenate([sample, choice])

        # Hidden layer with ReLU
        z1 = self.W1 @ x + self.b1
        h = np.maximum(0, z1)  # ReLU

        # Output layer with sigmoid
        z2 = self.W2 @ h + self.b2
        y = 1 / (1 + np.exp(-z2))  # Sigmoid

        cache = {'x': x, 'z1': z1, 'h': h, 'z2': z2, 'y': y}
        return float(y[0]), cache

    def backward(self, cache: Dict, target: float):
        """
        Backward pass with gradient descent update.

        Args:
            cache: Forward pass cache
            target: Target output (1 for GO, 0 for NOGO)
        """
        x, h, y = cache['x'], cache['h'], cache['y']
        z1 = cache['z1']

        # Output layer gradient (BCE loss derivative)
        dy = y - target  # Shape: (1,)

        # W2 gradient
        dW2 = np.outer(dy, h)
        db2 = dy

        # Hidden layer gradient
        dh = self.W2.T @ dy
        dz1 = dh * (z1 > 0)  # ReLU derivative

        # W1 gradient
        dW1 = np.outer(dz1, x)
        db1 = dz1

        # Update weights
        lr = self.config.learning_rate
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, sample: np.ndarray, choice: np.ndarray) -> bool:
        """Predict GO (True) or NOGO (False)."""
        prob, _ = self.forward(sample, choice)
        return prob > 0.5


class MLPTaskRunner:
    """
    Runs DMTS/DNMTS task with MLP baseline.
    """

    def __init__(
        self,
        model: MLPBaseline,
        task_type: str = 'DMTS',
        seed: Optional[int] = None
    ):
        self.model = model
        self.task_type = task_type  # 'DMTS' or 'DNMTS'
        self.rng = np.random.default_rng(seed)

    def run_trial(
        self,
        sample: Stimulus,
        match: Stimulus,
        nonmatch: Stimulus,
        learn: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Run a single trial.

        Args:
            sample: Sample stimulus
            match: Matching choice stimulus
            nonmatch: Non-matching choice stimulus
            learn: Whether to update weights

        Returns:
            Tuple of (correct, trial_info)
        """
        # Randomly order choices
        match_first = self.rng.random() < 0.5

        if match_first:
            choice1, choice2 = match, nonmatch
        else:
            choice1, choice2 = nonmatch, match

        # Get model predictions for both choices
        prob1, cache1 = self.model.forward(sample.vector, choice1.vector)
        prob2, cache2 = self.model.forward(sample.vector, choice2.vector)

        # Model chooses higher probability option
        chose_first = prob1 > prob2

        if chose_first:
            chose_match = match_first
            chosen_cache = cache1
            chosen_prob = prob1
        else:
            chose_match = not match_first
            chosen_cache = cache2
            chosen_prob = prob2

        # Determine correctness based on task type
        if self.task_type == 'DMTS':
            correct = chose_match  # Match = reward
            target = 1.0 if match_first else 0.0  # Target for choice1
        else:  # DNMTS
            correct = not chose_match  # Non-match = reward
            target = 0.0 if match_first else 1.0  # Target for choice1

        # Learn from outcome
        if learn:
            # Train on both choices with appropriate targets
            if self.task_type == 'DMTS':
                target1 = 1.0 if match_first else 0.0
                target2 = 0.0 if match_first else 1.0
            else:
                target1 = 0.0 if match_first else 1.0
                target2 = 1.0 if match_first else 0.0

            self.model.backward(cache1, target1)
            self.model.backward(cache2, target2)

        return correct, {
            'chose_match': chose_match,
            'prob1': prob1,
            'prob2': prob2
        }

    def run_block(
        self,
        stimuli: List[Stimulus],
        n_trials: int,
        learn: bool = True
    ) -> List[bool]:
        """
        Run a block of trials.

        Args:
            stimuli: List of stimuli to use
            n_trials: Number of trials
            learn: Whether to update weights

        Returns:
            List of correct/incorrect outcomes
        """
        outcomes = []

        for _ in range(n_trials):
            # Random sample
            sample_idx = self.rng.integers(len(stimuli))
            sample = stimuli[sample_idx]

            # Random different stimulus for nonmatch
            other_indices = [i for i in range(len(stimuli)) if i != sample_idx]
            nonmatch_idx = self.rng.choice(other_indices)
            nonmatch = stimuli[nonmatch_idx]

            correct, _ = self.run_trial(sample, sample, nonmatch, learn=learn)
            outcomes.append(correct)

        return outcomes


def run_mlp_experiment(
    training_stimuli: List[Stimulus],
    transfer_stimuli: List[Stimulus],
    task_type: str = 'DMTS',
    n_training: int = 100,
    n_transfer: int = 20,
    n_hidden: int = 100,
    learning_rate: float = 0.01,
    seed: Optional[int] = None
) -> Dict:
    """
    Run complete MLP experiment for comparison with MBGN.

    Args:
        training_stimuli: Stimuli for training
        transfer_stimuli: Novel stimuli for transfer test
        task_type: 'DMTS' or 'DNMTS'
        n_training: Number of training trials
        n_transfer: Number of transfer trials
        n_hidden: Hidden layer size
        learning_rate: Learning rate
        seed: Random seed

    Returns:
        Dictionary with training and transfer accuracy
    """
    n_input = len(training_stimuli[0].vector)

    config = MLPConfig(
        n_input=n_input,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        seed=seed
    )

    model = MLPBaseline(config)
    runner = MLPTaskRunner(model, task_type=task_type, seed=seed)

    # Training
    train_outcomes = runner.run_block(training_stimuli, n_training, learn=True)
    train_accuracy = np.mean(train_outcomes)

    # Transfer (no learning)
    transfer_outcomes = runner.run_block(transfer_stimuli, n_transfer, learn=False)
    transfer_accuracy = np.mean(transfer_outcomes)

    return {
        'training_accuracy': train_accuracy,
        'transfer_accuracy': transfer_accuracy,
        'train_outcomes': train_outcomes,
        'transfer_outcomes': transfer_outcomes
    }
