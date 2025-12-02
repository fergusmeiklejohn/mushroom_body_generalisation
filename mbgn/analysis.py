"""
Analysis and visualization tools for MBGN experiments.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .model import MBGN, ForwardResult
from .stimuli import Stimulus
from .task import TrialResult
from .training import ExperimentResults, compute_statistics


@dataclass
class RepresentationAnalysis:
    """Analysis of sparse representations."""
    sparsity: float  # Fraction of active units
    overlap_same: float  # Average overlap for same stimulus pairs
    overlap_different: float  # Average overlap for different stimulus pairs
    code_similarity_matrix: np.ndarray  # Pairwise similarity of codes


class Analyzer:
    """
    Analysis tools for MBGN experiments.
    """

    def __init__(self, model: MBGN):
        """
        Initialize analyzer with a model.

        Args:
            model: MBGN model to analyze
        """
        self.model = model

    def analyze_representations(
        self,
        stimuli: List[Stimulus],
        n_repeats: int = 5
    ) -> RepresentationAnalysis:
        """
        Analyze sparse code properties for a set of stimuli.

        Args:
            stimuli: Stimuli to analyze
            n_repeats: Number of presentations per stimulus

        Returns:
            RepresentationAnalysis with metrics
        """
        n_stimuli = len(stimuli)

        # Collect sparse codes
        codes = []
        for stim in stimuli:
            self.model.reset_accommodation()
            result = self.model.forward(stim.vector, update_accommodation=False)
            codes.append(result.sparse_rep)

        codes = np.array(codes)

        # Compute sparsity
        sparsity = np.mean(codes > 0)

        # Compute pairwise similarities
        similarity_matrix = np.zeros((n_stimuli, n_stimuli))
        for i in range(n_stimuli):
            for j in range(n_stimuli):
                # Cosine similarity
                norm_i = np.linalg.norm(codes[i])
                norm_j = np.linalg.norm(codes[j])
                if norm_i > 0 and norm_j > 0:
                    similarity_matrix[i, j] = (
                        np.dot(codes[i], codes[j]) / (norm_i * norm_j)
                    )

        # Compute average overlap for same vs different
        overlap_same = np.mean(np.diag(similarity_matrix))
        mask = ~np.eye(n_stimuli, dtype=bool)
        overlap_different = np.mean(similarity_matrix[mask])

        return RepresentationAnalysis(
            sparsity=sparsity,
            overlap_same=overlap_same,
            overlap_different=overlap_different,
            code_similarity_matrix=similarity_matrix
        )

    def analyze_accommodation(
        self,
        stimulus: Stimulus,
        n_presentations: int = 5,
        inter_presentation_time: float = 0.5
    ) -> Dict[str, List[float]]:
        """
        Analyze accommodation dynamics for repeated presentations.

        Args:
            stimulus: Stimulus to present repeatedly
            n_presentations: Number of presentations
            inter_presentation_time: Time between presentations

        Returns:
            Dictionary with accommodation metrics over time
        """
        self.model.reset_accommodation()

        aggregate_activities = []
        active_unit_counts = []
        accommodation_levels = []

        for i in range(n_presentations):
            # Present stimulus
            result = self.model.forward(stimulus.vector, update_accommodation=True)

            aggregate_activities.append(result.aggregate_activity)
            active_unit_counts.append(np.sum(result.sparse_rep > 0))
            accommodation_levels.append(np.mean(self.model.accommodation_state))

            # Decay between presentations
            if i < n_presentations - 1:
                self.model.decay_accommodation(inter_presentation_time)

        return {
            'aggregate_activity': aggregate_activities,
            'active_units': active_unit_counts,
            'accommodation_level': accommodation_levels
        }

    def analyze_same_different(
        self,
        stimuli: List[Stimulus]
    ) -> Dict[str, Any]:
        """
        Analyze how well the model distinguishes same vs different.

        Args:
            stimuli: List of stimuli to test

        Returns:
            Dictionary with same/different discrimination metrics
        """
        same_aggregates = []
        diff_aggregates = []

        for i, stim1 in enumerate(stimuli):
            for j, stim2 in enumerate(stimuli):
                # Reset and present first stimulus
                self.model.reset_accommodation()
                self.model.forward(stim1.vector, update_accommodation=True)

                # Brief delay
                self.model.decay_accommodation(0.5)

                # Present second stimulus
                result = self.model.forward(stim2.vector, update_accommodation=False)

                if i == j:
                    same_aggregates.append(result.aggregate_activity)
                else:
                    diff_aggregates.append(result.aggregate_activity)

        return {
            'same_aggregate_mean': np.mean(same_aggregates),
            'same_aggregate_std': np.std(same_aggregates),
            'diff_aggregate_mean': np.mean(diff_aggregates),
            'diff_aggregate_std': np.std(diff_aggregates),
            'discriminability': (
                np.mean(diff_aggregates) - np.mean(same_aggregates)
            ) / (np.std(same_aggregates) + np.std(diff_aggregates) + 1e-8)
        }


def compare_conditions(
    results_dict: Dict[str, List[ExperimentResults]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare results across experimental conditions.

    Args:
        results_dict: Dictionary mapping condition names to result lists

    Returns:
        Dictionary with statistics for each condition
    """
    comparison = {}

    for condition_name, results in results_dict.items():
        stats = compute_statistics(results)
        comparison[condition_name] = stats

    return comparison


def format_comparison_table(comparison: Dict[str, Dict[str, float]]) -> str:
    """
    Format comparison results as a table.

    Args:
        comparison: Output from compare_conditions

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Condition':<20} {'Train Acc':<15} {'Transfer Acc':<15}")
    lines.append("-" * 70)

    for condition, stats in comparison.items():
        train = f"{stats['training_accuracy_mean']:.1%} ± {stats['training_accuracy_std']:.1%}"
        transfer = f"{stats['transfer_accuracy_mean']:.1%} ± {stats['transfer_accuracy_std']:.1%}"
        lines.append(f"{condition:<20} {train:<15} {transfer:<15}")

    lines.append("=" * 70)
    return "\n".join(lines)


def learning_curve(results: ExperimentResults) -> Tuple[List[int], List[float]]:
    """
    Extract learning curve from experiment results.

    Args:
        results: Experiment results

    Returns:
        Tuple of (trial_numbers, cumulative_accuracies)
    """
    trial_nums = []
    cum_accs = []
    correct_count = 0

    for i, trial_result in enumerate(results.training_results):
        if trial_result.correct:
            correct_count += 1
        trial_nums.append(i + 1)
        cum_accs.append(correct_count / (i + 1))

    return trial_nums, cum_accs


def plot_learning_curve(
    results: ExperimentResults,
    ax=None,
    label: str = None
):
    """
    Plot learning curve.

    Args:
        results: Experiment results
        ax: Matplotlib axes (creates new figure if None)
        label: Label for the curve

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    trial_nums, cum_accs = learning_curve(results)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(trial_nums, cum_accs, label=label)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Cumulative Accuracy')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.set_ylim(0, 1)

    return ax


def plot_ablation_comparison(
    results_dict: Dict[str, List[ExperimentResults]],
    metric: str = 'transfer_accuracy'
):
    """
    Plot bar chart comparing ablation conditions.

    Args:
        results_dict: Dictionary mapping condition names to result lists
        metric: Which metric to plot

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    comparison = compare_conditions(results_dict)

    conditions = list(comparison.keys())
    means = [comparison[c][f'{metric}_mean'] for c in conditions]
    stds = [comparison[c][f'{metric}_std'] for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5)

    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Ablation Study Results')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_accommodation_dynamics(
    dynamics: Dict[str, List[float]],
    ax=None
):
    """
    Plot accommodation dynamics over repeated presentations.

    Args:
        dynamics: Output from Analyzer.analyze_accommodation
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    presentations = range(1, len(dynamics['aggregate_activity']) + 1)

    ax.plot(
        presentations,
        dynamics['aggregate_activity'],
        'o-',
        label='Aggregate Activity'
    )

    ax.set_xlabel('Presentation')
    ax.set_ylabel('Aggregate Activity')
    ax.set_title('Accommodation: Response to Repeated Stimuli')
    ax.legend()

    return ax


def plot_same_different_histogram(
    analysis: Dict[str, Any],
    ax=None
):
    """
    Plot histogram of aggregate activity for same vs different pairs.

    Args:
        analysis: Output from Analyzer.analyze_same_different
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Create approximate histograms from mean/std
    same_mean = analysis['same_aggregate_mean']
    same_std = analysis['same_aggregate_std']
    diff_mean = analysis['diff_aggregate_mean']
    diff_std = analysis['diff_aggregate_std']

    x = np.linspace(0, max(diff_mean + 3*diff_std, same_mean + 3*same_std), 100)

    from scipy.stats import norm
    same_dist = norm.pdf(x, same_mean, same_std + 1e-8)
    diff_dist = norm.pdf(x, diff_mean, diff_std + 1e-8)

    ax.fill_between(x, same_dist, alpha=0.5, label='Same stimulus')
    ax.fill_between(x, diff_dist, alpha=0.5, label='Different stimulus')

    ax.axvline(same_mean, color='blue', linestyle='--')
    ax.axvline(diff_mean, color='orange', linestyle='--')

    ax.set_xlabel('Aggregate Activity')
    ax.set_ylabel('Density')
    ax.set_title(f"Same vs Different (d' = {analysis['discriminability']:.2f})")
    ax.legend()

    return ax
