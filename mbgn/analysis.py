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


# ============================================================================
# Numerosity Analysis Functions
# ============================================================================

def analyze_numerosity_aggregate(
    model: MBGN,
    stimulus_generator,
    numerosities: List[int],
    n_samples: int = 50
) -> Dict[str, Any]:
    """
    Analyze how aggregate activity correlates with numerosity.

    Args:
        model: MBGN model
        stimulus_generator: NumerosityStimulusGenerator instance
        numerosities: List of numerosities to test
        n_samples: Number of samples per numerosity

    Returns:
        Dictionary with correlation analysis
    """
    aggregates = []
    nums = []

    for n in numerosities:
        for _ in range(n_samples):
            stim = stimulus_generator.make_stimulus(n)
            model.reset_accommodation()
            result = model.forward(stim.vector, update_accommodation=False)
            aggregates.append(result.aggregate_activity)
            nums.append(n)

    aggregates = np.array(aggregates)
    nums = np.array(nums)

    # Compute correlation
    correlation = np.corrcoef(nums, aggregates)[0, 1]

    # Compute statistics by numerosity
    mean_by_num = {n: np.mean(aggregates[nums == n]) for n in numerosities}
    std_by_num = {n: np.std(aggregates[nums == n]) for n in numerosities}

    # Check linearity via R-squared
    slope, intercept = np.polyfit(nums, aggregates, 1)
    predicted = slope * nums + intercept
    ss_res = np.sum((aggregates - predicted) ** 2)
    ss_tot = np.sum((aggregates - np.mean(aggregates)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'mean_by_numerosity': mean_by_num,
        'std_by_numerosity': std_by_num,
        'raw_aggregates': aggregates,
        'raw_numerosities': nums
    }


def analyze_numerical_distance_effect(
    results: List,  # List of NumerosityTrialResult
) -> Dict[str, Any]:
    """
    Analyze accuracy as a function of numerical distance.

    The numerical distance effect predicts that accuracy should increase
    with |n_a - n_b|, similar to what's observed in bees and humans.

    Args:
        results: List of NumerosityTrialResult objects

    Returns:
        Dictionary with distance effect analysis
    """
    by_distance: Dict[int, List[bool]] = {}

    for result in results:
        distance = abs(result.trial.n_a - result.trial.n_b)
        if distance not in by_distance:
            by_distance[distance] = []
        by_distance[distance].append(result.correct)

    # Compute accuracy by distance
    accuracy_by_distance = {
        dist: np.mean(correct_list)
        for dist, correct_list in by_distance.items()
    }

    # Check for monotonic increase (distance effect)
    distances = sorted(accuracy_by_distance.keys())
    accuracies = [accuracy_by_distance[d] for d in distances]

    is_monotonic = all(
        accuracies[i] <= accuracies[i+1] + 0.05  # Allow small noise
        for i in range(len(accuracies)-1)
    )

    # Compute correlation between distance and accuracy
    if len(distances) > 1:
        dist_corr = np.corrcoef(distances, accuracies)[0, 1]
    else:
        dist_corr = 0.0

    return {
        'accuracy_by_distance': accuracy_by_distance,
        'is_monotonic': is_monotonic,
        'distance_accuracy_correlation': dist_corr,
        'distances': distances,
        'accuracies': accuracies
    }


def plot_aggregate_vs_numerosity(
    analysis: Dict[str, Any],
    ax=None
):
    """
    Plot aggregate activity vs. numerosity.

    Args:
        analysis: Output from analyze_numerosity_aggregate
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

    numerosities = sorted(analysis['mean_by_numerosity'].keys())
    means = [analysis['mean_by_numerosity'][n] for n in numerosities]
    stds = [analysis['std_by_numerosity'][n] for n in numerosities]

    # Plot scatter of individual points
    ax.scatter(
        analysis['raw_numerosities'],
        analysis['raw_aggregates'],
        alpha=0.3,
        s=20,
        label='Individual samples'
    )

    # Plot means with error bars
    ax.errorbar(
        numerosities, means, yerr=stds,
        fmt='o-', markersize=10, capsize=5,
        label=f"Mean ± SD (r={analysis['correlation']:.3f})"
    )

    # Plot regression line
    x_line = np.array([min(numerosities), max(numerosities)])
    y_line = analysis['slope'] * x_line + analysis['intercept']
    ax.plot(x_line, y_line, 'r--', alpha=0.7, label=f"Linear fit (R²={analysis['r_squared']:.3f})")

    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Aggregate Activity')
    ax.set_title('Aggregate Activity vs. Numerosity')
    ax.legend()
    ax.set_xticks(numerosities)

    return ax


def plot_numerical_distance_effect(
    analysis: Dict[str, Any],
    ax=None
):
    """
    Plot accuracy vs. numerical distance.

    Args:
        analysis: Output from analyze_numerical_distance_effect
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

    distances = analysis['distances']
    accuracies = analysis['accuracies']

    ax.plot(distances, accuracies, 'o-', markersize=10, linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')

    ax.set_xlabel('Numerical Distance |N₁ - N₂|')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"Numerical Distance Effect (r={analysis['distance_accuracy_correlation']:.3f})")
    ax.set_xticks(distances)
    ax.set_ylim(0.3, 1.0)
    ax.legend()

    return ax


def plot_numerosity_transfer_matrix(
    pair_accuracy: Dict[Tuple[int, int], float],
    ax=None
):
    """
    Plot heatmap of accuracy for all numerosity pairs.

    Args:
        pair_accuracy: Dictionary mapping (n_a, n_b) pairs to accuracy
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    # Get all unique numerosities
    all_nums = set()
    for (n1, n2) in pair_accuracy.keys():
        all_nums.add(n1)
        all_nums.add(n2)
    all_nums = sorted(all_nums)

    # Build matrix
    matrix = np.full((len(all_nums), len(all_nums)), np.nan)
    for (n1, n2), acc in pair_accuracy.items():
        i = all_nums.index(min(n1, n2))
        j = all_nums.index(max(n1, n2))
        matrix[i, j] = acc

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(all_nums)))
    ax.set_yticks(range(len(all_nums)))
    ax.set_xticklabels(all_nums)
    ax.set_yticklabels(all_nums)
    ax.set_xlabel('Numerosity (larger)')
    ax.set_ylabel('Numerosity (smaller)')
    ax.set_title('Accuracy by Numerosity Pair')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Accuracy')

    # Add text annotations
    for i in range(len(all_nums)):
        for j in range(len(all_nums)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.0%}',
                       ha='center', va='center', fontsize=10)

    return ax


def compare_numerosity_same_different(
    numerosity_results: Dict[str, Any],
    same_different_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare numerosity and same/different transfer performance.

    Args:
        numerosity_results: Results from numerosity experiment
        same_different_results: Results from same/different experiment

    Returns:
        Dictionary with comparison metrics
    """
    return {
        'numerosity_training': numerosity_results.get('training_accuracy', 0),
        'numerosity_transfer': numerosity_results.get('transfer_full_accuracy', 0),
        'same_different_training': same_different_results.training_accuracy,
        'same_different_transfer': same_different_results.transfer_accuracy,
        'transfer_difference': (
            numerosity_results.get('transfer_full_accuracy', 0) -
            same_different_results.transfer_accuracy
        ),
        'both_above_chance': (
            numerosity_results.get('transfer_full_accuracy', 0) > 0.55 and
            same_different_results.transfer_accuracy > 0.55
        )
    }


def format_numerosity_results_table(
    experiment_results: Dict[str, Any]
) -> str:
    """
    Format numerosity experiment results as an ASCII table.

    Args:
        experiment_results: Output from run_all_experiments

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 70)
    lines.append("NUMEROSITY EXPERIMENT RESULTS")
    lines.append("=" * 70)
    lines.append(f"{'Experiment':<30} {'Training':<15} {'Transfer':<15}")
    lines.append("-" * 70)

    experiments = [
        ('exp1_baseline', 'Exp 1: Baseline'),
        ('exp2_novel_counts', 'Exp 2: Novel Counts'),
        ('exp3_novel_types', 'Exp 3: Novel Types'),
        ('exp4_full_transfer', 'Exp 4: Full Transfer'),
        ('exp7_choose_fewer', 'Exp 7: Choose Fewer'),
        ('exp8_distance_effect', 'Exp 8: Distance Effect'),
    ]

    for key, name in experiments:
        if key in experiment_results and experiment_results[key] is not None:
            result = experiment_results[key]
            train = f"{result.training_accuracy:.1%}"
            transfer = f"{result.transfer_accuracy:.1%}"
            lines.append(f"{name:<30} {train:<15} {transfer:<15}")

    lines.append("=" * 70)

    # Add ablation results if present
    if 'exp6_ablation' in experiment_results and experiment_results['exp6_ablation']:
        lines.append("\nABLATION STUDY")
        lines.append("-" * 70)
        for condition, result in experiment_results['exp6_ablation'].items():
            train = f"{result.training_accuracy:.1%}"
            transfer = f"{result.transfer_accuracy:.1%}"
            lines.append(f"{condition:<30} {train:<15} {transfer:<15}")
        lines.append("=" * 70)

    return "\n".join(lines)
