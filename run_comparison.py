#!/usr/bin/env python3
"""
Compare Same/Different and Numerosity Tasks.

This script directly compares the two relational concepts that MBGN can learn:
1. Same/Different (Phase 1): Does stimulus B match sample A?
2. Numerosity (Phase 2): Which stimulus has more elements?

Key question: Do both tasks use the same aggregate pathway mechanism?

Usage:
    python run_comparison.py              # Run comparison with 10 seeds
    python run_comparison.py --n-seeds 5  # Use 5 seeds
    python run_comparison.py --quick      # Quick test with 3 seeds
"""

import argparse
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from scipy import stats

from mbgn import MBGN, MBGNConfig
from mbgn.training import Trainer, TrainingConfig
from mbgn.task import TaskType
from mbgn.numerosity_experiments import (
    NumerosityExperimentConfig,
    run_experiment_4_full_transfer,
)


@dataclass
class ComparisonResult:
    """Results from comparing two tasks."""
    task_name: str
    accuracies: List[float]
    mean: float
    std: float
    ci_lower: float
    ci_upper: float


def compute_ci(data: List[float], confidence: float = 0.95):
    """Compute confidence interval."""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * std_err
    return (mean - margin, mean + margin)


def run_same_different_experiment(seed: int, verbose: bool = False) -> float:
    """Run same/different experiment with given seed."""
    config = TrainingConfig(
        n_training_stimuli=4,
        n_transfer_stimuli=4,
        stimulus_dims=50,
        stimulus_type='binary',
        n_familiarization=10,
        n_training=100,
        n_transfer=40,
        block_size=20,
        task_type=TaskType.DMTS,
        model_seed=seed,
        stimulus_seed=seed,
        task_seed=seed
    )

    model_config = MBGNConfig(
        n_input=50,
        n_expansion=2000,
        sparsity_fraction=0.1,
        seed=seed
    )

    trainer = Trainer(model_config, config)
    results = trainer.run_experiment(verbose=verbose)
    return results.transfer_accuracy


def run_numerosity_experiment(seed: int, verbose: bool = False) -> float:
    """Run numerosity experiment with given seed."""
    config = NumerosityExperimentConfig(
        n_input=50,
        stimulus_type='sparse',
        training_numerosities=[2, 3, 5, 6],
        transfer_numerosities=[4, 7],
        n_training_trials=100,
        n_transfer_trials=40,
        block_size=20,
        model_seed=seed,
        task_seed=seed,
        stimulus_seed=seed,
        use_accommodation=False,
    )

    result = run_experiment_4_full_transfer(config, verbose=verbose)
    return result.transfer_accuracy


def run_comparison(n_seeds: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Run full comparison between same/different and numerosity tasks.
    """
    # Generate seeds
    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    print("=" * 70)
    print(f"SAME/DIFFERENT vs NUMEROSITY COMPARISON ({n_seeds} seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds}")

    # Run same/different experiments
    print("\n" + "-" * 70)
    print("Running SAME/DIFFERENT experiments...")
    print("-" * 70)
    sd_accuracies = []
    for i, seed in enumerate(seeds):
        acc = run_same_different_experiment(seed, verbose=False)
        sd_accuracies.append(acc)
        if verbose:
            print(f"  Seed {seed}: {acc:.1%}")

    # Run numerosity experiments
    print("\n" + "-" * 70)
    print("Running NUMEROSITY experiments...")
    print("-" * 70)
    num_accuracies = []
    for i, seed in enumerate(seeds):
        acc = run_numerosity_experiment(seed, verbose=False)
        num_accuracies.append(acc)
        if verbose:
            print(f"  Seed {seed}: {acc:.1%}")

    # Compute statistics
    sd_mean, sd_std = np.mean(sd_accuracies), np.std(sd_accuracies, ddof=1)
    sd_ci = compute_ci(sd_accuracies)
    num_mean, num_std = np.mean(num_accuracies), np.std(num_accuracies, ddof=1)
    num_ci = compute_ci(num_accuracies)

    sd_result = ComparisonResult(
        task_name="Same/Different",
        accuracies=sd_accuracies,
        mean=sd_mean,
        std=sd_std,
        ci_lower=sd_ci[0],
        ci_upper=sd_ci[1]
    )

    num_result = ComparisonResult(
        task_name="Numerosity",
        accuracies=num_accuracies,
        mean=num_mean,
        std=num_std,
        ci_lower=num_ci[0],
        ci_upper=num_ci[1]
    )

    # Statistical comparison
    # Paired t-test (same seeds)
    t_stat, p_value = stats.ttest_rel(sd_accuracies, num_accuracies)

    # Effect size (Cohen's d for paired samples)
    diff = np.array(sd_accuracies) - np.array(num_accuracies)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Task':<20} {'Mean':>10} {'Std':>10} {'95% CI':>20}")
    print("-" * 60)
    print(f"{'Same/Different':<20} {sd_mean:>9.1%} {sd_std:>9.1%} [{sd_ci[0]:.1%}, {sd_ci[1]:.1%}]")
    print(f"{'Numerosity':<20} {num_mean:>9.1%} {num_std:>9.1%} [{num_ci[0]:.1%}, {num_ci[1]:.1%}]")

    print("\n" + "-" * 70)
    print("STATISTICAL COMPARISON")
    print("-" * 70)
    print(f"Mean difference: {sd_mean - num_mean:+.1%}")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Effect size (Cohen's d): {d:.3f}")
    print(f"Significant difference (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if abs(sd_mean - num_mean) < 0.05 and p_value > 0.05:
        print("\nBoth tasks achieve SIMILAR transfer accuracy.")
        print("This supports the hypothesis that MBGN uses the same")
        print("aggregate pathway mechanism for both relational concepts.")
    elif num_mean > sd_mean and p_value < 0.05:
        print("\nNumerosity shows HIGHER transfer than same/different.")
        print("This may be because numerosity doesn't require accommodation,")
        print("making it a simpler task for the aggregate pathway.")
    elif sd_mean > num_mean and p_value < 0.05:
        print("\nSame/different shows HIGHER transfer than numerosity.")
        print("This might indicate that accommodation provides a stronger")
        print("signal than raw aggregate comparison.")
    else:
        print("\nResults are inconclusive due to high variance.")

    return {
        'same_different': sd_result,
        'numerosity': num_result,
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': d,
        'seeds': seeds
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare Same/Different and Numerosity tasks'
    )
    parser.add_argument(
        '--n-seeds', type=int, default=10,
        help='Number of random seeds (default: 10)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run with 3 seeds'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.n_seeds
    verbose = not args.quiet

    results = run_comparison(n_seeds=n_seeds, verbose=verbose)

    return results


if __name__ == '__main__':
    main()
