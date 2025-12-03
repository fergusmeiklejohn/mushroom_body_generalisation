#!/usr/bin/env python3
"""
Statistical Analysis for MBGN Phase 2 Numerosity Experiments.

Runs experiments with multiple seeds to calculate:
- Mean accuracy with confidence intervals
- Significance tests (binomial test against chance)
- Comparison between conditions

Usage:
    python run_statistical_analysis.py                  # Run all analyses
    python run_statistical_analysis.py --n-seeds 10    # Use 10 seeds
    python run_statistical_analysis.py --quick          # Quick run with 5 seeds
"""

import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from scipy import stats
import warnings

from mbgn import MBGN, MBGNConfig
from mbgn.numerosity_stimuli import NumerosityStimulusGenerator
from mbgn.numerosity_experiments import (
    NumerosityExperimentConfig,
    run_experiment_1_baseline,
    run_experiment_2_novel_counts,
    run_experiment_3_novel_types,
    run_experiment_4_full_transfer,
    run_experiment_6_ablation,
    run_experiment_7_choose_fewer,
    run_experiment_8_distance_effect,
)


@dataclass
class StatisticalResult:
    """Results from statistical analysis of an experiment."""
    name: str
    n_seeds: int
    accuracies: List[float]
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    p_value: float  # Against chance (0.5)
    significant: bool
    effect_size: float  # Cohen's d


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)

    # Use t-distribution for small samples
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * std_err

    return (mean - margin, mean + margin)


def binomial_test_vs_chance(
    accuracies: List[float],
    n_trials_per_seed: int,
    chance: float = 0.5
) -> float:
    """
    Test if accuracy is significantly above chance.

    Uses an aggregate binomial test across all seeds.
    """
    # Total successes and trials
    total_correct = sum(acc * n_trials_per_seed for acc in accuracies)
    total_trials = len(accuracies) * n_trials_per_seed

    # One-sided binomial test
    result = stats.binomtest(
        int(total_correct),
        total_trials,
        chance,
        alternative='greater'
    )
    return result.pvalue


def cohens_d(accuracies: List[float], chance: float = 0.5) -> float:
    """Compute Cohen's d effect size against chance."""
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)
    if std == 0:
        return float('inf') if mean > chance else 0
    return (mean - chance) / std


def run_experiment_with_seeds(
    experiment_func,
    experiment_name: str,
    seeds: List[int],
    base_config: NumerosityExperimentConfig,
    verbose: bool = True
) -> StatisticalResult:
    """
    Run an experiment multiple times with different seeds.
    """
    accuracies = []

    if verbose:
        print(f"\nRunning {experiment_name} with {len(seeds)} seeds...")

    for i, seed in enumerate(seeds):
        # Create config with this seed
        config = NumerosityExperimentConfig(
            n_input=base_config.n_input,
            stimulus_type=base_config.stimulus_type,
            training_numerosities=base_config.training_numerosities,
            transfer_numerosities=base_config.transfer_numerosities,
            n_training_trials=base_config.n_training_trials,
            n_transfer_trials=base_config.n_transfer_trials,
            block_size=base_config.block_size,
            model_seed=seed,
            task_seed=seed,
            stimulus_seed=seed,
            use_accommodation=base_config.use_accommodation,
        )

        # Run experiment
        result = experiment_func(config, verbose=False)
        accuracies.append(result.transfer_accuracy)

        if verbose:
            print(f"  Seed {seed}: {result.transfer_accuracy:.1%}")

    # Compute statistics
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)
    ci_lower, ci_upper = compute_confidence_interval(accuracies)
    p_value = binomial_test_vs_chance(
        accuracies,
        base_config.n_transfer_trials
    )
    effect_size = cohens_d(accuracies)

    return StatisticalResult(
        name=experiment_name,
        n_seeds=len(seeds),
        accuracies=accuracies,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=p_value < 0.05,
        effect_size=effect_size
    )


def run_ablation_with_seeds(
    seeds: List[int],
    base_config: NumerosityExperimentConfig,
    verbose: bool = True
) -> Dict[str, StatisticalResult]:
    """Run ablation study with multiple seeds."""
    conditions = ['full_model', 'no_aggregate', 'with_accommodation']
    results = {cond: [] for cond in conditions}

    if verbose:
        print(f"\nRunning ablation study with {len(seeds)} seeds...")

    for i, seed in enumerate(seeds):
        config = NumerosityExperimentConfig(
            n_input=base_config.n_input,
            stimulus_type=base_config.stimulus_type,
            training_numerosities=base_config.training_numerosities,
            transfer_numerosities=base_config.transfer_numerosities,
            n_training_trials=base_config.n_training_trials,
            n_transfer_trials=base_config.n_transfer_trials,
            block_size=base_config.block_size,
            model_seed=seed,
            task_seed=seed,
            stimulus_seed=seed,
            use_accommodation=base_config.use_accommodation,
        )

        ablation_results = run_experiment_6_ablation(config, verbose=False)

        for cond in conditions:
            results[cond].append(ablation_results[cond].transfer_accuracy)

        if verbose:
            print(f"  Seed {seed}: full={ablation_results['full_model'].transfer_accuracy:.1%}, "
                  f"no_agg={ablation_results['no_aggregate'].transfer_accuracy:.1%}, "
                  f"w_accom={ablation_results['with_accommodation'].transfer_accuracy:.1%}")

    # Compute statistics for each condition
    stat_results = {}
    for cond in conditions:
        accuracies = results[cond]
        mean = np.mean(accuracies)
        std = np.std(accuracies, ddof=1)
        ci_lower, ci_upper = compute_confidence_interval(accuracies)
        p_value = binomial_test_vs_chance(accuracies, base_config.n_transfer_trials)
        effect_size = cohens_d(accuracies)

        stat_results[cond] = StatisticalResult(
            name=f"Ablation: {cond}",
            n_seeds=len(seeds),
            accuracies=accuracies,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            significant=p_value < 0.05,
            effect_size=effect_size
        )

    return stat_results


def compare_conditions(
    result1: StatisticalResult,
    result2: StatisticalResult
) -> Dict[str, Any]:
    """Compare two conditions using paired t-test."""
    # Paired t-test (assuming same seeds)
    t_stat, p_value = stats.ttest_rel(result1.accuracies, result2.accuracies)

    # Effect size (Cohen's d for paired samples)
    diff = np.array(result1.accuracies) - np.array(result2.accuracies)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else float('inf')

    return {
        'condition1': result1.name,
        'condition2': result2.name,
        'mean_diff': result1.mean - result2.mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': d
    }


def print_statistical_summary(results: Dict[str, StatisticalResult]):
    """Print a formatted summary of statistical results."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\n{'Experiment':<30} {'Mean':>8} {'95% CI':>16} {'p-value':>10} {'d':>8}")
    print("-" * 80)

    for name, result in results.items():
        ci_str = f"[{result.ci_lower:.1%}, {result.ci_upper:.1%}]"
        sig_marker = "*" if result.significant else ""
        d_str = f"{result.effect_size:.2f}" if result.effect_size != float('inf') else ">10"
        print(f"{result.name:<30} {result.mean:>7.1%} {ci_str:>16} {result.p_value:>9.2e}{sig_marker} {d_str:>8}")

    print("\n* p < 0.05 (significantly above chance)")
    print("d = Cohen's d effect size (0.2=small, 0.5=medium, 0.8=large)")


def print_ablation_comparison(
    ablation_results: Dict[str, StatisticalResult]
):
    """Print comparison between ablation conditions."""
    print("\n" + "=" * 80)
    print("ABLATION COMPARISONS")
    print("=" * 80)

    # Compare full_model vs no_aggregate
    if 'full_model' in ablation_results and 'no_aggregate' in ablation_results:
        comparison = compare_conditions(
            ablation_results['full_model'],
            ablation_results['no_aggregate']
        )
        print(f"\nFull Model vs No Aggregate:")
        print(f"  Mean difference: {comparison['mean_diff']:+.1%}")
        print(f"  t-statistic: {comparison['t_statistic']:.3f}")
        print(f"  p-value: {comparison['p_value']:.4f}")
        print(f"  Significant: {'Yes' if comparison['significant'] else 'No'}")

    # Compare full_model vs with_accommodation
    if 'full_model' in ablation_results and 'with_accommodation' in ablation_results:
        comparison = compare_conditions(
            ablation_results['full_model'],
            ablation_results['with_accommodation']
        )
        print(f"\nFull Model vs With Accommodation:")
        print(f"  Mean difference: {comparison['mean_diff']:+.1%}")
        print(f"  t-statistic: {comparison['t_statistic']:.3f}")
        print(f"  p-value: {comparison['p_value']:.4f}")
        print(f"  Significant: {'Yes' if comparison['significant'] else 'No'}")


def run_full_statistical_analysis(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete statistical analysis with multiple seeds.
    """
    # Generate seeds
    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    if verbose:
        print("=" * 80)
        print(f"MBGN PHASE 2: STATISTICAL ANALYSIS ({n_seeds} seeds)")
        print("=" * 80)
        print(f"Seeds: {seeds}")

    # Base config
    base_config = NumerosityExperimentConfig(
        n_training_trials=100,
        n_transfer_trials=40,
        block_size=20
    )

    all_results = {}

    # Run main experiments
    experiments = [
        (run_experiment_1_baseline, "Exp1: Baseline"),
        (run_experiment_2_novel_counts, "Exp2: Novel Counts"),
        (run_experiment_3_novel_types, "Exp3: Novel Types"),
        (run_experiment_4_full_transfer, "Exp4: Full Transfer"),
        (run_experiment_7_choose_fewer, "Exp7: Choose Fewer"),
        (run_experiment_8_distance_effect, "Exp8: Distance Effect"),
    ]

    for exp_func, exp_name in experiments:
        result = run_experiment_with_seeds(
            exp_func, exp_name, seeds, base_config, verbose
        )
        all_results[exp_name] = result

    # Run ablation study
    ablation_results = run_ablation_with_seeds(seeds, base_config, verbose)
    for cond, result in ablation_results.items():
        all_results[f"Ablation: {cond}"] = result

    # Print summary
    if verbose:
        print_statistical_summary(all_results)
        print_ablation_comparison(ablation_results)

    return {
        'experiments': all_results,
        'ablation': ablation_results,
        'seeds': seeds,
        'config': base_config
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run statistical analysis for MBGN Phase 2 experiments'
    )
    parser.add_argument(
        '--n-seeds', type=int, default=10,
        help='Number of random seeds to use (default: 10)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run with 5 seeds'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    n_seeds = 5 if args.quick else args.n_seeds
    verbose = not args.quiet

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    results = run_full_statistical_analysis(n_seeds=n_seeds, verbose=verbose)

    # Final summary
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    # Check if all main experiments are significant
    main_exps = [
        "Exp1: Baseline", "Exp2: Novel Counts", "Exp3: Novel Types",
        "Exp4: Full Transfer", "Exp7: Choose Fewer"
    ]
    all_significant = all(
        results['experiments'][exp].significant
        for exp in main_exps
    )

    if all_significant:
        print("\nAll main experiments show performance significantly above chance (p < 0.05).")

        # Check effect sizes
        full_transfer = results['experiments']["Exp4: Full Transfer"]
        print(f"\nFull transfer (critical test):")
        print(f"  Mean accuracy: {full_transfer.mean:.1%}")
        print(f"  95% CI: [{full_transfer.ci_lower:.1%}, {full_transfer.ci_upper:.1%}]")

        if full_transfer.mean > 0.70:
            print("\n  STRONG SUCCESS: Transfer accuracy exceeds 70%")
            print("  MBGN demonstrates robust numerosity learning and transfer.")
        elif full_transfer.mean > 0.60:
            print("\n  MODERATE SUCCESS: Transfer accuracy exceeds 60%")
            print("  MBGN can learn numerosity but transfer is less robust than same/different.")
        else:
            print("\n  PARTIAL SUCCESS: Transfer accuracy below 60%")
            print("  Numerosity learning works but may require architectural modifications.")

    return results


if __name__ == '__main__':
    main()
