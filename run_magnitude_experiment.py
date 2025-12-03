#!/usr/bin/env python3
"""
Magnitude discrimination experiments for MBGN.

Tests whether MBGN can learn to compare stimuli by intensity/brightness,
extending the aggregate pathway beyond numerosity.

Usage:
    python run_magnitude_experiment.py              # Run all experiments
    python run_magnitude_experiment.py --verify     # Just verify signal
    python run_magnitude_experiment.py --quick      # Quick test
"""

import argparse
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from mbgn import MBGN, MBGNConfig
from mbgn.magnitude_stimuli import (
    MagnitudeStimulusGenerator,
    verify_magnitude_correlation
)
from mbgn.magnitude_task import (
    MagnitudeTask,
    MagnitudeTaskType,
    MagnitudeTaskRunner,
    MagnitudeTrialResult
)


@dataclass
class MagnitudeExperimentResult:
    """Results from a magnitude experiment."""
    name: str
    training_accuracy: float
    transfer_accuracy: float
    training_by_block: List[float]
    magnitude_effect: Dict[float, float]


def verify_magnitude_signal(verbose: bool = True) -> Dict:
    """Verify that aggregate activity correlates with magnitude."""
    print("=" * 60)
    print("MAGNITUDE SIGNAL VERIFICATION")
    print("=" * 60)

    magnitudes = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Test at stimulus level
    print("\n1. Stimulus-level correlation:")
    gen = MagnitudeStimulusGenerator(
        n_input=50,
        n_active=10,
        magnitude_type='intensity',
        seed=42
    )

    verification = verify_magnitude_correlation(
        gen, magnitudes, n_samples=100
    )
    print(f"   Correlation: {verification['correlation']:.4f}")
    print(f"   Mean activity by magnitude:")
    for m in magnitudes:
        mean = verification['mean_by_magnitude'][m]
        std = verification['std_by_magnitude'][m]
        print(f"     M={m:.1f}: {mean:.2f} +/- {std:.2f}")

    # Test through model
    print("\n2. Expansion layer correlation:")
    model_config = MBGNConfig(n_input=50, seed=42)
    model = MBGN(model_config)

    model_activities = {m: [] for m in magnitudes}
    for mag in magnitudes:
        for _ in range(100):
            stim = gen.generate_random(mag)
            result = model.forward(stim.pattern, update_accommodation=False)
            model_activities[mag].append(result.aggregate_activity)

    # Compute correlation
    all_mags = []
    all_acts = []
    for mag in magnitudes:
        all_mags.extend([mag] * len(model_activities[mag]))
        all_acts.extend(model_activities[mag])

    model_correlation = np.corrcoef(all_mags, all_acts)[0, 1]
    print(f"   Correlation: {model_correlation:.4f}")
    print(f"   Mean aggregate by magnitude:")
    for m in magnitudes:
        mean = np.mean(model_activities[m])
        std = np.std(model_activities[m])
        print(f"     M={m:.1f}: {mean:.1f} +/- {std:.1f}")

    # Check monotonicity
    means = [np.mean(model_activities[m]) for m in magnitudes]
    is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
    print(f"   Is monotonic: {'Yes' if is_monotonic else 'No'}")

    print("\n3. Assessment:")
    if model_correlation > 0.9:
        print("   STRONG signal - magnitude experiments should work well")
    elif model_correlation > 0.7:
        print("   GOOD signal - magnitude experiments should work")
    else:
        print("   WEAK signal - may need modifications")

    return {
        'stimulus_correlation': verification['correlation'],
        'model_correlation': model_correlation,
        'is_monotonic': is_monotonic
    }


def run_baseline_experiment(
    n_training: int = 100,
    n_transfer: int = 40,
    seed: int = 42,
    verbose: bool = True
) -> MagnitudeExperimentResult:
    """
    Experiment 1: Magnitude Baseline

    Can MBGN learn to choose the brighter stimulus?
    """
    if verbose:
        print("=" * 60)
        print("Magnitude Experiment 1: Baseline")
        print("=" * 60)

    # Training magnitudes
    training_mags = [0.2, 0.4, 0.6, 0.8]
    transfer_mags = [0.3, 0.5, 0.7]  # Novel magnitudes

    # Create generator and model
    gen = MagnitudeStimulusGenerator(
        n_input=50, n_active=10, magnitude_type='intensity', seed=seed
    )
    model_config = MBGNConfig(n_input=50, seed=seed)
    model = MBGN(model_config)

    # Create task
    task = MagnitudeTask(
        magnitudes=training_mags,
        stimulus_generator=gen,
        task_type=MagnitudeTaskType.CHOOSE_BRIGHTER,
        seed=seed
    )

    runner = MagnitudeTaskRunner(
        model, task, use_accommodation=False, seed=seed
    )

    # Training
    training_results = []
    accuracy_by_block = []
    block_size = 20

    for block_start in range(0, n_training, block_size):
        block_results = runner.run_block(
            min(block_size, n_training - block_start),
            learn=True,
            stimulus_type='A',
            use_random_patterns=True
        )
        training_results.extend(block_results)
        block_acc = runner.compute_accuracy(block_results)
        accuracy_by_block.append(block_acc)
        if verbose:
            print(f"  Block {len(accuracy_by_block)}: {block_acc:.1%}")

    training_accuracy = runner.compute_accuracy(training_results)

    # Transfer to novel magnitudes
    transfer_task = MagnitudeTask(
        magnitudes=transfer_mags,
        stimulus_generator=gen,
        task_type=MagnitudeTaskType.CHOOSE_BRIGHTER,
        seed=seed + 1000
    )
    transfer_runner = MagnitudeTaskRunner(
        model, transfer_task, use_accommodation=False, seed=seed + 1000
    )

    transfer_results = transfer_runner.run_block(
        n_transfer, learn=False, stimulus_type='B', use_random_patterns=True
    )
    transfer_accuracy = runner.compute_accuracy(transfer_results)

    # Magnitude effect
    all_results = training_results + transfer_results
    mag_effect = runner.compute_accuracy_by_diff(all_results)

    if verbose:
        print(f"\nTraining accuracy: {training_accuracy:.1%}")
        print(f"Transfer accuracy: {transfer_accuracy:.1%}")
        print(f"\nAccuracy by magnitude difference:")
        for diff in sorted(mag_effect.keys()):
            print(f"  Diff {diff:.2f}: {mag_effect[diff]:.1%}")

    return MagnitudeExperimentResult(
        name="Baseline",
        training_accuracy=training_accuracy,
        transfer_accuracy=transfer_accuracy,
        training_by_block=accuracy_by_block,
        magnitude_effect=mag_effect
    )


def run_full_transfer_experiment(
    n_training: int = 100,
    n_transfer: int = 40,
    seed: int = 42,
    verbose: bool = True
) -> MagnitudeExperimentResult:
    """
    Experiment 2: Full Transfer

    Transfer to both novel magnitudes AND novel patterns.
    """
    if verbose:
        print("=" * 60)
        print("Magnitude Experiment 2: Full Transfer")
        print("=" * 60)

    training_mags = [0.2, 0.4, 0.6, 0.8]
    transfer_mags = [0.1, 0.3, 0.5, 0.7, 0.9]

    gen = MagnitudeStimulusGenerator(
        n_input=50, n_active=10, magnitude_type='intensity', seed=seed
    )
    model_config = MBGNConfig(n_input=50, seed=seed)
    model = MBGN(model_config)

    # Training with type A patterns
    task = MagnitudeTask(
        magnitudes=training_mags,
        stimulus_generator=gen,
        task_type=MagnitudeTaskType.CHOOSE_BRIGHTER,
        seed=seed
    )

    runner = MagnitudeTaskRunner(
        model, task, use_accommodation=False, seed=seed
    )

    training_results = []
    accuracy_by_block = []
    block_size = 20

    for block_start in range(0, n_training, block_size):
        block_results = runner.run_block(
            min(block_size, n_training - block_start),
            learn=True,
            stimulus_type='A'
        )
        training_results.extend(block_results)
        block_acc = runner.compute_accuracy(block_results)
        accuracy_by_block.append(block_acc)
        if verbose:
            print(f"  Block {len(accuracy_by_block)}: {block_acc:.1%}")

    training_accuracy = runner.compute_accuracy(training_results)

    # Transfer with type B patterns (novel) and novel magnitudes
    gen.clear_cache()  # New patterns for type B
    transfer_task = MagnitudeTask(
        magnitudes=transfer_mags,
        stimulus_generator=gen,
        task_type=MagnitudeTaskType.CHOOSE_BRIGHTER,
        seed=seed + 1000
    )
    transfer_runner = MagnitudeTaskRunner(
        model, transfer_task, use_accommodation=False, seed=seed + 1000
    )

    transfer_results = transfer_runner.run_block(
        n_transfer, learn=False, stimulus_type='B', use_random_patterns=True
    )
    transfer_accuracy = runner.compute_accuracy(transfer_results)

    mag_effect = runner.compute_accuracy_by_diff(transfer_results)

    if verbose:
        print(f"\nTraining accuracy: {training_accuracy:.1%}")
        print(f"Full Transfer accuracy: {transfer_accuracy:.1%}")

    return MagnitudeExperimentResult(
        name="Full Transfer",
        training_accuracy=training_accuracy,
        transfer_accuracy=transfer_accuracy,
        training_by_block=accuracy_by_block,
        magnitude_effect=mag_effect
    )


def run_all_experiments(
    n_seeds: int = 5,
    verbose: bool = True
) -> Dict[str, List[MagnitudeExperimentResult]]:
    """Run all magnitude experiments with multiple seeds."""
    print("=" * 70)
    print(f"MAGNITUDE EXPERIMENTS ({n_seeds} seeds)")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))
    print(f"Seeds: {seeds}")

    all_results = {
        'baseline': [],
        'full_transfer': []
    }

    for seed in seeds:
        if verbose:
            print(f"\n--- Seed {seed} ---")

        baseline = run_baseline_experiment(seed=seed, verbose=False)
        all_results['baseline'].append(baseline)

        full_transfer = run_full_transfer_experiment(seed=seed, verbose=False)
        all_results['full_transfer'].append(full_transfer)

        if verbose:
            print(f"  Baseline: {baseline.transfer_accuracy:.1%}")
            print(f"  Full Transfer: {full_transfer.transfer_accuracy:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for exp_name, results in all_results.items():
        transfer_accs = [r.transfer_accuracy for r in results]
        mean = np.mean(transfer_accs)
        std = np.std(transfer_accs)
        print(f"{exp_name}: {mean:.1%} +/- {std:.1%}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run MBGN magnitude discrimination experiments'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Just verify magnitude signal'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test with 3 seeds'
    )
    parser.add_argument(
        '--n-seeds', type=int, default=5,
        help='Number of seeds (default: 5)'
    )

    args = parser.parse_args()

    if args.verify:
        verify_magnitude_signal()
        return

    n_seeds = 3 if args.quick else args.n_seeds
    results = run_all_experiments(n_seeds=n_seeds)

    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    baseline_mean = np.mean([r.transfer_accuracy for r in results['baseline']])
    transfer_mean = np.mean([r.transfer_accuracy for r in results['full_transfer']])

    if transfer_mean > 0.70:
        print("\nSTRONG SUCCESS: Magnitude discrimination transfers well.")
        print("This confirms MBGN's aggregate pathway supports multiple")
        print("relational concepts: same/different, numerosity, AND magnitude.")
    elif transfer_mean > 0.60:
        print("\nMODERATE SUCCESS: Magnitude discrimination works.")
        print("Transfer is present but may be less robust than numerosity.")
    else:
        print("\nWEAK RESULT: Magnitude transfer is limited.")
        print("The aggregate pathway may not preserve fine-grained intensity differences.")


if __name__ == '__main__':
    main()
