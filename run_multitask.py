#!/usr/bin/env python3
"""
Multi-task learning experiments for MBGN.

Tests whether a single model can learn BOTH same/different AND numerosity
relations. This is a key test for whether the aggregate pathway provides
a general substrate for relational learning.

Key questions:
1. Can one model learn both tasks?
2. Does learning one task help or hurt the other?
3. What is the cost of multi-task learning?

Usage:
    python run_multitask.py              # Run multi-task experiment
    python run_multitask.py --n-seeds 5  # Use 5 seeds
    python run_multitask.py --quick      # Quick test
"""

import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from mbgn import MBGN, MBGNConfig
from mbgn.stimuli import create_experiment_stimuli
from mbgn.task import TaskType, ExperimentRunner as SDExperimentRunner
from mbgn.numerosity_stimuli import NumerosityStimulusGenerator
from mbgn.numerosity_task import (
    NumerosityTask,
    NumerosityTaskType,
    NumerosityTaskRunner,
    NumerosityTrialResult
)


@dataclass
class MultiTaskResult:
    """Results from multi-task learning experiment."""
    # Single-task baselines
    sd_only_transfer: float
    num_only_transfer: float

    # Multi-task performance
    sd_after_num_transfer: float
    num_after_sd_transfer: float
    interleaved_sd_transfer: float
    interleaved_num_transfer: float


def run_single_task_sd(seed: int, n_training: int = 100, n_transfer: int = 40) -> float:
    """Run same/different task alone."""
    model_config = MBGNConfig(n_input=50, seed=seed)
    model = MBGN(model_config)

    training_stimuli, transfer_stimuli = create_experiment_stimuli(
        n_training=4, n_transfer=4, n_dims=50, stimulus_type='binary', seed=seed
    )

    model.calibrate_baseline(training_stimuli)
    model.set_aggregate_bias('DMTS')

    runner = SDExperimentRunner(
        model=model,
        training_stimuli=training_stimuli,
        transfer_stimuli=transfer_stimuli,
        task_type=TaskType.DMTS,
        seed=seed
    )

    results = runner.run_experiment(
        n_familiarization=10,
        n_training=n_training,
        n_transfer=n_transfer,
        block_size=20
    )

    return results['transfer_accuracy']


def run_single_task_num(seed: int, n_training: int = 100, n_transfer: int = 40) -> float:
    """Run numerosity task alone."""
    model_config = MBGNConfig(n_input=50, seed=seed)
    model = MBGN(model_config)

    stim_gen = NumerosityStimulusGenerator(
        n_input=50, stimulus_type='sparse', seed=seed
    )

    task = NumerosityTask(
        numerosities=[2, 3, 5, 6],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed
    )

    runner = NumerosityTaskRunner(
        model, task, use_accommodation=False, seed=seed
    )

    # Training
    training_results = runner.run_block(n_training, learn=True, balanced=True)

    # Transfer to novel numerosities
    transfer_task = NumerosityTask(
        numerosities=[4, 7],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed + 1000
    )
    transfer_runner = NumerosityTaskRunner(
        model, transfer_task, use_accommodation=False, seed=seed + 1000
    )

    transfer_results = transfer_runner.run_block(n_transfer, learn=False, balanced=True)
    return runner.compute_accuracy(transfer_results)


def run_sequential_sd_then_num(
    seed: int,
    n_training: int = 100,
    n_transfer: int = 40
) -> Tuple[float, float]:
    """
    Train on same/different first, then numerosity.

    Returns:
        (sd_transfer_accuracy, num_transfer_accuracy)
    """
    model_config = MBGNConfig(n_input=50, seed=seed)
    model = MBGN(model_config)

    # --- Phase 1: Same/Different ---
    training_stimuli, transfer_stimuli = create_experiment_stimuli(
        n_training=4, n_transfer=4, n_dims=50, stimulus_type='binary', seed=seed
    )

    model.calibrate_baseline(training_stimuli)
    model.set_aggregate_bias('DMTS')

    sd_runner = SDExperimentRunner(
        model=model,
        training_stimuli=training_stimuli,
        transfer_stimuli=transfer_stimuli,
        task_type=TaskType.DMTS,
        seed=seed
    )

    sd_results = sd_runner.run_experiment(
        n_familiarization=10,
        n_training=n_training,
        n_transfer=n_transfer,
        block_size=20
    )
    sd_transfer = sd_results['transfer_accuracy']

    # --- Phase 2: Numerosity (same model) ---
    stim_gen = NumerosityStimulusGenerator(
        n_input=50, stimulus_type='sparse', seed=seed
    )

    num_task = NumerosityTask(
        numerosities=[2, 3, 5, 6],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed
    )

    num_runner = NumerosityTaskRunner(
        model, num_task, use_accommodation=False, seed=seed
    )

    # Train numerosity
    num_runner.run_block(n_training, learn=True, balanced=True)

    # Transfer numerosity
    transfer_task = NumerosityTask(
        numerosities=[4, 7],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed + 1000
    )
    transfer_num_runner = NumerosityTaskRunner(
        model, transfer_task, use_accommodation=False, seed=seed + 1000
    )
    num_transfer_results = transfer_num_runner.run_block(n_transfer, learn=False, balanced=True)
    num_transfer = num_runner.compute_accuracy(num_transfer_results)

    return sd_transfer, num_transfer


def run_sequential_num_then_sd(
    seed: int,
    n_training: int = 100,
    n_transfer: int = 40
) -> Tuple[float, float]:
    """
    Train on numerosity first, then same/different.

    Returns:
        (num_transfer_accuracy, sd_transfer_accuracy)
    """
    model_config = MBGNConfig(n_input=50, seed=seed)
    model = MBGN(model_config)

    # --- Phase 1: Numerosity ---
    stim_gen = NumerosityStimulusGenerator(
        n_input=50, stimulus_type='sparse', seed=seed
    )

    num_task = NumerosityTask(
        numerosities=[2, 3, 5, 6],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed
    )

    num_runner = NumerosityTaskRunner(
        model, num_task, use_accommodation=False, seed=seed
    )

    # Train numerosity
    num_runner.run_block(n_training, learn=True, balanced=True)

    # Test numerosity transfer
    transfer_task = NumerosityTask(
        numerosities=[4, 7],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed + 1000
    )
    transfer_num_runner = NumerosityTaskRunner(
        model, transfer_task, use_accommodation=False, seed=seed + 1000
    )
    num_transfer_results = transfer_num_runner.run_block(n_transfer, learn=False, balanced=True)
    num_transfer = num_runner.compute_accuracy(num_transfer_results)

    # --- Phase 2: Same/Different (same model) ---
    training_stimuli, transfer_stimuli = create_experiment_stimuli(
        n_training=4, n_transfer=4, n_dims=50, stimulus_type='binary', seed=seed
    )

    model.calibrate_baseline(training_stimuli)
    model.set_aggregate_bias('DMTS')

    sd_runner = SDExperimentRunner(
        model=model,
        training_stimuli=training_stimuli,
        transfer_stimuli=transfer_stimuli,
        task_type=TaskType.DMTS,
        seed=seed
    )

    sd_results = sd_runner.run_experiment(
        n_familiarization=10,
        n_training=n_training,
        n_transfer=n_transfer,
        block_size=20
    )
    sd_transfer = sd_results['transfer_accuracy']

    return num_transfer, sd_transfer


def run_interleaved_training(
    seed: int,
    n_training: int = 100,
    n_transfer: int = 40
) -> Tuple[float, float]:
    """
    Interleave training on both tasks.

    Alternates between same/different and numerosity trials.

    Returns:
        (sd_transfer_accuracy, num_transfer_accuracy)
    """
    model_config = MBGNConfig(n_input=50, seed=seed)
    model = MBGN(model_config)

    # Setup same/different
    training_stimuli, transfer_stimuli = create_experiment_stimuli(
        n_training=4, n_transfer=4, n_dims=50, stimulus_type='binary', seed=seed
    )
    model.calibrate_baseline(training_stimuli)

    sd_runner = SDExperimentRunner(
        model=model,
        training_stimuli=training_stimuli,
        transfer_stimuli=transfer_stimuli,
        task_type=TaskType.DMTS,
        seed=seed
    )

    # Setup numerosity
    stim_gen = NumerosityStimulusGenerator(
        n_input=50, stimulus_type='sparse', seed=seed
    )

    num_task = NumerosityTask(
        numerosities=[2, 3, 5, 6],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed
    )

    num_runner = NumerosityTaskRunner(
        model, num_task, use_accommodation=False, seed=seed
    )

    # Interleaved training
    rng = np.random.RandomState(seed)
    block_size = 10

    for block in range(n_training // block_size):
        # Alternate which task goes first
        if block % 2 == 0:
            # Same/different block (need to enable accommodation)
            model.set_aggregate_bias('DMTS')
            sd_runner._run_training_block(block_size // 2)

            # Numerosity block (disable accommodation effect by resetting)
            model.reset_accommodation()
            num_runner.run_block(block_size // 2, learn=True, balanced=True)
        else:
            # Numerosity first
            model.reset_accommodation()
            num_runner.run_block(block_size // 2, learn=True, balanced=True)

            # Same/different
            model.set_aggregate_bias('DMTS')
            sd_runner._run_training_block(block_size // 2)

    # Test same/different transfer
    model.set_aggregate_bias('DMTS')
    sd_transfer_results = sd_runner._run_transfer_trials(n_transfer)
    sd_transfer = sum(1 for r in sd_transfer_results if r.correct) / len(sd_transfer_results)

    # Test numerosity transfer
    model.reset_accommodation()
    transfer_task = NumerosityTask(
        numerosities=[4, 7],
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=seed + 1000
    )
    transfer_num_runner = NumerosityTaskRunner(
        model, transfer_task, use_accommodation=False, seed=seed + 1000
    )
    num_transfer_results = transfer_num_runner.run_block(n_transfer, learn=False, balanced=True)
    num_transfer = num_runner.compute_accuracy(num_transfer_results)

    return sd_transfer, num_transfer


def run_multitask_experiment(n_seeds: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """Run full multi-task learning experiment."""
    print("=" * 70)
    print(f"MULTI-TASK LEARNING EXPERIMENT ({n_seeds} seeds)")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))
    print(f"Seeds: {seeds}")

    results = {
        'sd_only': [],
        'num_only': [],
        'sd_then_num_sd': [],
        'sd_then_num_num': [],
        'num_then_sd_num': [],
        'num_then_sd_sd': []
    }

    for seed in seeds:
        if verbose:
            print(f"\n--- Seed {seed} ---")

        # Single-task baselines
        sd_only = run_single_task_sd(seed)
        num_only = run_single_task_num(seed)
        results['sd_only'].append(sd_only)
        results['num_only'].append(num_only)

        if verbose:
            print(f"  SD only: {sd_only:.1%}")
            print(f"  Num only: {num_only:.1%}")

        # Sequential: SD then Num
        sd_transfer, num_transfer = run_sequential_sd_then_num(seed)
        results['sd_then_num_sd'].append(sd_transfer)
        results['sd_then_num_num'].append(num_transfer)

        if verbose:
            print(f"  SD->Num: SD={sd_transfer:.1%}, Num={num_transfer:.1%}")

        # Sequential: Num then SD
        num_transfer2, sd_transfer2 = run_sequential_num_then_sd(seed)
        results['num_then_sd_num'].append(num_transfer2)
        results['num_then_sd_sd'].append(sd_transfer2)

        if verbose:
            print(f"  Num->SD: Num={num_transfer2:.1%}, SD={sd_transfer2:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Condition':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 50)

    for key, values in results.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{key:<25} {mean:>9.1%} {std:>9.1%}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Does learning num after SD hurt SD?
    sd_drop = np.mean(results['sd_only']) - np.mean(results['sd_then_num_sd'])
    print(f"\nSD degradation after Num training: {sd_drop:+.1%}")

    # Does learning SD after Num hurt Num?
    num_drop = np.mean(results['num_only']) - np.mean(results['num_then_sd_num'])
    print(f"Num degradation after SD training: {num_drop:+.1%}")

    # Multi-task performance (correct averaging)
    combined_sd = (np.mean(results['sd_then_num_sd']) + np.mean(results['num_then_sd_sd'])) / 2
    combined_num = (np.mean(results['sd_then_num_num']) + np.mean(results['num_then_sd_num'])) / 2
    print(f"\nAverage multi-task SD: {combined_sd:.1%}")
    print(f"Average multi-task Num: {combined_num:.1%}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if abs(sd_drop) < 0.05 and abs(num_drop) < 0.05:
        print("\nSUCCESS: No catastrophic forgetting!")
        print("The model can learn both relational concepts without interference.")
        print("SD and Num maintain their single-task performance levels.")
    elif combined_sd > 0.60 and combined_num > 0.60:
        print("\nSUCCESS: Model can learn both relational concepts!")
        print("The aggregate pathway provides a general substrate for")
        print("multiple relational rules.")
    elif combined_num > 0.60:
        print("\nPARTIAL: Numerosity is retained but SD degrades.")
        print("Tasks may compete for the same pathway resources.")
    else:
        print("\nFAILURE: Significant interference between tasks.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-task learning experiments'
    )
    parser.add_argument(
        '--n-seeds', type=int, default=10,
        help='Number of seeds (default: 10)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test with 3 seeds'
    )

    args = parser.parse_args()
    n_seeds = 3 if args.quick else args.n_seeds

    results = run_multitask_experiment(n_seeds=n_seeds)


if __name__ == '__main__':
    main()
