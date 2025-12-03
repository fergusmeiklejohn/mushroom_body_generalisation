#!/usr/bin/env python3
"""
Phase 3: Finding MBGN's Limits

This script implements harder tests designed to find where MBGN fails.
Phase 2 showed near-perfect performance - here we push until it breaks.

Usage:
    python run_phase3_experiments.py --suite numerosity_limits
    python run_phase3_experiments.py --suite noise_tolerance
    python run_phase3_experiments.py --suite spatial_limits
    python run_phase3_experiments.py --suite architecture_stress
    python run_phase3_experiments.py --all
"""

import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from mbgn import MBGN, MBGNConfig
from mbgn.numerosity_stimuli import NumerosityStimulusGenerator
from mbgn.numerosity_task import (
    NumerosityTask,
    NumerosityTaskType,
    NumerosityTaskRunner,
)


# =============================================================================
# SUITE 1: Numerosity Limits (Weber Fraction)
# =============================================================================

def run_fine_discrimination_test(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test fine-grained numerosity discrimination.

    Can MBGN discriminate 5 vs 6? 10 vs 11?
    Find the Weber fraction.
    """
    print("=" * 70)
    print("PHASE 3a: FINE-GRAINED NUMEROSITY DISCRIMINATION")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    # Test pairs with different ratios
    test_pairs = [
        # (smaller, larger, ratio)
        ([2], [4], 0.50),   # Easy: 2/4 = 0.5
        ([3], [5], 0.60),   # Medium: 3/5 = 0.6
        ([4], [5], 0.80),   # Hard: 4/5 = 0.8
        ([5], [6], 0.83),   # Harder: 5/6 = 0.83
        ([6], [7], 0.86),   # Even harder
        ([7], [8], 0.875),  # Very hard
        ([9], [10], 0.90),  # Extreme
        ([10], [11], 0.91), # Near impossible?
    ]

    results = {}

    for train_nums, test_nums, ratio in test_pairs:
        pair_name = f"{train_nums[0]}_vs_{test_nums[0]}"
        accuracies = []

        # Use both as training AND test (just discrimination)
        all_nums = train_nums + test_nums

        for seed in seeds:
            model_config = MBGNConfig(n_input=50, seed=seed)
            model = MBGN(model_config)

            stim_gen = NumerosityStimulusGenerator(
                n_input=50, stimulus_type='sparse', seed=seed
            )

            task = NumerosityTask(
                numerosities=all_nums,
                stimulus_generator=stim_gen,
                task_type=NumerosityTaskType.CHOOSE_MORE,
                seed=seed
            )

            runner = NumerosityTaskRunner(
                model, task, use_accommodation=False, seed=seed
            )

            # Training
            runner.run_block(50, learn=True, balanced=True)

            # Testing
            test_results = runner.run_block(40, learn=False, balanced=True)
            acc = runner.compute_accuracy(test_results)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[pair_name] = {
            'ratio': ratio,
            'mean': mean_acc,
            'std': std_acc,
            'accuracies': accuracies
        }

        if verbose:
            print(f"  {pair_name} (ratio={ratio:.2f}): {mean_acc:.1%} ± {std_acc:.1%}")

    # Find Weber fraction (where accuracy drops below threshold)
    print("\n" + "-" * 70)
    print("WEBER FRACTION ANALYSIS")
    print("-" * 70)

    threshold = 0.75  # Consider "reliable" above 75%
    for pair_name, data in sorted(results.items(), key=lambda x: x[1]['ratio']):
        status = "✓" if data['mean'] >= threshold else "✗"
        print(f"  {status} Ratio {data['ratio']:.2f}: {data['mean']:.1%}")

    return results


def run_larger_numerosities_test(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test with larger numerosities where Weber's law predicts harder discrimination.
    """
    print("\n" + "=" * 70)
    print("PHASE 3a: LARGER NUMEROSITIES")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    # Test ranges
    test_ranges = [
        ([2, 3, 5, 6], [4, 7], "Small (2-7)"),
        ([8, 10, 12, 14], [9, 11, 13], "Large (8-14)"),
        ([15, 18, 21, 24], [16, 19, 22], "Very Large (15-24)"),
    ]

    results = {}

    for train_nums, transfer_nums, name in test_ranges:
        accuracies = []

        for seed in seeds:
            # Need larger input space for more elements
            n_input = max(max(train_nums + transfer_nums) * 2, 50)

            model_config = MBGNConfig(n_input=n_input, seed=seed)
            model = MBGN(model_config)

            stim_gen = NumerosityStimulusGenerator(
                n_input=n_input, stimulus_type='sparse', seed=seed
            )

            # Training
            train_task = NumerosityTask(
                numerosities=train_nums,
                stimulus_generator=stim_gen,
                task_type=NumerosityTaskType.CHOOSE_MORE,
                seed=seed
            )
            train_runner = NumerosityTaskRunner(
                model, train_task, use_accommodation=False, seed=seed
            )
            train_runner.run_block(100, learn=True, balanced=True)

            # Transfer
            transfer_task = NumerosityTask(
                numerosities=transfer_nums,
                stimulus_generator=stim_gen,
                task_type=NumerosityTaskType.CHOOSE_MORE,
                seed=seed + 1000
            )
            transfer_runner = NumerosityTaskRunner(
                model, transfer_task, use_accommodation=False, seed=seed + 1000
            )
            test_results = transfer_runner.run_block(40, learn=False, balanced=True)
            acc = train_runner.compute_accuracy(test_results)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[name] = {'mean': mean_acc, 'std': std_acc}

        if verbose:
            print(f"  {name}: {mean_acc:.1%} ± {std_acc:.1%}")

    return results


# =============================================================================
# SUITE 2: Noise Tolerance
# =============================================================================

class NoisyNumerosityStimulusGenerator(NumerosityStimulusGenerator):
    """Generator that adds noise to stimuli."""

    def __init__(self, noise_std: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std

    def make_stimulus(self, n_elements: int, element_type: str = 'A',
                      instance_seed=None, **kwargs):
        stim = super().make_stimulus(n_elements, element_type, instance_seed, **kwargs)

        if self.noise_std > 0:
            # Add Gaussian noise to active units
            noise = self.rng.normal(0, self.noise_std, stim.vector.shape)
            stim.vector = np.clip(stim.vector + noise, 0, 2).astype(np.float32)

        return stim


class VariableIntensityGenerator(NumerosityStimulusGenerator):
    """Generator where each element has random intensity."""

    def __init__(self, intensity_range: Tuple[float, float] = (0.3, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.intensity_range = intensity_range

    def make_stimulus(self, n_elements: int, element_type: str = 'A',
                      instance_seed=None, **kwargs):
        if instance_seed is not None:
            rng = np.random.default_rng(instance_seed)
        else:
            rng = self.rng

        vector = np.zeros(self.n_input, dtype=np.float32)
        indices = rng.choice(self.n_input, n_elements, replace=False)
        intensities = rng.uniform(self.intensity_range[0], self.intensity_range[1], n_elements)
        vector[indices] = intensities.astype(np.float32)

        from mbgn.numerosity_stimuli import NumerosityStimulus
        return NumerosityStimulus(
            vector=vector,
            n_elements=n_elements,
            element_type=element_type,
            seed=instance_seed
        )


def run_noise_tolerance_test(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test how noise affects numerosity discrimination.
    """
    print("\n" + "=" * 70)
    print("PHASE 3a: NOISE TOLERANCE")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = {}

    for noise_std in noise_levels:
        accuracies = []

        for seed in seeds:
            model_config = MBGNConfig(n_input=50, seed=seed)
            model = MBGN(model_config)

            stim_gen = NoisyNumerosityStimulusGenerator(
                n_input=50, stimulus_type='sparse', seed=seed, noise_std=noise_std
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
            runner.run_block(100, learn=True, balanced=True)

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
            test_results = transfer_runner.run_block(40, learn=False, balanced=True)
            acc = runner.compute_accuracy(test_results)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[f"noise_{noise_std}"] = {'mean': mean_acc, 'std': std_acc, 'noise_std': noise_std}

        if verbose:
            print(f"  Noise σ={noise_std:.1f}: {mean_acc:.1%} ± {std_acc:.1%}")

    return results


def run_intensity_confound_test(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test if MBGN learns numerosity or just total intensity.

    With variable intensity, 3 bright elements might have same total
    activity as 5 dim elements. Can MBGN still discriminate?
    """
    print("\n" + "=" * 70)
    print("PHASE 3a: INTENSITY vs NUMEROSITY CONFOUND")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    intensity_ranges = [
        ((0.9, 1.0), "Low variance"),
        ((0.5, 1.0), "Medium variance"),
        ((0.2, 1.0), "High variance"),
        ((0.1, 1.0), "Extreme variance"),
    ]

    results = {}

    for intensity_range, name in intensity_ranges:
        accuracies = []

        for seed in seeds:
            model_config = MBGNConfig(n_input=50, seed=seed)
            model = MBGN(model_config)

            stim_gen = VariableIntensityGenerator(
                n_input=50, seed=seed, intensity_range=intensity_range
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
            runner.run_block(100, learn=True, balanced=True)

            # Transfer
            transfer_task = NumerosityTask(
                numerosities=[4, 7],
                stimulus_generator=stim_gen,
                task_type=NumerosityTaskType.CHOOSE_MORE,
                seed=seed + 1000
            )
            transfer_runner = NumerosityTaskRunner(
                model, transfer_task, use_accommodation=False, seed=seed + 1000
            )
            test_results = transfer_runner.run_block(40, learn=False, balanced=True)
            acc = runner.compute_accuracy(test_results)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[name] = {
            'mean': mean_acc,
            'std': std_acc,
            'intensity_range': intensity_range
        }

        if verbose:
            print(f"  {name} {intensity_range}: {mean_acc:.1%} ± {std_acc:.1%}")

    return results


# =============================================================================
# SUITE 3: Spatial Limits (Expected Failures)
# =============================================================================

def run_spatial_task_test(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test spatial tasks that should FAIL because aggregate carries no spatial info.

    Above/Below task: Is the dot above or below center?
    """
    print("\n" + "=" * 70)
    print("PHASE 3b: SPATIAL TASKS (Expected to FAIL)")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    grid_size = 7  # 7x7 = 49 ≈ 50 inputs
    n_input = grid_size * grid_size

    results = {'above_below': [], 'left_right': []}

    for seed in seeds:
        model_config = MBGNConfig(n_input=n_input, seed=seed)
        model = MBGN(model_config)
        rng = np.random.default_rng(seed)

        # Generate above/below stimuli
        def make_above_stimulus():
            vec = np.zeros((grid_size, grid_size), dtype=np.float32)
            # Dot in upper half
            row = rng.integers(0, grid_size // 2)
            col = rng.integers(0, grid_size)
            vec[row, col] = 1.0
            return vec.flatten()

        def make_below_stimulus():
            vec = np.zeros((grid_size, grid_size), dtype=np.float32)
            # Dot in lower half
            row = rng.integers(grid_size // 2 + 1, grid_size)
            col = rng.integers(0, grid_size)
            vec[row, col] = 1.0
            return vec.flatten()

        # Training
        for _ in range(100):
            is_above = rng.random() > 0.5
            stim = make_above_stimulus() if is_above else make_below_stimulus()
            result = model.forward(stim, update_accommodation=False)

            # Correct: "above" = higher aggregate? (doesn't matter, both have 1 dot)
            correct = is_above == result.decision
            reward = 1.0 if correct else -1.0
            model.update_weights(reward, result)

        # Testing
        correct_count = 0
        for _ in range(40):
            is_above = rng.random() > 0.5
            stim = make_above_stimulus() if is_above else make_below_stimulus()
            result = model.forward(stim, update_accommodation=False)
            if is_above == result.decision:
                correct_count += 1

        results['above_below'].append(correct_count / 40)

    mean_above_below = np.mean(results['above_below'])
    std_above_below = np.std(results['above_below'])

    if verbose:
        print(f"  Above/Below: {mean_above_below:.1%} ± {std_above_below:.1%}")
        if mean_above_below < 0.55:
            print("  → EXPECTED FAILURE: Aggregate pathway cannot encode spatial position")
        else:
            print("  → UNEXPECTED: Model somehow learned spatial task!")

    return {
        'above_below': {
            'mean': mean_above_below,
            'std': std_above_below,
            'expected_failure': mean_above_below < 0.55
        }
    }


# =============================================================================
# SUITE 4: Architecture Stress Tests
# =============================================================================

def run_expansion_size_test(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test different expansion layer sizes.
    """
    print("\n" + "=" * 70)
    print("PHASE 3c: EXPANSION LAYER SIZE")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    expansion_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    results = {}

    for n_expansion in expansion_sizes:
        accuracies = []

        for seed in seeds:
            model_config = MBGNConfig(
                n_input=50,
                n_expansion=n_expansion,
                seed=seed
            )
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

            runner.run_block(100, learn=True, balanced=True)

            transfer_task = NumerosityTask(
                numerosities=[4, 7],
                stimulus_generator=stim_gen,
                task_type=NumerosityTaskType.CHOOSE_MORE,
                seed=seed + 1000
            )
            transfer_runner = NumerosityTaskRunner(
                model, transfer_task, use_accommodation=False, seed=seed + 1000
            )
            test_results = transfer_runner.run_block(40, learn=False, balanced=True)
            acc = runner.compute_accuracy(test_results)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[n_expansion] = {'mean': mean_acc, 'std': std_acc}

        if verbose:
            print(f"  N_expansion={n_expansion}: {mean_acc:.1%} ± {std_acc:.1%}")

    return results


def run_sparsity_test(
    n_seeds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test different k-WTA sparsity levels.
    """
    print("\n" + "=" * 70)
    print("PHASE 3c: k-WTA SPARSITY")
    print("=" * 70)

    np.random.seed(42)
    seeds = list(np.random.randint(0, 10000, n_seeds))

    sparsity_fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    results = {}

    for sparsity in sparsity_fractions:
        accuracies = []

        for seed in seeds:
            model_config = MBGNConfig(
                n_input=50,
                n_expansion=2000,
                sparsity_fraction=sparsity,
                seed=seed
            )
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

            runner.run_block(100, learn=True, balanced=True)

            transfer_task = NumerosityTask(
                numerosities=[4, 7],
                stimulus_generator=stim_gen,
                task_type=NumerosityTaskType.CHOOSE_MORE,
                seed=seed + 1000
            )
            transfer_runner = NumerosityTaskRunner(
                model, transfer_task, use_accommodation=False, seed=seed + 1000
            )
            test_results = transfer_runner.run_block(40, learn=False, balanced=True)
            acc = runner.compute_accuracy(test_results)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        k_value = int(2000 * sparsity)
        results[sparsity] = {'mean': mean_acc, 'std': std_acc, 'k': k_value}

        if verbose:
            print(f"  Sparsity={sparsity:.0%} (k={k_value}): {mean_acc:.1%} ± {std_acc:.1%}")

    return results


# =============================================================================
# Main
# =============================================================================

def run_all_phase3(n_seeds: int = 10, verbose: bool = True):
    """Run all Phase 3 experiments."""
    all_results = {}

    # Suite 1: Numerosity limits
    all_results['fine_discrimination'] = run_fine_discrimination_test(n_seeds, verbose)
    all_results['larger_numerosities'] = run_larger_numerosities_test(n_seeds, verbose)

    # Suite 2: Noise tolerance
    all_results['noise_tolerance'] = run_noise_tolerance_test(n_seeds, verbose)
    all_results['intensity_confound'] = run_intensity_confound_test(n_seeds, verbose)

    # Suite 3: Spatial limits
    all_results['spatial_tasks'] = run_spatial_task_test(n_seeds, verbose)

    # Suite 4: Architecture
    all_results['expansion_size'] = run_expansion_size_test(n_seeds, verbose)
    all_results['sparsity'] = run_sparsity_test(n_seeds, verbose)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY: FAILURE MODES IDENTIFIED")
    print("=" * 70)

    print("\n1. NUMEROSITY LIMITS:")
    for pair, data in all_results['fine_discrimination'].items():
        if data['mean'] < 0.75:
            print(f"   ✗ {pair}: {data['mean']:.1%} (Weber fraction exceeded)")

    print("\n2. NOISE TOLERANCE:")
    for name, data in all_results['noise_tolerance'].items():
        if data['mean'] < 0.75:
            print(f"   ✗ {name}: {data['mean']:.1%}")

    print("\n3. SPATIAL TASKS:")
    spatial = all_results['spatial_tasks']['above_below']
    if spatial['expected_failure']:
        print(f"   ✗ Above/Below: {spatial['mean']:.1%} (confirmed spatial limitation)")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run Phase 3 experiments to find MBGN limits'
    )
    parser.add_argument(
        '--suite', type=str,
        choices=['numerosity_limits', 'noise_tolerance', 'spatial_limits',
                 'architecture_stress', 'all'],
        default='all',
        help='Which test suite to run'
    )
    parser.add_argument(
        '--n-seeds', type=int, default=10,
        help='Number of seeds per test'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run with 3 seeds'
    )

    args = parser.parse_args()
    n_seeds = 3 if args.quick else args.n_seeds

    if args.suite == 'all':
        run_all_phase3(n_seeds)
    elif args.suite == 'numerosity_limits':
        run_fine_discrimination_test(n_seeds)
        run_larger_numerosities_test(n_seeds)
    elif args.suite == 'noise_tolerance':
        run_noise_tolerance_test(n_seeds)
        run_intensity_confound_test(n_seeds)
    elif args.suite == 'spatial_limits':
        run_spatial_task_test(n_seeds)
    elif args.suite == 'architecture_stress':
        run_expansion_size_test(n_seeds)
        run_sparsity_test(n_seeds)


if __name__ == '__main__':
    main()
