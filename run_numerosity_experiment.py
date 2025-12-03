#!/usr/bin/env python3
"""
Run numerosity experiments for MBGN Phase 2.

This script provides a command-line interface to run the numerosity
experiments specified in mbgn_phase2_numerosity.md.

Usage:
    python run_numerosity_experiment.py                    # Run all experiments
    python run_numerosity_experiment.py --exp 1            # Run experiment 1 only
    python run_numerosity_experiment.py --exp 1 2 4        # Run experiments 1, 2, and 4
    python run_numerosity_experiment.py --verify           # Verify numerosity signal
    python run_numerosity_experiment.py --quick            # Quick test (fewer trials)
    python run_numerosity_experiment.py --seed 123         # Custom seed
"""

import argparse
import sys
import numpy as np

from mbgn import MBGN, MBGNConfig
from mbgn.numerosity_stimuli import NumerosityStimulusGenerator, verify_numerosity_correlation
from mbgn.numerosity_experiments import (
    NumerosityExperimentConfig,
    run_experiment_1_baseline,
    run_experiment_2_novel_counts,
    run_experiment_3_novel_types,
    run_experiment_4_full_transfer,
    run_experiment_5_comparison,
    run_experiment_6_ablation,
    run_experiment_7_choose_fewer,
    run_experiment_8_distance_effect,
    run_all_experiments,
)
from mbgn.analysis import format_numerosity_results_table


def verify_numerosity_signal(config: NumerosityExperimentConfig, verbose: bool = True):
    """
    Verify that aggregate activity correlates with numerosity.

    This is a diagnostic check to ensure the random projection
    preserves the numerosity signal before running experiments.
    """
    print("=" * 60)
    print("NUMEROSITY SIGNAL VERIFICATION")
    print("=" * 60)

    # Create stimulus generator
    stim_gen = NumerosityStimulusGenerator(
        n_input=config.n_input,
        stimulus_type=config.stimulus_type,
        seed=config.stimulus_seed
    )

    # Test numerosities
    all_numerosities = sorted(set(
        config.training_numerosities + config.transfer_numerosities
    ))

    # Verify at stimulus level
    print("\n1. Stimulus-level correlation:")
    stim_verification = verify_numerosity_correlation(
        stim_gen,
        numerosities=all_numerosities,
        n_samples=100
    )
    print(f"   Correlation: {stim_verification['correlation']:.4f}")
    print(f"   Mean activity by numerosity:")
    for n in all_numerosities:
        mean = stim_verification['mean_by_numerosity'][n]
        std = stim_verification['std_by_numerosity'][n]
        print(f"     N={n}: {mean:.2f} +/- {std:.2f}")

    # Create model and verify at expansion layer
    print("\n2. Expansion layer correlation:")
    model_config = MBGNConfig(n_input=config.n_input, seed=config.model_seed)
    model = MBGN(model_config)

    model_verification = model.verify_numerosity_signal(
        numerosities=all_numerosities,
        stimulus_generator=stim_gen,
        n_samples=100
    )
    print(f"   Correlation: {model_verification['correlation']:.4f}")
    print(f"   Is monotonic: {'Yes' if model_verification['is_monotonic'] else 'No'}")
    print(f"   Mean aggregate by numerosity:")
    for n in all_numerosities:
        mean = model_verification['mean_by_numerosity'][n]
        std = model_verification['std_by_numerosity'][n]
        print(f"     N={n}: {mean:.2f} +/- {std:.2f}")

    # Assessment
    print("\n3. Assessment:")
    if model_verification['correlation'] > 0.9:
        print("   STRONG signal - numerosity experiments should work well")
    elif model_verification['correlation'] > 0.7:
        print("   GOOD signal - numerosity experiments should work")
    elif model_verification['correlation'] > 0.5:
        print("   MODERATE signal - experiments may show partial success")
    else:
        print("   WEAK signal - consider architectural modifications")

    return {
        'stimulus_verification': stim_verification,
        'model_verification': model_verification
    }


def run_quick_test(verbose: bool = True):
    """
    Run a quick test to verify the numerosity pipeline works.
    """
    print("=" * 60)
    print("QUICK NUMEROSITY TEST")
    print("=" * 60)

    config = NumerosityExperimentConfig(
        n_training_trials=30,
        n_transfer_trials=20,
        block_size=10
    )

    # Verify signal first
    verify_numerosity_signal(config, verbose=False)

    # Run baseline experiment
    print("\nRunning quick baseline experiment...")
    result = run_experiment_1_baseline(config, verbose=verbose)

    # Summary
    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    print(f"Training accuracy: {result.training_accuracy:.1%}")
    print(f"Test accuracy: {result.transfer_accuracy:.1%}")

    if result.transfer_accuracy > 0.60:
        print("\nStatus: PASS - Numerosity learning is working")
    else:
        print("\nStatus: MARGINAL - May need more training or tuning")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run MBGN Phase 2 Numerosity Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_numerosity_experiment.py                    # Run all experiments
  python run_numerosity_experiment.py --exp 1            # Run experiment 1 only
  python run_numerosity_experiment.py --exp 1 2 4        # Run experiments 1, 2, and 4
  python run_numerosity_experiment.py --verify           # Verify numerosity signal
  python run_numerosity_experiment.py --quick            # Quick validation test

Experiments:
  1: Baseline - Can MBGN learn numerosity?
  2: Novel Counts - Transfer to unseen numerosities
  3: Novel Types - Transfer to new element types
  4: Full Transfer - Both novel counts AND types (critical test)
  5: Comparison - Compare with same/different task
  6: Ablation - Test component contributions
  7: Choose Fewer - Test rule inversion
  8: Distance Effect - Test numerical distance effect
        """
    )

    parser.add_argument(
        '--exp', type=int, nargs='+',
        help='Experiment number(s) to run (1-8). Omit to run all.'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Just verify numerosity signal, do not run experiments'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick validation test with fewer trials'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--n-training', type=int, default=100,
        help='Number of training trials (default: 100)'
    )
    parser.add_argument(
        '--n-transfer', type=int, default=40,
        help='Number of transfer trials (default: 40)'
    )
    parser.add_argument(
        '--stimulus-type', type=str, default='sparse',
        choices=['sparse', 'binary', 'gaussian_2d'],
        help='Type of stimuli to use (default: sparse)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Create config
    config = NumerosityExperimentConfig(
        model_seed=args.seed,
        task_seed=args.seed,
        stimulus_seed=args.seed,
        n_training_trials=args.n_training,
        n_transfer_trials=args.n_transfer,
        stimulus_type=args.stimulus_type
    )

    verbose = not args.quiet

    # Handle special modes
    if args.verify:
        verify_numerosity_signal(config, verbose=verbose)
        return

    if args.quick:
        run_quick_test(verbose=verbose)
        return

    # Map experiment numbers to functions
    experiment_functions = {
        1: run_experiment_1_baseline,
        2: run_experiment_2_novel_counts,
        3: run_experiment_3_novel_types,
        4: run_experiment_4_full_transfer,
        5: run_experiment_5_comparison,
        6: run_experiment_6_ablation,
        7: run_experiment_7_choose_fewer,
        8: run_experiment_8_distance_effect,
    }

    # Run selected experiments or all
    if args.exp:
        # Run specific experiments
        results = {}
        for exp_num in args.exp:
            if exp_num not in experiment_functions:
                print(f"Unknown experiment number: {exp_num}")
                continue

            exp_func = experiment_functions[exp_num]
            try:
                result = exp_func(config, verbose=verbose)
                results[f'exp{exp_num}'] = result
            except Exception as e:
                print(f"Error running experiment {exp_num}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        # Print summary
        if results and verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            for exp_name, result in results.items():
                if isinstance(result, dict):
                    # Multiple results (e.g., ablation, comparison)
                    for sub_name, sub_result in result.items():
                        print(f"{exp_name}/{sub_name}: Transfer={sub_result.transfer_accuracy:.1%}")
                else:
                    print(f"{exp_name}: Transfer={result.transfer_accuracy:.1%}")

    else:
        # Run all experiments
        results = run_all_experiments(config, verbose=verbose)

        # Print formatted table
        if verbose:
            print("\n")
            print(format_numerosity_results_table(results))


if __name__ == '__main__':
    main()
