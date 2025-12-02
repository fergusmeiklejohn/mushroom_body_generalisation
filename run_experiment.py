#!/usr/bin/env python3
"""
Main script to run MBGN experiments.

Usage:
    python run_experiment.py                  # Run quick test
    python run_experiment.py --full           # Run full experiment
    python run_experiment.py --ablation       # Run ablation study
"""

import argparse
import numpy as np

from mbgn import MBGN, StimulusGenerator
from mbgn.model import MBGNConfig
from mbgn.stimuli import create_experiment_stimuli
from mbgn.task import TaskType, ExperimentRunner
from mbgn.training import Trainer, TrainingConfig, run_quick_test
from mbgn.analysis import (
    Analyzer, compare_conditions, format_comparison_table,
    learning_curve
)


def run_basic_tests():
    """Run basic tests to verify components work."""
    print("=" * 60)
    print("MBGN Basic Component Tests")
    print("=" * 60)

    # Test 1: Stimulus generation
    print("\n1. Testing stimulus generation...")
    gen = StimulusGenerator(n_dims=50, seed=42)
    training = gen.generate_training_set(4)
    transfer = gen.generate_transfer_set(4)
    print(f"   Generated {len(training)} training, {len(transfer)} transfer stimuli")
    print(f"   Stimulus shape: {training[0].vector.shape}")
    print(f"   Stimulus density: {training[0].vector.mean():.2f}")

    # Test 2: Model creation
    print("\n2. Testing model creation...")
    config = MBGNConfig(n_input=50, n_expansion=2000, seed=42)
    model = MBGN(config)
    print(f"   Input dim: {config.n_input}")
    print(f"   Expansion dim: {config.n_expansion}")
    print(f"   k (active units): {model.k}")
    print(f"   W_proj shape: {model.W_proj.shape}")
    print(f"   W_specific shape: {model.W_specific.shape}")

    # Test 3: Forward pass
    print("\n3. Testing forward pass...")
    result = model.forward(training[0].vector)
    print(f"   Sparse rep shape: {result.sparse_rep.shape}")
    print(f"   Active units: {np.sum(result.sparse_rep > 0)}")
    print(f"   Aggregate activity: {result.aggregate_activity:.2f}")
    print(f"   Decision: {'GO' if result.decision else 'NOGO'}")

    # Test 4: Accommodation
    print("\n4. Testing accommodation...")
    model.reset_accommodation()
    activities = []
    for i in range(5):
        result = model.forward(training[0].vector, update_accommodation=True)
        activities.append(result.aggregate_activity)
        model.decay_accommodation(0.5)

    print(f"   Activities over presentations: {[f'{a:.1f}' for a in activities]}")
    reduction = (activities[0] - activities[-1]) / activities[0] * 100
    print(f"   Activity reduction: {reduction:.1f}%")

    # Test 5: Same/Different discrimination
    print("\n5. Testing same/different discrimination...")
    analyzer = Analyzer(model)
    analysis = analyzer.analyze_same_different(training)
    print(f"   Same aggregate: {analysis['same_aggregate_mean']:.2f} ± {analysis['same_aggregate_std']:.2f}")
    print(f"   Diff aggregate: {analysis['diff_aggregate_mean']:.2f} ± {analysis['diff_aggregate_std']:.2f}")
    print(f"   Discriminability: {analysis['discriminability']:.2f}")

    print("\n" + "=" * 60)
    print("All basic tests passed!")
    print("=" * 60)


def run_full_experiment(task_type: str = 'DMTS', seed: int = 42, n_training: int = 100):
    """Run a full experiment with training and transfer."""
    print("=" * 60)
    print(f"MBGN Full Experiment ({task_type})")
    print("=" * 60)

    # Configuration - using updated defaults from model
    model_config = MBGNConfig(
        n_input=50,
        n_expansion=2000,
        sparsity_fraction=0.05,
        seed=seed
    )

    training_config = TrainingConfig(
        n_training_stimuli=4,
        n_transfer_stimuli=4,
        stimulus_dims=50,
        stimulus_type='binary',
        n_familiarization=10,
        n_training=n_training,
        n_transfer=20,
        block_size=10,
        task_type=TaskType.DMTS if task_type == 'DMTS' else TaskType.DNMTS,
        stimulus_seed=seed,
        model_seed=seed,
        task_seed=seed
    )

    # Run experiment
    trainer = Trainer(model_config, training_config)
    results = trainer.run_experiment(verbose=True)

    # Print detailed results
    print("\n" + "-" * 60)
    print("Learning Curve (accuracy by block):")
    for i, acc in enumerate(results.training_accuracy_by_block):
        block_start = i * training_config.block_size + 1
        block_end = min((i + 1) * training_config.block_size, n_training)
        print(f"  Trials {block_start:3d}-{block_end:3d}: {acc:.1%}")

    # Compute last block accuracy (most relevant for learning success)
    last_block_acc = results.training_accuracy_by_block[-1] if results.training_accuracy_by_block else 0.5
    print(f"\nFinal Block Accuracy: {last_block_acc:.1%}")
    print(f"Overall Training Accuracy: {results.training_accuracy:.1%}")
    print(f"Transfer Accuracy: {results.transfer_accuracy:.1%}")

    # Check success criteria
    print("\n" + "-" * 60)
    print("Success Criteria Check:")
    learning_success = last_block_acc >= 0.70
    transfer_success = results.transfer_accuracy > 0.50
    print(f"  Learning (last block ≥70%): {'PASS' if learning_success else 'FAIL'} ({last_block_acc:.1%})")
    print(f"  Transfer (>50%): {'PASS' if transfer_success else 'FAIL'} ({results.transfer_accuracy:.1%})")

    return results


def run_ablation_study(n_repeats: int = 5, seed: int = 42):
    """Run ablation study comparing model variants."""
    print("=" * 60)
    print("MBGN Ablation Study")
    print("=" * 60)

    model_config = MBGNConfig(
        n_input=50,
        n_expansion=2000,
        sparsity_fraction=0.05,
        seed=seed
    )

    training_config = TrainingConfig(
        n_training_stimuli=4,
        n_transfer_stimuli=4,
        n_familiarization=10,
        n_training=60,
        n_transfer=20,
        stimulus_seed=seed,
        model_seed=seed,
        task_seed=seed
    )

    trainer = Trainer(model_config, training_config)
    results = trainer.run_ablation_study(n_repeats=n_repeats, verbose=False)

    # Print comparison
    comparison = compare_conditions(results)
    print("\n" + format_comparison_table(comparison))

    # Statistical analysis
    print("\nKey Findings:")
    full = comparison['full_model']
    no_acc = comparison['no_accommodation']
    no_agg = comparison['no_aggregate']

    print(f"  - Accommodation ablation: Transfer {no_acc['transfer_accuracy_mean']:.1%} "
          f"(vs {full['transfer_accuracy_mean']:.1%} full model)")
    print(f"  - Aggregate pathway ablation: Transfer {no_agg['transfer_accuracy_mean']:.1%} "
          f"(vs {full['transfer_accuracy_mean']:.1%} full model)")

    if full['transfer_accuracy_mean'] > no_acc['transfer_accuracy_mean']:
        print("  → Accommodation contributes to transfer!")
    if full['transfer_accuracy_mean'] > no_agg['transfer_accuracy_mean']:
        print("  → Aggregate pathway contributes to transfer!")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run MBGN experiments')
    parser.add_argument('--full', action='store_true',
                        help='Run full experiment')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--task', type=str, default='DMTS',
                        choices=['DMTS', 'DNMTS'],
                        help='Task type (default: DMTS)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--repeats', type=int, default=5,
                        help='Number of repeats for ablation study')

    args = parser.parse_args()

    if args.ablation:
        run_ablation_study(n_repeats=args.repeats, seed=args.seed)
    elif args.full:
        run_full_experiment(task_type=args.task, seed=args.seed)
    else:
        # Run basic tests by default
        run_basic_tests()
        print("\n")
        run_full_experiment(task_type=args.task, seed=args.seed)


if __name__ == '__main__':
    main()
