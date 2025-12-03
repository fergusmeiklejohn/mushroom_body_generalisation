"""
Numerosity experiments for MBGN Phase 2.

Implements the 8 experiments specified in mbgn_phase2_numerosity.md:

1. Baseline: Can MBGN learn numerosity comparison?
2. Novel Counts: Transfer to unseen numerosities
3. Novel Types: Transfer to new element types
4. Full Transfer: Both novel counts AND types
5. Comparison: Same/different vs numerosity
6. Ablation: Which components are essential?
7. Choose Fewer: Can the rule be inverted?
8. Distance Effect: Numerical distance effect
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .model import MBGN, MBGNConfig, AblatedMBGN
from .numerosity_stimuli import NumerosityStimulusGenerator, verify_numerosity_correlation
from .numerosity_task import (
    NumerosityTask,
    NumerosityTaskType,
    NumerosityTaskRunner,
    NumerosityExperimentRunner,
    NumerosityTrialResult,
)


@dataclass
class NumerosityExperimentConfig:
    """Configuration for numerosity experiments."""
    # Stimulus parameters
    n_input: int = 50
    stimulus_type: str = 'sparse'

    # Numerosities
    training_numerosities: List[int] = field(default_factory=lambda: [2, 3, 5, 6])
    transfer_numerosities: List[int] = field(default_factory=lambda: [4, 7])

    # Training parameters
    n_training_trials: int = 100
    n_transfer_trials: int = 40
    block_size: int = 20

    # Model parameters
    model_seed: int = 42
    task_seed: int = 42
    stimulus_seed: int = 42

    # Experiment settings
    use_accommodation: bool = False
    n_repeats: int = 5  # For statistical reliability


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    name: str
    training_accuracy: float
    transfer_accuracy: float
    training_accuracy_by_block: List[float]
    detailed_results: Dict[str, Any] = field(default_factory=dict)


def run_experiment_1_baseline(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> ExperimentResult:
    """
    Experiment 1: Numerosity Baseline

    Question: Can MBGN learn numerosity comparison with training stimuli?

    Method:
    1. Generate training set: numerosities {2, 3, 5, 6}
    2. Train on "choose more" task
    3. Test on same numerosities (held-out trials)

    Success criterion: >75% accuracy on training numerosities
    """
    config = config or NumerosityExperimentConfig()

    if verbose:
        print("=" * 60)
        print("Experiment 1: Numerosity Baseline")
        print("=" * 60)

    # Create stimulus generator
    stim_gen = NumerosityStimulusGenerator(
        n_input=config.n_input,
        stimulus_type=config.stimulus_type,
        seed=config.stimulus_seed
    )

    # Verify numerosity signal before training
    verification = verify_numerosity_correlation(
        stim_gen,
        numerosities=config.training_numerosities,
        n_samples=50
    )
    if verbose:
        print(f"Numerosity-aggregate correlation: {verification['correlation']:.3f}")
        print(f"Mean aggregate by numerosity: {verification['mean_by_numerosity']}")

    # Create model
    model_config = MBGNConfig(
        n_input=config.n_input,
        seed=config.model_seed
    )
    model = MBGN(model_config)

    # Create task
    task = NumerosityTask(
        numerosities=config.training_numerosities,
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=config.task_seed
    )

    # Run training
    runner = NumerosityTaskRunner(
        model, task,
        use_accommodation=config.use_accommodation,
        seed=config.task_seed
    )

    training_results = []
    accuracy_by_block = []

    for block_start in range(0, config.n_training_trials, config.block_size):
        block_end = min(block_start + config.block_size, config.n_training_trials)
        block_trials = block_end - block_start

        block_results = runner.run_block(block_trials, learn=True, balanced=True)
        training_results.extend(block_results)
        block_acc = runner.compute_accuracy(block_results)
        accuracy_by_block.append(block_acc)

        if verbose:
            print(f"  Block {len(accuracy_by_block)}: {block_acc:.1%}")

    # Test on same numerosities (held-out trials, no learning)
    test_results = runner.run_block(
        config.n_transfer_trials, learn=False, balanced=True
    )
    test_accuracy = runner.compute_accuracy(test_results)

    training_accuracy = runner.compute_accuracy(training_results)

    if verbose:
        print(f"\nTraining accuracy: {training_accuracy:.1%}")
        print(f"Test accuracy (same numerosities): {test_accuracy:.1%}")
        success = test_accuracy > 0.75
        print(f"Success (>75%): {'YES' if success else 'NO'}")

    return ExperimentResult(
        name="Exp1: Baseline",
        training_accuracy=training_accuracy,
        transfer_accuracy=test_accuracy,
        training_accuracy_by_block=accuracy_by_block,
        detailed_results={
            'verification': verification,
            'distance_effect': runner.compute_accuracy_by_distance(test_results)
        }
    )


def run_experiment_2_novel_counts(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> ExperimentResult:
    """
    Experiment 2: Numerosity Transfer (Novel Counts)

    Question: Does numerosity rule transfer to unseen numerosities?

    Method:
    1. Train on {2, 3, 5, 6} elements
    2. Test on {4, 7} elements (never seen during training)
    3. No learning during transfer

    Success criterion: >65% accuracy on novel numerosities
    """
    config = config or NumerosityExperimentConfig()

    if verbose:
        print("=" * 60)
        print("Experiment 2: Novel Counts Transfer")
        print("=" * 60)

    # Create stimulus generator
    stim_gen = NumerosityStimulusGenerator(
        n_input=config.n_input,
        stimulus_type=config.stimulus_type,
        seed=config.stimulus_seed
    )

    # Create model
    model_config = MBGNConfig(n_input=config.n_input, seed=config.model_seed)
    model = MBGN(model_config)

    # Create experiment runner
    exp_runner = NumerosityExperimentRunner(
        model=model,
        training_numerosities=config.training_numerosities,
        transfer_numerosities=config.transfer_numerosities,
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        training_element_type='A',
        transfer_element_type='A',  # Same element type, novel counts
        use_accommodation=config.use_accommodation,
        seed=config.task_seed
    )

    # Run experiment
    results = exp_runner.run_experiment(
        n_training=config.n_training_trials,
        n_transfer=config.n_transfer_trials,
        block_size=config.block_size
    )

    if verbose:
        print(f"Training accuracy: {results['training_accuracy']:.1%}")
        print(f"Transfer (novel counts): {results['transfer_novel_num_accuracy']:.1%}")
        print(f"Distance effect: {results['distance_effect_transfer']}")
        success = results['transfer_novel_num_accuracy'] > 0.65
        print(f"Success (>65%): {'YES' if success else 'NO'}")

    return ExperimentResult(
        name="Exp2: Novel Counts",
        training_accuracy=results['training_accuracy'],
        transfer_accuracy=results['transfer_novel_num_accuracy'],
        training_accuracy_by_block=results['training_accuracy_by_block'],
        detailed_results={
            'distance_effect': results['distance_effect_transfer'],
            'full_results': results
        }
    )


def run_experiment_3_novel_types(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> ExperimentResult:
    """
    Experiment 3: Numerosity Transfer (Novel Element Types)

    Question: Does numerosity rule transfer to novel stimulus types?

    Method:
    1. Train on element type A (specific position patterns)
    2. Test on element type B (different position patterns)
    3. Same numerosities as training

    Success criterion: >70% accuracy on novel element types
    """
    config = config or NumerosityExperimentConfig()

    if verbose:
        print("=" * 60)
        print("Experiment 3: Novel Element Types Transfer")
        print("=" * 60)

    # Create stimulus generator
    stim_gen = NumerosityStimulusGenerator(
        n_input=config.n_input,
        stimulus_type=config.stimulus_type,
        seed=config.stimulus_seed
    )

    # Create model
    model_config = MBGNConfig(n_input=config.n_input, seed=config.model_seed)
    model = MBGN(model_config)

    # Create experiment runner
    exp_runner = NumerosityExperimentRunner(
        model=model,
        training_numerosities=config.training_numerosities,
        transfer_numerosities=config.transfer_numerosities,
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        training_element_type='A',
        transfer_element_type='B',  # Novel element type
        use_accommodation=config.use_accommodation,
        seed=config.task_seed
    )

    # Run experiment
    results = exp_runner.run_experiment(
        n_training=config.n_training_trials,
        n_transfer=config.n_transfer_trials,
        block_size=config.block_size
    )

    if verbose:
        print(f"Training accuracy: {results['training_accuracy']:.1%}")
        print(f"Transfer (novel type): {results['transfer_novel_type_accuracy']:.1%}")
        success = results['transfer_novel_type_accuracy'] > 0.70
        print(f"Success (>70%): {'YES' if success else 'NO'}")

    return ExperimentResult(
        name="Exp3: Novel Types",
        training_accuracy=results['training_accuracy'],
        transfer_accuracy=results['transfer_novel_type_accuracy'],
        training_accuracy_by_block=results['training_accuracy_by_block'],
        detailed_results={'full_results': results}
    )


def run_experiment_4_full_transfer(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> ExperimentResult:
    """
    Experiment 4: Full Transfer

    Question: Does the rule transfer to both novel counts AND novel types?

    This is the critical test analogous to the same/different transfer experiment.

    Method:
    1. Train on {2, 3, 5, 6} with element type A
    2. Test on {4, 7} with element type B

    Success criterion: >60% accuracy
    """
    config = config or NumerosityExperimentConfig()

    if verbose:
        print("=" * 60)
        print("Experiment 4: Full Transfer (Critical Test)")
        print("=" * 60)

    # Create stimulus generator
    stim_gen = NumerosityStimulusGenerator(
        n_input=config.n_input,
        stimulus_type=config.stimulus_type,
        seed=config.stimulus_seed
    )

    # Create model
    model_config = MBGNConfig(n_input=config.n_input, seed=config.model_seed)
    model = MBGN(model_config)

    # Create experiment runner
    exp_runner = NumerosityExperimentRunner(
        model=model,
        training_numerosities=config.training_numerosities,
        transfer_numerosities=config.transfer_numerosities,
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        training_element_type='A',
        transfer_element_type='B',
        use_accommodation=config.use_accommodation,
        seed=config.task_seed
    )

    # Run experiment
    results = exp_runner.run_experiment(
        n_training=config.n_training_trials,
        n_transfer=config.n_transfer_trials,
        block_size=config.block_size
    )

    if verbose:
        print(f"Training accuracy: {results['training_accuracy']:.1%}")
        print(f"Full Transfer accuracy: {results['transfer_full_accuracy']:.1%}")
        success = results['transfer_full_accuracy'] > 0.60
        print(f"Success (>60%): {'YES' if success else 'NO'}")

    return ExperimentResult(
        name="Exp4: Full Transfer",
        training_accuracy=results['training_accuracy'],
        transfer_accuracy=results['transfer_full_accuracy'],
        training_accuracy_by_block=results['training_accuracy_by_block'],
        detailed_results={
            'all_transfer_accuracies': {
                'novel_num': results['transfer_novel_num_accuracy'],
                'novel_type': results['transfer_novel_type_accuracy'],
                'full': results['transfer_full_accuracy']
            },
            'full_results': results
        }
    )


def run_experiment_5_comparison(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> Dict[str, ExperimentResult]:
    """
    Experiment 5: Comparison with Same/Different

    Question: How does numerosity transfer compare to same/different transfer?

    Method:
    1. Run numerosity experiment with full transfer
    2. Run same/different experiment with same model config
    3. Compare transfer accuracies

    This imports from the existing training module for same/different.
    """
    config = config or NumerosityExperimentConfig()

    if verbose:
        print("=" * 60)
        print("Experiment 5: Comparison with Same/Different")
        print("=" * 60)

    # Run numerosity experiment
    numerosity_result = run_experiment_4_full_transfer(config, verbose=False)

    # Run same/different experiment using existing infrastructure
    from .training import Trainer, TrainingConfig
    from .task import TaskType

    sd_config = TrainingConfig(
        n_training_stimuli=4,
        n_transfer_stimuli=4,
        stimulus_dims=config.n_input,
        stimulus_type='binary',
        n_training=config.n_training_trials,
        n_transfer=config.n_transfer_trials,
        block_size=config.block_size,
        task_type=TaskType.DMTS,
        model_seed=config.model_seed,
        stimulus_seed=config.stimulus_seed,
        task_seed=config.task_seed
    )

    trainer = Trainer(sd_config)
    sd_results = trainer.run_experiment()

    sd_experiment_result = ExperimentResult(
        name="Same/Different",
        training_accuracy=sd_results.training_accuracy,
        transfer_accuracy=sd_results.transfer_accuracy,
        training_accuracy_by_block=sd_results.training_accuracy_by_block,
        detailed_results={'full_results': sd_results}
    )

    if verbose:
        print(f"\nNumerosity Transfer: {numerosity_result.transfer_accuracy:.1%}")
        print(f"Same/Different Transfer: {sd_experiment_result.transfer_accuracy:.1%}")
        diff = numerosity_result.transfer_accuracy - sd_experiment_result.transfer_accuracy
        print(f"Difference: {diff:+.1%}")

    return {
        'numerosity': numerosity_result,
        'same_different': sd_experiment_result
    }


def run_experiment_6_ablation(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> Dict[str, ExperimentResult]:
    """
    Experiment 6: Ablation Study

    Question: Which components are necessary for numerosity transfer?

    Ablations:
    1. Full model (baseline)
    2. No aggregate pathway → expect failure
    3. With accommodation → expect interference
    4. Different sparsity levels

    Uses AblatedMBGN for controlled ablations.
    """
    config = config or NumerosityExperimentConfig()

    if verbose:
        print("=" * 60)
        print("Experiment 6: Ablation Study")
        print("=" * 60)

    results = {}

    # Ablation conditions
    conditions = [
        ('full_model', {'disable_accommodation': False, 'disable_aggregate_pathway': False}),
        ('no_aggregate', {'disable_accommodation': False, 'disable_aggregate_pathway': True}),
        ('with_accommodation', {'disable_accommodation': False, 'disable_aggregate_pathway': False}),
    ]

    for condition_name, ablation_params in conditions:
        if verbose:
            print(f"\n--- Condition: {condition_name} ---")

        # Create stimulus generator
        stim_gen = NumerosityStimulusGenerator(
            n_input=config.n_input,
            stimulus_type=config.stimulus_type,
            seed=config.stimulus_seed
        )

        # Create ablated model
        model_config = MBGNConfig(n_input=config.n_input, seed=config.model_seed)

        if condition_name == 'no_aggregate':
            model = AblatedMBGN(
                model_config,
                disable_aggregate_pathway=True
            )
        else:
            model = MBGN(model_config)

        # For with_accommodation, we enable it during task running
        use_accommodation = (condition_name == 'with_accommodation')

        # Create experiment runner
        exp_runner = NumerosityExperimentRunner(
            model=model,
            training_numerosities=config.training_numerosities,
            transfer_numerosities=config.transfer_numerosities,
            stimulus_generator=stim_gen,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            training_element_type='A',
            transfer_element_type='B',
            use_accommodation=use_accommodation,
            seed=config.task_seed
        )

        # Run experiment
        exp_results = exp_runner.run_experiment(
            n_training=config.n_training_trials,
            n_transfer=config.n_transfer_trials,
            block_size=config.block_size
        )

        results[condition_name] = ExperimentResult(
            name=f"Ablation: {condition_name}",
            training_accuracy=exp_results['training_accuracy'],
            transfer_accuracy=exp_results['transfer_full_accuracy'],
            training_accuracy_by_block=exp_results['training_accuracy_by_block'],
            detailed_results={'full_results': exp_results}
        )

        if verbose:
            print(f"  Training: {exp_results['training_accuracy']:.1%}")
            print(f"  Transfer: {exp_results['transfer_full_accuracy']:.1%}")

    return results


def run_experiment_7_choose_fewer(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> ExperimentResult:
    """
    Experiment 7: "Choose Fewer" Reversal

    Question: Can the model learn the opposite rule?

    Method:
    1. Train on "choose fewer" instead of "choose more"
    2. Test transfer

    Expected outcome: Should work equally well (just inverts the aggregate comparison)
    """
    config = config or NumerosityExperimentConfig()

    if verbose:
        print("=" * 60)
        print("Experiment 7: Choose Fewer Reversal")
        print("=" * 60)

    # Create stimulus generator
    stim_gen = NumerosityStimulusGenerator(
        n_input=config.n_input,
        stimulus_type=config.stimulus_type,
        seed=config.stimulus_seed
    )

    # Create model
    model_config = MBGNConfig(n_input=config.n_input, seed=config.model_seed)
    model = MBGN(model_config)

    # Create experiment runner with CHOOSE_FEWER
    exp_runner = NumerosityExperimentRunner(
        model=model,
        training_numerosities=config.training_numerosities,
        transfer_numerosities=config.transfer_numerosities,
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_FEWER,  # Opposite rule
        training_element_type='A',
        transfer_element_type='B',
        use_accommodation=config.use_accommodation,
        seed=config.task_seed
    )

    # Run experiment
    results = exp_runner.run_experiment(
        n_training=config.n_training_trials,
        n_transfer=config.n_transfer_trials,
        block_size=config.block_size
    )

    if verbose:
        print(f"Training accuracy: {results['training_accuracy']:.1%}")
        print(f"Full Transfer accuracy: {results['transfer_full_accuracy']:.1%}")
        success = results['transfer_full_accuracy'] > 0.60
        print(f"Success (>60%): {'YES' if success else 'NO'}")

    return ExperimentResult(
        name="Exp7: Choose Fewer",
        training_accuracy=results['training_accuracy'],
        transfer_accuracy=results['transfer_full_accuracy'],
        training_accuracy_by_block=results['training_accuracy_by_block'],
        detailed_results={'full_results': results}
    )


def run_experiment_8_distance_effect(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> ExperimentResult:
    """
    Experiment 8: Numerical Distance Effect

    Question: Does the model show the numerical distance effect (like bees and humans)?

    Method:
    1. Train on full range of numerosities
    2. Test on all pairs
    3. Plot accuracy vs. numerical distance (|N₁ - N₂|)

    Expected outcome: Accuracy should increase with numerical distance
    """
    config = config or NumerosityExperimentConfig()

    # Use all numerosities for this experiment
    all_numerosities = sorted(set(config.training_numerosities + config.transfer_numerosities))

    if verbose:
        print("=" * 60)
        print("Experiment 8: Numerical Distance Effect")
        print("=" * 60)
        print(f"Testing numerosities: {all_numerosities}")

    # Create stimulus generator
    stim_gen = NumerosityStimulusGenerator(
        n_input=config.n_input,
        stimulus_type=config.stimulus_type,
        seed=config.stimulus_seed
    )

    # Create model
    model_config = MBGNConfig(n_input=config.n_input, seed=config.model_seed)
    model = MBGN(model_config)

    # Create task with all numerosities
    task = NumerosityTask(
        numerosities=all_numerosities,
        stimulus_generator=stim_gen,
        task_type=NumerosityTaskType.CHOOSE_MORE,
        seed=config.task_seed
    )

    # Run training
    runner = NumerosityTaskRunner(
        model, task,
        use_accommodation=config.use_accommodation,
        seed=config.task_seed
    )

    training_results = []
    accuracy_by_block = []

    for block_start in range(0, config.n_training_trials * 2, config.block_size):
        block_end = min(block_start + config.block_size, config.n_training_trials * 2)
        block_trials = block_end - block_start

        block_results = runner.run_block(block_trials, learn=True, balanced=True)
        training_results.extend(block_results)
        block_acc = runner.compute_accuracy(block_results)
        accuracy_by_block.append(block_acc)

        if verbose:
            print(f"  Block {len(accuracy_by_block)}: {block_acc:.1%}")

    # Test all pairs (no learning)
    test_results = runner.run_block(
        config.n_transfer_trials * 2, learn=False, balanced=True
    )

    # Compute distance effect
    distance_effect = runner.compute_accuracy_by_distance(test_results)
    pair_accuracy = runner.compute_accuracy_by_pair(test_results)

    if verbose:
        print("\nAccuracy by numerical distance:")
        for dist in sorted(distance_effect.keys()):
            print(f"  Distance {dist}: {distance_effect[dist]:.1%}")

        # Check if distance effect is present (monotonically increasing)
        distances = sorted(distance_effect.keys())
        accuracies = [distance_effect[d] for d in distances]
        is_monotonic = all(
            accuracies[i] <= accuracies[i+1]
            for i in range(len(accuracies)-1)
        )
        print(f"\nDistance effect present (monotonic): {'YES' if is_monotonic else 'NO'}")

    return ExperimentResult(
        name="Exp8: Distance Effect",
        training_accuracy=runner.compute_accuracy(training_results),
        transfer_accuracy=runner.compute_accuracy(test_results),
        training_accuracy_by_block=accuracy_by_block,
        detailed_results={
            'distance_effect': distance_effect,
            'pair_accuracy': pair_accuracy
        }
    )


def run_all_experiments(
    config: Optional[NumerosityExperimentConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run all numerosity experiments and return consolidated results.
    """
    config = config or NumerosityExperimentConfig()

    all_results = {}

    print("\n" + "=" * 70)
    print("MBGN PHASE 2: NUMEROSITY EXPERIMENTS")
    print("=" * 70 + "\n")

    # Experiment 1: Baseline
    all_results['exp1_baseline'] = run_experiment_1_baseline(config, verbose)

    # Experiment 2: Novel Counts
    all_results['exp2_novel_counts'] = run_experiment_2_novel_counts(config, verbose)

    # Experiment 3: Novel Types
    all_results['exp3_novel_types'] = run_experiment_3_novel_types(config, verbose)

    # Experiment 4: Full Transfer
    all_results['exp4_full_transfer'] = run_experiment_4_full_transfer(config, verbose)

    # Experiment 5: Comparison (optional - requires same/different infrastructure)
    try:
        all_results['exp5_comparison'] = run_experiment_5_comparison(config, verbose)
    except Exception as e:
        if verbose:
            print(f"Experiment 5 skipped: {e}")
        all_results['exp5_comparison'] = None

    # Experiment 6: Ablation
    all_results['exp6_ablation'] = run_experiment_6_ablation(config, verbose)

    # Experiment 7: Choose Fewer
    all_results['exp7_choose_fewer'] = run_experiment_7_choose_fewer(config, verbose)

    # Experiment 8: Distance Effect
    all_results['exp8_distance_effect'] = run_experiment_8_distance_effect(config, verbose)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        summary_table = [
            ("Exp 1: Baseline", all_results['exp1_baseline'].transfer_accuracy, ">75%"),
            ("Exp 2: Novel Counts", all_results['exp2_novel_counts'].transfer_accuracy, ">65%"),
            ("Exp 3: Novel Types", all_results['exp3_novel_types'].transfer_accuracy, ">70%"),
            ("Exp 4: Full Transfer", all_results['exp4_full_transfer'].transfer_accuracy, ">60%"),
            ("Exp 7: Choose Fewer", all_results['exp7_choose_fewer'].transfer_accuracy, ">60%"),
            ("Exp 8: Distance Effect", all_results['exp8_distance_effect'].transfer_accuracy, "N/A"),
        ]

        print(f"{'Experiment':<25} {'Accuracy':<12} {'Criterion':<10} {'Pass?':<6}")
        print("-" * 55)

        for name, acc, criterion in summary_table:
            if criterion != "N/A":
                threshold = float(criterion[1:-1]) / 100
                passed = "YES" if acc > threshold else "NO"
            else:
                passed = "N/A"
            print(f"{name:<25} {acc:>8.1%}     {criterion:<10} {passed:<6}")

    return all_results
