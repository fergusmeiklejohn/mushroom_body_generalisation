"""
Training utilities for MBGN experiments.

Provides high-level training interface and experiment management.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import time

from .model import MBGN, MBGNConfig, AblatedMBGN
from .stimuli import Stimulus, create_experiment_stimuli
from .task import (
    TaskType, DMTSTask, DNMTSTask, TaskRunner,
    ExperimentRunner, TrialResult
)


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    # Stimuli
    n_training_stimuli: int = 4
    n_transfer_stimuli: int = 4
    stimulus_dims: int = 50
    stimulus_type: str = 'binary'

    # Training
    n_familiarization: int = 10
    n_training: int = 60
    n_transfer: int = 20
    block_size: int = 10

    # Task
    task_type: TaskType = TaskType.DMTS

    # Random seeds
    stimulus_seed: Optional[int] = 42
    model_seed: Optional[int] = 42
    task_seed: Optional[int] = 42


@dataclass
class ExperimentResults:
    """Complete results from an experiment."""
    config: TrainingConfig
    model_config: MBGNConfig
    training_results: List[TrialResult]
    transfer_results: List[TrialResult]
    training_accuracy_by_block: List[float]
    training_accuracy: float
    transfer_accuracy: float
    model_state_initial: Dict[str, Any]
    model_state_final: Dict[str, Any]
    run_time: float


class Trainer:
    """
    High-level trainer for MBGN experiments.
    """

    def __init__(
        self,
        model_config: Optional[MBGNConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            model_config: Configuration for MBGN model
            training_config: Configuration for training
        """
        self.model_config = model_config or MBGNConfig()
        self.training_config = training_config or TrainingConfig()

    def run_experiment(
        self,
        model: Optional[MBGN] = None,
        training_stimuli: Optional[List[Stimulus]] = None,
        transfer_stimuli: Optional[List[Stimulus]] = None,
        verbose: bool = True
    ) -> ExperimentResults:
        """
        Run a complete training and transfer experiment.

        Args:
            model: Pre-initialized model (creates new one if None)
            training_stimuli: Pre-generated training stimuli
            transfer_stimuli: Pre-generated transfer stimuli
            verbose: Print progress

        Returns:
            ExperimentResults with all data
        """
        start_time = time.time()
        cfg = self.training_config

        # Create stimuli if not provided
        if training_stimuli is None or transfer_stimuli is None:
            training_stimuli, transfer_stimuli = create_experiment_stimuli(
                n_training=cfg.n_training_stimuli,
                n_transfer=cfg.n_transfer_stimuli,
                n_dims=cfg.stimulus_dims,
                stimulus_type=cfg.stimulus_type,
                seed=cfg.stimulus_seed
            )

        # Create model if not provided
        if model is None:
            self.model_config.n_input = cfg.stimulus_dims
            self.model_config.seed = cfg.model_seed
            model = MBGN(self.model_config)
            # Calibrate aggregate baseline to this stimulus set (reduces variance)
            model.calibrate_baseline(training_stimuli)
            # Bias aggregate pathway for the task type (reduces variance)
            model.set_aggregate_bias(cfg.task_type.name)

        # Save initial state
        initial_state = model.get_state()

        # Create experiment runner
        runner = ExperimentRunner(
            model=model,
            training_stimuli=training_stimuli,
            transfer_stimuli=transfer_stimuli,
            task_type=cfg.task_type,
            seed=cfg.task_seed
        )

        if verbose:
            print(f"Running {cfg.task_type.name} experiment...")
            print(f"  Training stimuli: {len(training_stimuli)}")
            print(f"  Transfer stimuli: {len(transfer_stimuli)}")
            print(f"  Training trials: {cfg.n_training}")
            print(f"  Transfer trials: {cfg.n_transfer}")

        # Run experiment
        results = runner.run_experiment(
            n_familiarization=cfg.n_familiarization,
            n_training=cfg.n_training,
            n_transfer=cfg.n_transfer,
            block_size=cfg.block_size
        )

        # Save final state
        final_state = model.get_state()

        run_time = time.time() - start_time

        if verbose:
            print(f"\nResults:")
            print(f"  Training accuracy: {results['training_accuracy']:.1%}")
            print(f"  Transfer accuracy: {results['transfer_accuracy']:.1%}")
            print(f"  Time: {run_time:.2f}s")

        return ExperimentResults(
            config=cfg,
            model_config=self.model_config,
            training_results=results['training'],
            transfer_results=results['transfer'],
            training_accuracy_by_block=results['training_accuracy_by_block'],
            training_accuracy=results['training_accuracy'],
            transfer_accuracy=results['transfer_accuracy'],
            model_state_initial=initial_state,
            model_state_final=final_state,
            run_time=run_time
        )

    def run_ablation_study(
        self,
        n_repeats: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[ExperimentResults]]:
        """
        Run ablation study comparing different model configurations.

        Args:
            n_repeats: Number of repetitions per condition
            verbose: Print progress

        Returns:
            Dictionary mapping condition name to list of results
        """
        cfg = self.training_config

        # Generate stimuli once (shared across conditions)
        training_stimuli, transfer_stimuli = create_experiment_stimuli(
            n_training=cfg.n_training_stimuli,
            n_transfer=cfg.n_transfer_stimuli,
            n_dims=cfg.stimulus_dims,
            stimulus_type=cfg.stimulus_type,
            seed=cfg.stimulus_seed
        )

        conditions = {
            'full_model': {
                'disable_accommodation': False,
                'disable_specific_pathway': False,
                'disable_aggregate_pathway': False
            },
            'no_accommodation': {
                'disable_accommodation': True,
                'disable_specific_pathway': False,
                'disable_aggregate_pathway': False
            },
            'no_specific': {
                'disable_accommodation': False,
                'disable_specific_pathway': True,
                'disable_aggregate_pathway': False
            },
            'no_aggregate': {
                'disable_accommodation': False,
                'disable_specific_pathway': False,
                'disable_aggregate_pathway': True
            }
        }

        all_results = {}

        for condition_name, ablation_params in conditions.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"Condition: {condition_name}")
                print(f"{'='*50}")

            condition_results = []

            for i in range(n_repeats):
                if verbose:
                    print(f"\n  Repeat {i+1}/{n_repeats}")

                # Create ablated model
                self.model_config.n_input = cfg.stimulus_dims
                self.model_config.seed = cfg.model_seed + i if cfg.model_seed else None

                model = AblatedMBGN(
                    config=self.model_config,
                    **ablation_params
                )
                # Calibrate aggregate baseline to this stimulus set (reduces variance)
                model.calibrate_baseline(training_stimuli)
                # Bias aggregate pathway for the task type (reduces variance)
                model.set_aggregate_bias(cfg.task_type.name)

                # Run experiment
                result = self.run_experiment(
                    model=model,
                    training_stimuli=training_stimuli,
                    transfer_stimuli=transfer_stimuli,
                    verbose=verbose
                )

                condition_results.append(result)

            all_results[condition_name] = condition_results

        return all_results


def compute_statistics(results: List[ExperimentResults]) -> Dict[str, float]:
    """
    Compute summary statistics from multiple experiment runs.

    Args:
        results: List of experiment results

    Returns:
        Dictionary with mean and std for key metrics
    """
    training_accs = [r.training_accuracy for r in results]
    transfer_accs = [r.transfer_accuracy for r in results]

    return {
        'training_accuracy_mean': np.mean(training_accs),
        'training_accuracy_std': np.std(training_accs),
        'transfer_accuracy_mean': np.mean(transfer_accs),
        'transfer_accuracy_std': np.std(transfer_accs)
    }


def run_quick_test(seed: int = 42) -> ExperimentResults:
    """
    Run a quick test to verify everything works.

    Args:
        seed: Random seed

    Returns:
        ExperimentResults
    """
    config = TrainingConfig(
        n_training_stimuli=4,
        n_transfer_stimuli=4,
        n_familiarization=5,
        n_training=30,
        n_transfer=10,
        stimulus_seed=seed,
        model_seed=seed,
        task_seed=seed
    )

    model_config = MBGNConfig(
        n_input=50,
        n_expansion=1000,
        sparsity_fraction=0.1,
        seed=seed
    )

    trainer = Trainer(model_config, config)
    return trainer.run_experiment(verbose=True)
