"""
Mushroom Body-Inspired Generalisation Network (MBGN)

A minimal neural network inspired by insect mushroom body architecture
for learning relational concepts that transfer to novel stimuli.
"""

from .model import MBGN, MBGNConfig, AblatedMBGN
from .stimuli import StimulusGenerator
from .task import DMTSTask, DNMTSTask
from .training import Trainer
from .analysis import Analyzer

# Phase 2: Numerosity modules
from .numerosity_stimuli import (
    NumerosityStimulus,
    NumerosityStimulusSet,
    NumerosityStimulusGenerator,
    create_numerosity_stimuli,
    verify_numerosity_correlation,
)
from .numerosity_task import (
    NumerosityTaskType,
    NumerosityTrial,
    NumerosityTrialResult,
    NumerosityTask,
    NumerosityTaskRunner,
    NumerosityExperimentRunner,
)
from .numerosity_experiments import (
    NumerosityExperimentConfig,
    ExperimentResult,
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

__version__ = "0.1.0"
__all__ = [
    # Core model
    "MBGN",
    "MBGNConfig",
    "AblatedMBGN",
    # Same/different stimuli and tasks
    "StimulusGenerator",
    "DMTSTask",
    "DNMTSTask",
    "Trainer",
    "Analyzer",
    # Numerosity stimuli
    "NumerosityStimulus",
    "NumerosityStimulusSet",
    "NumerosityStimulusGenerator",
    "create_numerosity_stimuli",
    "verify_numerosity_correlation",
    # Numerosity tasks
    "NumerosityTaskType",
    "NumerosityTrial",
    "NumerosityTrialResult",
    "NumerosityTask",
    "NumerosityTaskRunner",
    "NumerosityExperimentRunner",
    # Numerosity experiments
    "NumerosityExperimentConfig",
    "ExperimentResult",
    "run_experiment_1_baseline",
    "run_experiment_2_novel_counts",
    "run_experiment_3_novel_types",
    "run_experiment_4_full_transfer",
    "run_experiment_5_comparison",
    "run_experiment_6_ablation",
    "run_experiment_7_choose_fewer",
    "run_experiment_8_distance_effect",
    "run_all_experiments",
]
