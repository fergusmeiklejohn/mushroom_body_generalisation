# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository implements a **Mushroom Body-Inspired Generalisation Network (MBGN)** - a neural network inspired by insect brain architecture to learn relational concepts (same/different) that transfer to novel stimuli.

**Core hypothesis**: The insect mushroom body achieves relational generalisation through three architectural features:
1. **Expansion to sparse representations**: Random projection from low-dim inputs to high-dim space with k-winner-take-all
2. **Dual readout pathways**: Specific pathway (pattern-based) and aggregate pathway (sum-based) for different learning types
3. **Sensory accommodation**: Short-term depression that creates automatic novelty detection

The system is tested on Delayed Match-to-Sample (DMTS) and Delayed Non-Match-to-Sample (DNMTS) tasks.

## Commands

### Running Experiments

```bash
# Basic test (runs component tests + quick experiment)
python run_experiment.py

# Full experiment with DMTS task (100 trials)
python run_experiment.py --full

# Full experiment with DNMTS task
python run_experiment.py --full --task DNMTS

# Ablation study (compares full model vs. variants)
python run_experiment.py --ablation --repeats 5

# Set random seed for reproducibility
python run_experiment.py --full --seed 42
```

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# No build step needed - pure Python/NumPy implementation
```

## Architecture

### Module Structure

The codebase is organized into focused modules within `mbgn/`:

- **`model.py`**: Core MBGN implementation
  - `MBGNConfig`: Configuration dataclass
  - `MBGN`: Main network with expansion, k-WTA, accommodation, and dual readout
  - `AblatedMBGN`: Variants for ablation studies
  - `ForwardResult`: Container for forward pass outputs

- **`stimuli.py`**: Stimulus generation and management
  - `Stimulus`: Stimulus representation with vector and metadata
  - `StimulusGenerator`: Creates binary or continuous pattern stimuli
  - `create_experiment_stimuli()`: Generates balanced training/transfer sets

- **`task.py`**: Task definitions and trial execution
  - `TaskType`: Enum for DMTS/DNMTS
  - `Trial`, `TrialResult`: Trial structure and outcomes
  - `BaseTask`, `DMTSTask`, `DNMTSTask`: Task implementations
  - `TaskRunner`: Executes single trials with network
  - `ExperimentRunner`: Orchestrates full experiments (familiarization → training → transfer)

- **`training.py`**: High-level training interface
  - `TrainingConfig`: Experiment configuration
  - `Trainer`: Main interface for running experiments and ablations
  - `ExperimentResults`: Complete results container

- **`analysis.py`**: Results analysis and visualization
  - `Analyzer`: Analyzes network representations and learning
  - Comparison utilities for ablation studies
  - Learning curve and performance metrics

### Data Flow

```
Input Stimulus (binary vector, 50-dim)
  ↓
W_proj (random sparse projection)
  ↓
Expansion Layer (2000-dim, pre-sparse)
  ↓
Accommodation (multiply by 1 - accommodation_state)
  ↓
k-Winner-Take-All (keep top 5%, ~100 units)
  ↓
Sparse Representation (Kenyon cells)
  ├─────────────────────┬──────────────────
  ↓                     ↓
Specific Pathway    Aggregate Pathway
(W_specific)        (sum → W_aggregate)
  ↓                     ↓
  └─────────────────────┘
           ↓
    Combined Output (GO/NOGO decision)
```

### Key Mechanisms

**Accommodation**: Recently active units have reduced responsiveness. When a stimulus is repeated, the same units fire but with lower total activity. This creates an automatic "same vs different" signal:
- Repeated stimulus → low aggregate activity → "SAME"
- Novel stimulus → high aggregate activity → "DIFFERENT"

**Dual Pathways**:
- **Specific**: Learns stimulus-specific associations (doesn't transfer)
- **Aggregate**: Learns based on total activity level (transfers because it's stimulus-independent)

**Learning**: Reward-modulated Hebbian learning with separate learning rates:
- `lr_specific = 0.001` (slow, for stimulus-specific associations)
- `lr_aggregate = 0.1` (fast, for learning abstract rules)

## Important Implementation Details

### Model Configuration

Default hyperparameters in `MBGNConfig` are based on biological data and preliminary tuning:
- `n_expansion = 2000` (~40× expansion, matching insect ratios)
- `sparsity_fraction = 0.05` (5% active, matching Kenyon cell activity)
- `connection_prob = 0.14` (~7 connections per KC, as in bees)
- `accommodation_alpha = 0.7` (produces ~50% reduction on repeat)
- `accommodation_tau = 30.0` (seconds, decay time constant)
- `aggregate_baseline = 600.0` (between "same" and "different" aggregate values)

These should generally not be changed unless running systematic parameter sweeps.

### Accommodation Dynamics

Accommodation state is updated on each forward pass when `update_accommodation=True`:
```python
# During trial phases
active_mask = (sparse_rep > 0).astype(float)
accommodation_state = accommodation_state + alpha * active_mask

# Between presentations
accommodation_state *= exp(-dt / tau)
```

The model automatically calls `decay_accommodation()` during trial delays and inter-trial intervals.

### Trial Structure

Each trial follows this sequence:
1. **Sample**: Present stimulus, induce accommodation, no decision
2. **Delay**: Brief pause, accommodation partially decays
3. **Choice 1**: Present first choice, get GO/NOGO decision
4. **Choice 2**: If NOGO on choice 1, present second choice
5. **Outcome**: Reward based on task type (DMTS: reward match, DNMTS: reward non-match)

Learning updates happen at the outcome phase using the reward signal.

### Transfer Testing

The critical test is whether the learned rule transfers to completely novel stimuli:
1. Train on stimulus set A (e.g., 4 patterns)
2. Test on stimulus set B (4 different patterns) **without** further learning
3. Compare transfer accuracy to final training accuracy

Success requires both accommodation and aggregate pathway - ablations verify this.

## Common Development Patterns

### Creating and Running a Custom Experiment

```python
from mbgn import MBGN
from mbgn.model import MBGNConfig
from mbgn.stimuli import create_experiment_stimuli
from mbgn.training import Trainer, TrainingConfig
from mbgn.task import TaskType

# Configure model and training
model_config = MBGNConfig(n_input=50, n_expansion=2000, seed=42)
training_config = TrainingConfig(
    n_training_stimuli=4,
    n_training=60,
    task_type=TaskType.DMTS,
    seed=42
)

# Run experiment
trainer = Trainer(model_config, training_config)
results = trainer.run_experiment(verbose=True)

print(f"Training: {results.training_accuracy:.1%}")
print(f"Transfer: {results.transfer_accuracy:.1%}")
```

### Analyzing Network Representations

```python
from mbgn.analysis import Analyzer

analyzer = Analyzer(model)
stimuli = training_stimuli[:4]

# Check same/different discriminability
analysis = analyzer.analyze_same_different(stimuli)
print(f"Discriminability: {analysis['discriminability']:.2f}")

# Visualize sparse codes
analyzer.visualize_sparse_codes(stimuli)

# Track accommodation dynamics
analyzer.plot_accommodation_dynamics(stimulus, n_repeats=5)
```

### Running Specific Ablations

```python
from mbgn import AblatedMBGN

# Create variant without accommodation
model = AblatedMBGN(
    config,
    disable_accommodation=True,
    disable_aggregate_pathway=False
)

# Run same experiment to compare
trainer = Trainer(model_config, training_config)
results = trainer.run_experiment(model=model)
```

## File Organization

```
mushroom_body_generalisation/
├── mbgn/                           # Main package
│   ├── __init__.py                 # Package exports
│   ├── model.py                    # MBGN implementation
│   ├── stimuli.py                  # Stimulus generation
│   ├── task.py                     # Task definitions
│   ├── training.py                 # Training interface
│   └── analysis.py                 # Analysis utilities
├── run_experiment.py               # Main experiment script
├── requirements.txt                # Dependencies
└── mushroom_body_generalisation_specification.md  # Detailed spec document
```

## Debugging and Troubleshooting

### Common Issues

**No learning during training**:
- Check learning rates (`lr_specific`, `lr_aggregate`) - aggregate should be higher
- Verify reward signal is correctly computed (1.0 for correct, 0.0 for incorrect)
- Check that accommodation is working (repeated stimuli should have lower aggregate activity)

**No transfer to novel stimuli**:
- Verify accommodation is enabled (not ablated)
- Verify aggregate pathway is enabled (not ablated)
- Check if specific pathway is dominating (try disabling it to test)
- Analyze aggregate activity: is there clear separation between "same" and "different"?

**Accommodation not working**:
- Check `accommodation_alpha` and `accommodation_tau` parameters
- Verify `update_accommodation=True` during stimulus presentations
- Check `decay_accommodation()` is called at appropriate times
- Print `model.accommodation_state` to inspect values

**Unstable learning**:
- Reduce learning rates
- Check weight clipping (should clip to `[-weight_max, weight_max]`)
- Verify reward baseline is being updated correctly

### Inspection Tools

```python
# Check model state
state = model.get_state()
print(f"W_specific range: {state['W_specific'].min():.2f} to {state['W_specific'].max():.2f}")
print(f"W_aggregate: {state['W_aggregate']}")
print(f"Accommodation state: {state['accommodation_state'][:10]}")  # First 10 units

# Track forward pass details
result = model.forward(stimulus.vector, update_accommodation=False)
print(f"Pre-sparse max: {result.pre_sparse.max():.2f}")
print(f"Active units: {np.sum(result.sparse_rep > 0)}")
print(f"Aggregate activity: {result.aggregate_activity:.2f}")
print(f"Decision: {result.decision}")
```

## Experimental Protocol

The default protocol follows bee experiments:

1. **Familiarization** (10 trials): Expose to task without learning, establish baseline
2. **Training** (60 trials): Learn task with feedback using small stimulus set
3. **Transfer Test** (20 trials): Test on novel stimuli without learning

Success criteria:
- **Learning**: Last block accuracy ≥ 70% during training
- **Transfer**: Transfer accuracy > 50% (significantly above chance)
- **Mechanism validation**: Ablations show both accommodation and aggregate pathway are necessary

## Parameter Tuning Guidelines

When experimenting with hyperparameters:

1. **Start with defaults** - they're based on biological data and work reasonably well
2. **Tune learning rates together** - maintain `lr_aggregate > lr_specific` (usually 10-100×)
3. **Sparsity affects capacity** - more active units (higher k) = more capacity but less separation
4. **Accommodation parameters interact with timing** - if trial delays change, adjust `tau`
5. **Aggregate baseline** should be between expected "same" and "different" aggregate values (check with analyzer)

The specification document (`mushroom_body_generalisation_specification.md`) contains detailed rationale for all parameters.
