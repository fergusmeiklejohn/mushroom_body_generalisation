# MBGN Phase 3: Finding the Limits

## 1. Why Current Tests Are Too Easy

Phase 2 achieved near-perfect (100%) accuracy on numerosity and magnitude tasks. This suggests we haven't found the architecture's limits. Key reasons:

### 1.1 The Aggregate Signal Is Too Clean

**Sparse binary stimuli**: With `n_elements` active units each set to 1.0:
- Total activity = exactly `n_elements`
- No noise, no overlap, perfect signal
- After k-WTA, aggregate still correlates r=0.99+ with numerosity

**Result**: The discrimination is trivially easy. The model just needs to learn "higher aggregate = more elements."

### 1.2 Numerical Differences Are Too Large

Current setup uses numerosities {2, 3, 5, 6} for training and {4, 7} for transfer:
- Minimum difference: 1 (e.g., 2 vs 3)
- Maximum difference: 5 (e.g., 2 vs 7)
- Average trial difference: ~2.5

**Problem**: Even with noise, a difference of 2-3 elements is easily detectable.

### 1.3 Ablation Doesn't Differentiate

The "no aggregate pathway" ablation still achieves 100% because:
- The specific pathway can also learn from pattern correlations
- The task is easy enough that degraded performance still suffices

---

## 2. Proposed Harder Tests

### 2.1 Fine-Grained Numerosity Discrimination

**Test**: Can MBGN discriminate 5 vs 6 (difference of 1)?

```
Training: {4, 5, 6, 7}
Transfer: {5 vs 6 only}

Prediction: Accuracy should drop significantly (maybe 60-80%)
Why: Weber's law suggests discrimination difficulty increases
     as ratio approaches 1.0 (5/6 = 0.83 vs 2/7 = 0.29)
```

**Expected insight**: Find the Weber fraction for MBGN.

### 2.2 Noisy Stimuli

**Test**: Add Gaussian noise to stimulus activations.

```python
def make_noisy_stimulus(n_elements, noise_std=0.2):
    stimulus = np.zeros(n_input)
    indices = np.random.choice(n_input, n_elements, replace=False)
    stimulus[indices] = 1.0 + np.random.normal(0, noise_std, n_elements)
    stimulus = np.clip(stimulus, 0, 2)  # Keep bounded
    return stimulus
```

**Parameters to test**:
- `noise_std`: 0.1, 0.2, 0.3, 0.5
- Measure accuracy degradation curve

**Expected insight**: Find noise tolerance threshold.

### 2.3 Overlapping Element Activation

**Test**: Elements activate overlapping input units (like real visual features).

```python
def make_overlapping_stimulus(n_elements, element_size=5, n_input=50):
    """Each element activates a random subset of units,
    but subsets can overlap."""
    stimulus = np.zeros(n_input)
    for _ in range(n_elements):
        indices = np.random.choice(n_input, element_size, replace=True)
        stimulus[indices] += 1.0
    return stimulus / stimulus.max()  # Normalize
```

**Problem this creates**: Total activity no longer equals n_elements.

### 2.4 Variable Element Intensity

**Test**: Elements have random intensities (not all 1.0).

```python
def make_variable_intensity_stimulus(n_elements, intensity_range=(0.3, 1.0)):
    stimulus = np.zeros(n_input)
    indices = np.random.choice(n_input, n_elements, replace=False)
    intensities = np.random.uniform(*intensity_range, n_elements)
    stimulus[indices] = intensities
    return stimulus
```

**Confound**: Now 3 bright elements might have same total activity as 5 dim ones.

**Expected insight**: Can MBGN learn numerosity (count) vs total intensity?

### 2.5 Larger Numerosities

**Test**: Extend to larger counts where Weber's law makes discrimination harder.

```
Training: {8, 10, 12, 14}
Transfer: {9, 11, 13}

Trials: 10 vs 11, 12 vs 13, etc.
```

**Expected insight**: At what numerosity does performance collapse?

### 2.6 Spatial Structure Requirements

**Test**: Relations that require spatial information, not just aggregate.

**Above/Below task**:
- Stimulus: 2D grid with one dot above or below center
- Task: Is the dot above or below?
- **Prediction**: MBGN should FAIL because aggregate carries no spatial info

**Symmetry task**:
- Stimulus: 2D pattern that is symmetric or asymmetric
- Task: Is the pattern symmetric?
- **Prediction**: MBGN should FAIL

These would confirm what the aggregate pathway CANNOT do.

### 2.7 Temporal Relations

**Test**: Same temporal structure as same/different, but for numerosity.

**Sequence numerosity**:
```
Trial: Present N1 dots, delay, present N2 dots
Task: Did the second have more or fewer?
With accommodation enabled
```

**Expected insight**: Does accommodation help or hurt numerosity?

### 2.8 Interference Between Tasks

**Test**: Create competing tasks that use the same pathway differently.

**Numerosity + Intensity conflict**:
```
Phase 1: Train "choose more elements"
Phase 2: Train "choose brighter" (with equal element counts)
Test: Both tasks
```

**Prediction**: Learning the second task should hurt the first (unlike SD + Num which use different mechanisms).

---

## 3. Architectural Weaknesses to Probe

### 3.1 Sparse Projection Randomness

**Current**: Fixed random projection W_proj set at initialization.

**Test**: How much does performance vary across random seeds?

```python
# Run same experiment with 100 different seeds
# Measure variance in transfer accuracy
# Some seeds might fail badly
```

### 3.2 k-WTA Threshold Sensitivity

**Current**: Fixed k = 5% of expansion layer (100 units out of 2000).

**Tests**:
- k = 1% (very sparse): Does signal survive?
- k = 20% (less sparse): Does specificity suffer?
- k = 50% (half): Should break numerosity

### 3.3 Expansion Layer Size

**Current**: 2000 units.

**Tests**:
- 200 units: Still work?
- 100 units: Breaking point?
- 50 units (same as input): Minimal expansion

### 3.4 Accommodation Parameters

**Current**: α=0.7, τ=30s.

**Test for same/different**:
- α=0.3 (weak accommodation): Should hurt SD
- α=0.95 (strong accommodation): Might also hurt
- τ=1s (fast decay): Should hurt SD (signal fades before test)

---

## 4. Proposed Experiment Structure

### Phase 3a: Find Numerosity Limits
1. Weber fraction estimation (fine discriminations)
2. Noise tolerance curve
3. Confound resistance (intensity vs count)

### Phase 3b: Find Spatial Limits
1. Above/below task (expected failure)
2. Symmetry task (expected failure)
3. Document what aggregate pathway CANNOT do

### Phase 3c: Architecture Stress Tests
1. Seed variance analysis
2. k-WTA sensitivity
3. Expansion layer size limits
4. Parameter sensitivity analysis

### Phase 3d: Task Interference
1. Conflicting aggregate rules
2. Multi-task capacity limits
3. Catastrophic forgetting conditions

---

## 5. Success Criteria

**Phase 3 is successful if we can identify:**

1. **Weber fraction**: The smallest numerical difference MBGN can reliably discriminate
2. **Noise threshold**: Maximum noise where numerosity still works
3. **Failure modes**: At least 2 tasks that MBGN fundamentally cannot solve
4. **Architecture limits**: Minimum expansion layer size, critical k-WTA range
5. **Interference conditions**: Scenarios that cause catastrophic forgetting

---

## 6. Implementation Priority

| Priority | Test | Rationale |
|----------|------|-----------|
| 1 | Fine-grained numerosity (5 vs 6) | Most direct test of limits |
| 2 | Noisy stimuli | Real-world relevance |
| 3 | Above/below spatial task | Confirm aggregate limitation |
| 4 | Variable intensity confound | Test for true numerosity learning |
| 5 | Architecture parameter sweeps | Understand robustness |
| 6 | Task interference | Multi-task limits |

---

## 7. Quick Start Commands

```bash
# Run Phase 3a experiments (to be implemented)
python run_phase3_experiments.py --suite numerosity_limits

# Run Phase 3b experiments
python run_phase3_experiments.py --suite spatial_limits

# Run Phase 3c experiments
python run_phase3_experiments.py --suite architecture_stress

# Run all Phase 3
python run_phase3_experiments.py --all
```

---

*Phase 3 focuses on finding failures, not successes. A good Phase 3 identifies clear boundaries of MBGN capabilities.*
