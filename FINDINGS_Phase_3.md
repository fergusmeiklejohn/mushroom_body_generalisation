# MBGN Phase 3: Finding the Limits - Findings Report

## Abstract

Phase 2 demonstrated that MBGN achieves near-perfect (100%) transfer on numerosity and magnitude discrimination tasks. This raised an important question: where does the architecture actually fail? Phase 3 systematically stress-tests MBGN to identify its fundamental limitations. We find three clear failure modes: (1) spatial position tasks where aggregate activity carries no information, (2) high noise conditions (σ > 0.5) that degrade the numerosity signal, and (3) intensity-count confounds where variable element brightness interferes with pure numerosity discrimination. Surprisingly, MBGN shows remarkable robustness to fine-grained discriminations (10 vs 11 at 97%), large numerosities (15-24 range), small expansion layers (100 units), and all tested sparsity levels. These results define the operational boundaries of the mushroom body-inspired architecture.

## 1. Introduction

### 1.1 Motivation

Phase 2 results were "too good" - achieving 100% accuracy across all conditions makes it impossible to understand the architecture's true capabilities and limitations. A robust scientific characterization requires knowing not just what a system can do, but what it cannot do.

### 1.2 Research Questions

1. **Weber fraction**: What is the finest numerical discrimination MBGN can achieve?
2. **Noise tolerance**: How much stimulus noise can MBGN tolerate?
3. **Confound resistance**: Can MBGN distinguish count from total intensity?
4. **Spatial limits**: Does MBGN fail on tasks requiring spatial information?
5. **Architecture limits**: What are the minimum expansion size and sparsity requirements?

### 1.3 Approach

We designed increasingly difficult tests in four categories:
- **Numerosity limits**: Fine discriminations, large numbers
- **Noise/confound resistance**: Gaussian noise, variable intensity
- **Spatial tasks**: Position discrimination (expected failure)
- **Architecture stress**: Parameter sweeps

## 2. Methods

### 2.1 Fine-Grained Numerosity Discrimination

Instead of comparing 2 vs 7 (ratio 0.29), we test progressively harder ratios:

| Comparison | Ratio | Difficulty |
|------------|-------|------------|
| 2 vs 4 | 0.50 | Easy |
| 3 vs 5 | 0.60 | Easy |
| 4 vs 5 | 0.80 | Medium |
| 5 vs 6 | 0.83 | Hard |
| 6 vs 7 | 0.86 | Hard |
| 7 vs 8 | 0.88 | Very Hard |
| 9 vs 10 | 0.90 | Extreme |
| 10 vs 11 | 0.91 | Extreme |

Weber's law predicts discrimination difficulty scales with ratio, not absolute difference.

### 2.2 Noisy Stimuli

We add Gaussian noise to stimulus activations:

```python
def make_noisy_stimulus(n_elements, noise_std):
    stimulus = np.zeros(n_input)
    indices = np.random.choice(n_input, n_elements, replace=False)
    stimulus[indices] = 1.0
    noise = np.random.normal(0, noise_std, n_input)
    stimulus = np.clip(stimulus + noise, 0, 2)
    return stimulus
```

Noise levels tested: σ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}

### 2.3 Variable Intensity (Count vs Intensity Confound)

Elements have random intensities instead of uniform 1.0:

```python
def make_variable_intensity_stimulus(n_elements, intensity_range):
    stimulus = np.zeros(n_input)
    indices = np.random.choice(n_input, n_elements, replace=False)
    intensities = np.random.uniform(intensity_range[0], intensity_range[1], n_elements)
    stimulus[indices] = intensities
    return stimulus
```

This creates a confound: 3 bright elements (intensity ~0.9 each, total ~2.7) might have similar aggregate activity to 5 dim elements (intensity ~0.5 each, total ~2.5).

### 2.4 Spatial Position Task

An "above/below" discrimination task:

```python
def make_above_stimulus(grid_size=7):
    """Single dot in upper half of grid."""
    vec = np.zeros((grid_size, grid_size))
    row = np.random.randint(0, grid_size // 2)  # Upper half
    col = np.random.randint(0, grid_size)
    vec[row, col] = 1.0
    return vec.flatten()

def make_below_stimulus(grid_size=7):
    """Single dot in lower half of grid."""
    vec = np.zeros((grid_size, grid_size))
    row = np.random.randint(grid_size // 2 + 1, grid_size)  # Lower half
    col = np.random.randint(0, grid_size)
    vec[row, col] = 1.0
    return vec.flatten()
```

Both stimuli have identical aggregate activity (1.0), so the aggregate pathway has no discriminative signal.

### 2.5 Architecture Parameter Sweeps

**Expansion layer size**: N ∈ {50, 100, 200, 500, 1000, 2000, 5000}

**k-WTA sparsity**: fraction ∈ {1%, 2%, 5%, 10%, 20%, 50%}

## 3. Results

### 3.1 Fine-Grained Numerosity Discrimination

| Comparison | Ratio | Accuracy | Std |
|------------|-------|----------|-----|
| 2 vs 4 | 0.50 | 100.0% | 0.0% |
| 3 vs 5 | 0.60 | 100.0% | 0.0% |
| 4 vs 5 | 0.80 | 100.0% | 0.0% |
| 5 vs 6 | 0.83 | 100.0% | 0.0% |
| 6 vs 7 | 0.86 | 99.8% | 0.8% |
| 7 vs 8 | 0.88 | 99.5% | 1.0% |
| 9 vs 10 | 0.90 | 98.0% | 1.9% |
| 10 vs 11 | 0.91 | 97.2% | 2.1% |

**Key Finding**: MBGN maintains >97% accuracy even at ratio 0.91 (10 vs 11). This far exceeds biological Weber fractions (~0.15 for bees). The aggregate pathway provides an almost perfect numerosity signal with sparse binary stimuli.

**Why so good?** With sparse binary stimuli:
- 10 elements → aggregate ≈ 10 units of activity
- 11 elements → aggregate ≈ 11 units of activity
- Difference is deterministic, not probabilistic

### 3.2 Larger Numerosities

| Range | Transfer Accuracy | Std |
|-------|-------------------|-----|
| Small (2-7) | 100.0% | 0.0% |
| Large (8-14) | 100.0% | 0.0% |
| Very Large (15-24) | 99.8% | 0.8% |

**Key Finding**: MBGN scales to large numerosities without degradation. The linear relationship between element count and aggregate activity is preserved regardless of absolute numbers.

### 3.3 Noise Tolerance

| Noise σ | Transfer Accuracy | Std | Status |
|---------|-------------------|-----|--------|
| 0.0 | 100.0% | 0.0% | ✓ Perfect |
| 0.1 | 100.0% | 0.0% | ✓ Perfect |
| 0.2 | 97.2% | 2.6% | ✓ Robust |
| 0.3 | 89.2% | 3.9% | ⚠ Degrading |
| 0.5 | 75.8% | 3.4% | ⚠ Borderline |
| 0.7 | 66.5% | 7.1% | ✗ Failing |
| 1.0 | 58.5% | 9.0% | ✗ Near chance |

**Key Finding**: MBGN tolerates noise up to σ ≈ 0.3 before significant degradation. At σ = 0.5, performance drops to 76% (borderline). At σ ≥ 0.7, performance approaches chance.

**Interpretation**: With σ = 0.5 noise on a signal of 1.0, the noise magnitude is half the signal magnitude. When noise equals or exceeds signal (σ ≥ 1.0), discrimination becomes impossible.

### 3.4 Intensity-Count Confound

| Intensity Range | Transfer Accuracy | Std |
|-----------------|-------------------|-----|
| (0.9, 1.0) Low variance | 100.0% | 0.0% |
| (0.5, 1.0) Medium variance | 99.5% | 1.5% |
| (0.2, 1.0) High variance | 92.3% | 4.2% |
| (0.1, 1.0) Extreme variance | 90.5% | 5.5% |

**Key Finding**: Variable intensity degrades performance to ~90% with extreme variance. This indicates MBGN partially conflates "total intensity" with "element count."

**Why partial success?** Even with (0.1, 1.0) intensity range:
- Expected intensity per element: 0.55
- 5 elements: expected total ≈ 2.75
- 3 elements: expected total ≈ 1.65
- The difference in expected totals still favors higher counts, but with more overlap

### 3.5 Spatial Position Task (Above/Below)

| Task | Accuracy | Std | Status |
|------|----------|-----|--------|
| Above/Below | 51.5% | 4.6% | ✗ Chance |

**Key Finding**: MBGN completely fails at spatial position discrimination, performing at chance level (50%).

**Why failure is expected**: Both "above" and "below" stimuli have exactly one active unit (aggregate = 1.0). The aggregate pathway has zero discriminative information about which unit is active, only that one unit is active.

**Significance**: This confirms the fundamental architectural limitation—the aggregate pathway computes a content-free summary statistic that discards all spatial/structural information.

### 3.6 Expansion Layer Size

| N_expansion | k (at 5%) | Transfer Accuracy | Std |
|-------------|-----------|-------------------|-----|
| 50 | 2-3 | 90.5% | 4.3% |
| 100 | 5 | 98.2% | 1.1% |
| 200 | 10 | 99.8% | 0.8% |
| 500 | 25 | 100.0% | 0.0% |
| 1000 | 50 | 100.0% | 0.0% |
| 2000 | 100 | 100.0% | 0.0% |
| 5000 | 250 | 100.0% | 0.0% |

**Key Finding**: MBGN requires ≥100 expansion units for robust performance (>98%). At N=50 (equal to input size), performance degrades to 90.5%.

**Interpretation**: The expansion layer provides:
1. Dimensionality expansion for pattern separation
2. Averaging over random projections to reduce noise
3. With only 50 units (no expansion), both benefits are lost

### 3.7 k-WTA Sparsity

| Sparsity | k (out of 2000) | Transfer Accuracy | Std |
|----------|-----------------|-------------------|-----|
| 1% | 20 | 100.0% | 0.0% |
| 2% | 40 | 100.0% | 0.0% |
| 5% | 100 | 100.0% | 0.0% |
| 10% | 200 | 100.0% | 0.0% |
| 20% | 400 | 100.0% | 0.0% |
| 50% | 1000 | 100.0% | 0.0% |

**Key Finding**: All tested sparsity levels achieve 100% accuracy. This is surprising—we expected very sparse (1%) or very dense (50%) to degrade performance.

**Why robust?** The numerosity signal (total element count) is preserved regardless of how many expansion units are kept active, because:
- More input → more pre-k-WTA activity → higher post-k-WTA values
- The aggregate (sum of top-k values) still correlates with input numerosity

## 4. Discussion

### 4.1 Summary of Failure Modes

| Category | Failure Condition | Accuracy |
|----------|-------------------|----------|
| Spatial | Above/below position | 51.5% (chance) |
| Noise | σ ≥ 0.7 | 66.5% |
| Noise | σ = 1.0 | 58.5% |
| Confound | Intensity range (0.1, 1.0) | 90.5% |

### 4.2 Summary of Robust Conditions

| Category | Condition | Accuracy |
|----------|-----------|----------|
| Fine discrimination | 10 vs 11 (ratio 0.91) | 97.2% |
| Large numbers | 15-24 range | 99.8% |
| Low noise | σ ≤ 0.2 | 97-100% |
| Expansion size | N ≥ 200 | 99.8-100% |
| Sparsity | All tested (1-50%) | 100% |

### 4.3 Architectural Implications

**What the aggregate pathway CAN do:**
- Compare total activity levels (numerosity, magnitude)
- Generalize to novel stimuli with same aggregate statistics
- Work across wide range of architecture parameters

**What the aggregate pathway CANNOT do:**
- Encode spatial position or structure
- Distinguish count from intensity when intensity varies
- Function in high-noise environments (σ > 0.5)

### 4.4 Comparison with Biological Systems

| Metric | MBGN | Honeybees | Humans |
|--------|------|-----------|--------|
| Weber fraction | ~0.91 | ~0.15 | ~0.15 |
| Noise tolerance | σ < 0.5 | Unknown | Varies |
| Spatial position | Fails | Succeeds | Succeeds |

MBGN's Weber fraction is much higher (better discrimination) than biological systems. This is because:
1. Sparse binary stimuli provide a perfect numerosity signal
2. No perceptual noise in the model
3. The aggregate pathway directly encodes count

Biological systems must extract numerosity from noisy visual input with variable element sizes, positions, and appearances—a much harder problem.

### 4.5 Why Spatial Tasks Fail

The aggregate pathway computes:
```
aggregate = sum(sparse_representation)
```

This is a **permutation-invariant** operation—reordering the elements doesn't change the sum. Spatial position information requires:
```
position = which_elements_are_active(sparse_representation)
```

This requires the specific pathway (pattern-based readout), but since both "above" and "below" stimuli are equally novel to the model, the specific pathway can't help either.

**Fundamental limitation**: Any task requiring spatial/structural discrimination is beyond MBGN's aggregate pathway.

### 4.6 Implications for Same/Different vs Numerosity

Phase 2 showed same/different (~70%) underperforms numerosity (100%). Phase 3 explains why:

| Task | Signal Type | Confounds |
|------|-------------|-----------|
| Same/Different | Relative (accommodation-based) | Accommodation noise, timing |
| Numerosity | Absolute (direct aggregate) | None with sparse binary stimuli |

Same/different requires the accommodation mechanism to create an aggregate difference, introducing noise. Numerosity directly reads the aggregate without this intermediate step.

## 5. Conclusions

### 5.1 Key Findings

1. **Spatial position = fundamental failure**: The aggregate pathway cannot encode "where," only "how much." This is an architectural limitation, not a parameter tuning issue.

2. **Noise threshold ≈ σ = 0.5**: Below this, MBGN is robust. Above this, performance degrades toward chance.

3. **Intensity-count confound exists but is partial**: With extreme intensity variance, accuracy drops to ~90%, indicating MBGN partially learns total intensity rather than pure count.

4. **Weber fraction > 0.9**: MBGN discriminates 10 vs 11 at 97%, far exceeding biological systems. This reflects the idealized nature of sparse binary stimuli.

5. **Expansion layer ≥ 100 required**: Below this, the averaging benefits of random projection are lost.

6. **Sparsity level doesn't matter**: All tested k-WTA levels (1-50%) work perfectly for numerosity.

### 5.2 Recommendations for Future Work

1. **Add perceptual noise** to better match biological conditions
2. **Test spatial extensions** (e.g., additional spatial pathway)
3. **Develop intensity-invariant numerosity** using normalization
4. **Compare with biological Weber fractions** using matched noise levels

### 5.3 Significance

Phase 3 establishes clear boundaries for the mushroom body-inspired architecture:

| ✓ MBGN Excels At | ✗ MBGN Fails At |
|------------------|-----------------|
| Aggregate comparisons | Spatial structure |
| Numerosity (count) | Position discrimination |
| Magnitude (intensity) | High-noise environments |
| Transfer to novel stimuli | Count vs intensity (partial) |

These boundaries define the operational domain where MBGN provides a valid model of insect relational learning.

## Appendix A: Running Phase 3 Experiments

```bash
# Run all Phase 3 tests
python run_phase3_experiments.py --all

# Run specific suites
python run_phase3_experiments.py --suite numerosity_limits
python run_phase3_experiments.py --suite noise_tolerance
python run_phase3_experiments.py --suite spatial_limits
python run_phase3_experiments.py --suite architecture_stress

# Quick test (3 seeds)
python run_phase3_experiments.py --quick
```

## Appendix B: Statistical Details

All experiments run with 10 random seeds. Results reported as mean ± standard deviation.

Significance testing not performed because:
1. Most conditions achieve ceiling (100%) or floor (50%) performance
2. The goal is characterization of limits, not hypothesis testing
3. Effect sizes are obvious from the means

## Appendix C: Code Structure

```
run_phase3_experiments.py     # Main experiment runner
├── run_fine_discrimination_test()
├── run_larger_numerosities_test()
├── run_noise_tolerance_test()
├── run_intensity_confound_test()
├── run_spatial_task_test()
├── run_expansion_size_test()
└── run_sparsity_test()

# New stimulus generators
├── NoisyNumerosityStimulusGenerator
└── VariableIntensityGenerator
```

---

*Phase 3 completed December 2024. Key finding: MBGN's aggregate pathway is robust for aggregate-based comparisons but fundamentally cannot solve tasks requiring spatial structure.*
