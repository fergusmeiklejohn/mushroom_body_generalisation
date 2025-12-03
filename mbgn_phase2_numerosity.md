# MBGN Phase 2: Relational Concept Generalization
## Extending Beyond Same/Different to Numerosity and Other Relations

---

# Executive Summary

Having demonstrated that the Mushroom Body-Inspired Generalization Network (MBGN) can learn same/different concepts that transfer to novel stimuli, we now ask: **Is this a general architecture for relational learning, or a special-purpose same/different detector?**

This document specifies experiments to test whether MBGN can learn other relational concepts, starting with **numerosity** ("which has more?") and potentially extending to **magnitude** ("which is larger/brighter?") and **spatial relations** ("which is above?").

**Core Hypothesis**: The MBGN architecture should be able to learn any relational concept that can be computed from aggregate statistics of the sparse representation. Relations requiring explicit spatial or structural comparison may require architectural extensions.

---

# Part 1: Why Numerosity?

## 1.1 Biological Precedent

Bees and other insects demonstrate numerical cognition:
- Bees can count landmarks to find food sources
- They can learn "choose the stimulus with more/fewer elements"
- They can transfer numerical rules to novel stimuli with different element types
- They show the "numerical distance effect": easier to discriminate 2 vs 8 than 4 vs 5

This suggests the mushroom body architecture supports numerical comparison, not just same/different.

## 1.2 Why Numerosity Should Work with MBGN

The same/different task works because:
1. Accommodation reduces activity for repeated stimuli
2. Aggregate pathway detects this activity difference
3. Rule transfers because accommodation affects all stimuli equally

Numerosity should work through a related mechanism:
1. Stimuli with more elements produce more total activity in the input
2. Random projection preserves this (roughly): more input activity → more expansion activity
3. Aggregate pathway can learn "higher aggregate → choose this" or vice versa
4. Rule transfers because the aggregate-to-numerosity relationship holds for novel stimuli

**Key difference from same/different**: Numerosity doesn't require accommodation—it requires the aggregate pathway to track *absolute* activity levels rather than *relative* change from accommodation.

## 1.3 Why It Might Not Work (Risks)

1. **Sparse coding might equalize**: If k-WTA always selects exactly k units, stimuli with different numerosities might produce identical aggregate activity. *Mitigation*: The values of the top-k units may differ even if count is fixed.

2. **Projection randomness might destroy numerosity signal**: Random projections might not preserve the numerosity→activity relationship reliably. *Mitigation*: With high expansion factor, law of large numbers should help.

3. **Interference with same/different**: If we train numerosity with the same architecture, accommodation might interfere. *Mitigation*: We can test with accommodation disabled, or use longer inter-stimulus intervals.

---

# Part 2: Task Specification

## 2.1 Numerosity Comparison Task

### Basic Structure

```
Trial Structure:
┌─────────────────────────────────────────────────────────────┐
│  Stimulus A        Delay         Stimulus B       Decision  │
│  (N₁ elements)     (brief)       (N₂ elements)    (A or B)  │
└─────────────────────────────────────────────────────────────┘

Reward: Choose the stimulus with MORE elements (or FEWER, depending on condition)
```

### Comparison with Same/Different Task

| Aspect | Same/Different | Numerosity |
|--------|---------------|------------|
| What varies | Stimulus identity | Number of elements |
| What's compared | "Is B same as A?" | "Does B have more than A?" |
| Role of accommodation | Essential (detects repetition) | Potentially harmful (confounds numerosity) |
| Aggregate signal | Activity *reduction* from accommodation | Activity *level* correlating with numerosity |
| Transfer test | Novel stimulus identities | Novel element types/arrangements |

### Task Variants

**Variant A: "Choose More" (CM)**
- Present two stimuli sequentially
- Reward for choosing the one with more elements
- Transfer: Novel element types, never-seen numerosities

**Variant B: "Choose Fewer" (CF)**
- Same structure, reward for choosing fewer
- Tests whether the rule can be inverted

**Variant C: "Match Numerosity" (MN)**
- Present sample with N elements
- Present two choices: one with N, one with M≠N
- Reward for choosing the matching numerosity
- This combines same/different logic with numerosity

## 2.2 Stimulus Design

### Requirements

Numerosity stimuli must:
1. Vary in number of elements (e.g., 2-8 elements)
2. Control for confounds (total area, density, spatial arrangement)
3. Support transfer testing (novel element types)

### Stimulus Representation

**Option A: Binary Patterns with Controlled Density**

```python
def make_numerosity_stimulus(n_elements, n_input=50, element_size=3):
    """
    Create stimulus with n_elements "active regions".
    
    Each element activates `element_size` adjacent input units.
    Total active units = n_elements * element_size
    This creates stimuli where numerosity correlates with total activity.
    """
    stimulus = np.zeros(n_input)
    # Randomly place n_elements non-overlapping elements
    positions = np.random.choice(n_input // element_size, n_elements, replace=False)
    for pos in positions:
        start = pos * element_size
        stimulus[start:start+element_size] = 1.0
    return stimulus
```

**Option B: Gaussian Blobs in 2D (flattened)**

```python
def make_numerosity_stimulus_2d(n_elements, grid_size=10, blob_sigma=0.8):
    """
    Create a 2D image with n_elements Gaussian blobs, then flatten.
    
    Allows for visual interpretation and more naturalistic stimuli.
    """
    image = np.zeros((grid_size, grid_size))
    positions = sample_non_overlapping_positions(n_elements, grid_size)
    for (x, y) in positions:
        image += gaussian_blob(grid_size, x, y, blob_sigma)
    return image.flatten()  # Shape: (grid_size^2,)
```

**Option C: Sparse Binary (Most Controlled)**

```python
def make_numerosity_stimulus_sparse(n_elements, n_input=50):
    """
    Simply activate exactly n_elements randomly chosen input units.
    
    Most direct mapping: numerosity = number of active inputs.
    """
    stimulus = np.zeros(n_input)
    active_indices = np.random.choice(n_input, n_elements, replace=False)
    stimulus[active_indices] = 1.0
    return stimulus
```

**Recommendation**: Start with Option C (sparse binary) for clearest interpretation, then test Option B for richer stimuli.

### Controlling Confounds

To ensure the model learns *numerosity* not *total intensity*:

1. **Fixed element intensity**: Each element contributes the same activation
2. **Variable element positions**: Elements placed randomly (so position pattern varies)
3. **Transfer with different element types**: Training elements ≠ test elements

### Training and Transfer Splits

**Training Set:**
- Numerosities: {2, 3, 5, 6} elements
- Element type A (e.g., specific random seeds for positions)

**Transfer Set (Novel Numerosities):**
- Numerosities: {4, 7} elements (never seen during training)
- Element type A (same type, novel counts)

**Transfer Set (Novel Element Types):**
- Numerosities: {2, 3, 5, 6} elements (same counts as training)
- Element type B (different position patterns/seeds)

**Transfer Set (Full Novel):**
- Numerosities: {4, 7} elements
- Element type B
- This is the hardest test: both count and type are novel

## 2.3 Trial Protocol

### Single Trial

```
1. PRESENT STIMULUS A
   - Generate stimulus with N₁ elements
   - Forward pass through MBGN
   - Record aggregate activity A₁
   - (Optionally disable accommodation for numerosity task)

2. DELAY
   - Brief delay (minimal accommodation decay needed)
   - Or no delay (present simultaneously in different channels)

3. PRESENT STIMULUS B
   - Generate stimulus with N₂ elements (N₂ ≠ N₁)
   - Forward pass through MBGN
   - Record aggregate activity A₂

4. DECISION
   - Compare A₁ and A₂ (or use combined output)
   - Choose stimulus with higher/lower aggregate (depending on task)

5. OUTCOME
   - Reward if chose correctly (more or fewer, depending on task)
   - Update weights via reward-modulated Hebbian learning
```

### Key Difference from Same/Different

In same/different, we present **sample, then choice, then choice** and ask "which matches?"

In numerosity, we present **stimulus A, then stimulus B** and ask "which has more?"

This is a **two-alternative forced choice (2AFC)** on a *comparison* rather than a *matching* task.

### Accommodation Handling

**Option 1: Disable accommodation for numerosity**
- Cleaner test of whether aggregate pathway can learn numerosity
- But diverges from the biological architecture

**Option 2: Keep accommodation, use longer delays**
- More biologically realistic
- Accommodation should decay, leaving numerosity signal intact

**Option 3: Present stimuli simultaneously (two input channels)**
- Avoids accommodation entirely
- Requires architectural modification (two parallel input streams)

**Recommendation**: Start with Option 1 (disable accommodation) to isolate the numerosity mechanism, then test Option 2 to see if the full architecture still works.

---

# Part 3: Architectural Considerations

## 3.1 Modifications Needed

### Minimal Modification (Preferred)

The existing MBGN architecture may work with minimal changes:

```python
class MBGNNumerosity(MBGN):
    def compare_stimuli(self, stim_a, stim_b, use_accommodation=False):
        """
        Compare two stimuli and return which has more elements.
        """
        # Disable accommodation for clean numerosity comparison
        old_accommodation = self.accommodation_state.copy()
        if not use_accommodation:
            self.accommodation_state = np.zeros_like(self.accommodation_state)
        
        # Process both stimuli
        _, info_a = self.forward(stim_a, update_accommodation=False)
        _, info_b = self.forward(stim_b, update_accommodation=False)
        
        # Restore accommodation state
        self.accommodation_state = old_accommodation
        
        # Compare aggregate activities
        agg_a = info_a['aggregate_activity']
        agg_b = info_b['aggregate_activity']
        
        # Decision: which has higher aggregate?
        # (For "choose more" task, higher aggregate should mean more elements)
        return 'A' if agg_a > agg_b else 'B', {
            'agg_a': agg_a,
            'agg_b': agg_b,
            'diff': agg_a - agg_b
        }
```

### Learning Rule Modification

For same/different, we learned:
- "Low aggregate (same) → GO" or "High aggregate (different) → GO"

For numerosity, we need to learn:
- "Higher aggregate → this one has more" 

This might require comparing aggregates directly rather than thresholding:

```python
def update_numerosity(self, chose_a, correct_a, info):
    """
    Update weights based on numerosity comparison.
    
    chose_a: Did the model choose stimulus A?
    correct_a: Was A the correct choice (had more elements)?
    """
    reward = 1.0 if (chose_a == correct_a) else -1.0
    
    # The aggregate pathway should learn that
    # higher aggregate = more elements
    # This is implicitly learned by rewarding correct comparisons
    
    # Update aggregate pathway
    agg_diff = info['agg_a'] - info['agg_b']
    
    # If A had more (correct_a=True) and agg_a > agg_b, that's good
    # If A had more but agg_a < agg_b, that's bad (model should adjust)
    
    # Simple approach: reinforce the aggregate pathway to
    # strengthen the correlation between aggregate and numerosity
    if correct_a:  # A had more elements
        # We want agg_a > agg_b, so reward if that's true
        self.W_aggregate += self.lr_aggregate * reward * np.sign(agg_diff)
```

**Note**: The learning rule for numerosity may need to be different from same/different. This is an open design question.

## 3.2 Potential Issues

### Issue 1: k-WTA May Equalize Activity

If we use hard k-WTA (exactly k units active), then all stimuli produce the same number of active units, destroying the numerosity signal.

**Solutions**:
a. Use the *values* of active units, not just counts
b. Use soft k-WTA (softmax with temperature)
c. Use threshold-based sparsity instead of k-WTA

```python
def soft_k_wta(z, k, temperature=1.0):
    """
    Soft winner-take-all: top-k get boosted, others suppressed but not zeroed.
    """
    topk_indices = np.argpartition(z, -k)[-k:]
    mask = np.zeros_like(z)
    mask[topk_indices] = 1.0
    
    # Soft version: use softmax-weighted mask
    weights = softmax(z / temperature)
    return z * (mask * 0.9 + weights * 0.1)  # Blend hard and soft
```

### Issue 2: Random Projection May Not Preserve Numerosity

If W_proj is very sparse or has high variance, the relationship between input numerosity and expansion-layer aggregate may be noisy.

**Solutions**:
a. Use denser projection (higher connection probability)
b. Use normalized projections (each expansion unit has similar total input weight)
c. Verify empirically before adding complexity

```python
def verify_numerosity_preservation(model, n_trials=100):
    """
    Check that stimuli with more elements produce higher aggregate activity.
    """
    results = []
    for n_elements in range(2, 9):
        for _ in range(n_trials):
            stim = make_numerosity_stimulus(n_elements)
            _, info = model.forward(stim)
            results.append({
                'n_elements': n_elements,
                'aggregate': info['aggregate_activity']
            })
    
    # Check correlation
    df = pd.DataFrame(results)
    correlation = df.groupby('n_elements')['aggregate'].mean().corr(
        pd.Series(range(2, 9))
    )
    return correlation  # Should be positive and high
```

### Issue 3: Overlap Between Numerosity and Same/Different

If we train both tasks with the same model, they might interfere. Same/different relies on accommodation; numerosity might be confounded by it.

**Solutions**:
a. Train separate models for each task (cleanest)
b. Train sequentially and test for interference
c. Train jointly with task-specific output heads

---

# Part 4: Experimental Plan

## 4.1 Experiment 1: Numerosity Baseline

**Question**: Can MBGN learn numerosity comparison with training stimuli?

**Method**:
1. Generate training set: numerosities {2, 3, 5, 6}
2. Train on "choose more" task for 100 trials
3. Test on same numerosities (held-out trials)

**Success criterion**: >75% accuracy on training numerosities

**Analysis**:
- Plot aggregate activity vs. numerosity (should be monotonic)
- Check that model learns to choose higher aggregate

## 4.2 Experiment 2: Numerosity Transfer (Novel Counts)

**Question**: Does numerosity rule transfer to unseen numerosities?

**Method**:
1. Train on {2, 3, 5, 6} elements
2. Test on {4, 7} elements (never seen during training)
3. No learning during transfer

**Success criterion**: >65% accuracy on novel numerosities

**Analysis**:
- Does performance correlate with numerical distance? (2 vs 7 easier than 4 vs 5)
- Is transfer symmetric? (extrapolation to 7 vs interpolation to 4)

## 4.3 Experiment 3: Numerosity Transfer (Novel Element Types)

**Question**: Does numerosity rule transfer to novel stimulus types?

**Method**:
1. Train on element type A (specific position patterns)
2. Test on element type B (different position patterns)
3. Same numerosities as training

**Success criterion**: >70% accuracy on novel element types

**Analysis**:
- Compare to Experiment 1 (same numerosities, same element type)
- Is transfer to novel elements as good as training?

## 4.4 Experiment 4: Full Transfer

**Question**: Does the rule transfer to both novel counts AND novel types?

**Method**:
1. Train on {2, 3, 5, 6} with element type A
2. Test on {4, 7} with element type B

**Success criterion**: >60% accuracy

**This is the critical test** analogous to the same/different transfer experiment.

## 4.5 Experiment 5: Comparison with Same/Different

**Question**: How does numerosity transfer compare to same/different transfer?

**Method**:
1. Run both tasks with identical architecture (except accommodation)
2. Compare transfer accuracies

**Expected outcome**: 
- Similar transfer performance would suggest a general relational mechanism
- Different performance would reveal task-specific factors

## 4.6 Experiment 6: Ablation Study

**Question**: Which components are necessary for numerosity transfer?

**Ablations**:
1. No aggregate pathway → expect failure
2. No sparse expansion (dense projection) → expect worse transfer
3. No k-WTA (dense representation) → expect worse discrimination
4. With accommodation → expect interference (accommodation confounds numerosity)

## 4.7 Experiment 7: "Choose Fewer" Reversal

**Question**: Can the model learn the opposite rule?

**Method**:
1. Train on "choose fewer" instead of "choose more"
2. Test transfer

**Expected outcome**: Should work equally well (just inverts the aggregate comparison)

## 4.8 Experiment 8: Numerical Distance Effect

**Question**: Does the model show the numerical distance effect (like bees and humans)?

**Method**:
1. Train on full range of numerosities
2. Test on all pairs
3. Plot accuracy vs. numerical distance (|N₁ - N₂|)

**Expected outcome**: Accuracy should increase with numerical distance

---

# Part 5: Success Criteria and Interpretation

## 5.1 What Would Constitute Success?

### Strong Success
- Transfer accuracy >70% on novel numerosities AND novel element types
- Performance comparable to same/different transfer
- Clear ablation results showing aggregate pathway is necessary
- Numerical distance effect present

**Interpretation**: MBGN is a general architecture for relational concepts computable from aggregate statistics.

### Moderate Success
- Transfer accuracy >60% on novel numerosities OR novel element types (but not both)
- Ablations show aggregate pathway helps but isn't essential

**Interpretation**: MBGN supports numerosity but less robustly than same/different. Architectural differences matter.

### Weak Success
- Transfer accuracy ~55-60% (slightly above chance)
- Performance much worse than same/different

**Interpretation**: Numerosity transfer is possible but requires modifications. The same/different mechanism doesn't fully generalize.

### Failure
- Transfer accuracy ~50% (chance)
- Model cannot learn numerosity even with training stimuli

**Interpretation**: MBGN architecture is specific to same/different. Numerosity requires different mechanisms.

## 5.2 What Would We Learn from Failure?

Failure would be informative:

1. **If training fails**: Random projection doesn't preserve numerosity signal. Need structured projections.

2. **If transfer fails**: The rule learned is stimulus-specific, not abstract. The aggregate pathway doesn't carry numerosity information in a generalizable way.

3. **If accommodation interferes**: The architecture is tuned for same/different and doesn't flexibly support other relations.

Each failure mode suggests specific architectural modifications.

---

# Part 6: Implementation Checklist

## Phase 1: Stimulus Generation
- [ ] Implement `make_numerosity_stimulus()` (sparse binary)
- [ ] Implement `make_numerosity_stimulus_2d()` (Gaussian blobs)
- [ ] Create training/transfer splits
- [ ] Verify numerosity is not confounded with other factors

## Phase 2: Task Infrastructure
- [ ] Implement `NumerosityTask` class
- [ ] Trial generation with balanced numerosity pairs
- [ ] Reward logic for "choose more" / "choose fewer"
- [ ] Accuracy tracking per numerosity pair

## Phase 3: Model Adaptation
- [ ] Add `compare_stimuli()` method to MBGN
- [ ] Add option to disable accommodation
- [ ] Verify aggregate activity correlates with numerosity
- [ ] Implement numerosity-specific learning rule if needed

## Phase 4: Experiments
- [ ] Experiment 1: Training baseline
- [ ] Experiment 2: Novel numerosity transfer
- [ ] Experiment 3: Novel element type transfer
- [ ] Experiment 4: Full transfer
- [ ] Experiment 5: Comparison with same/different
- [ ] Experiment 6: Ablation study
- [ ] Experiment 7: "Choose fewer" reversal
- [ ] Experiment 8: Numerical distance effect

## Phase 5: Analysis
- [ ] Aggregate vs. numerosity plots
- [ ] Transfer accuracy comparisons
- [ ] Numerical distance effect plot
- [ ] Ablation results table
- [ ] Comparison with MLP baseline

## Phase 6: Documentation
- [ ] Update FINDINGS.md with numerosity results
- [ ] Document any architectural modifications
- [ ] Summarize implications for relational learning

---

# Part 7: Open Questions

Things we don't know and should track:

1. **Should we disable accommodation for numerosity?**
   - Pro: Cleaner signal, avoids confound
   - Con: Diverges from biological architecture
   - Empirical question: Does it actually interfere?

2. **How should the learning rule change?**
   - Same/different: Learn threshold on aggregate
   - Numerosity: Learn comparison between aggregates
   - Might need relative rather than absolute aggregate

3. **What's the right sparsity level?**
   - Same/different worked with k=5%
   - Numerosity might benefit from softer sparsity

4. **How does numerical ratio vs. distance affect transfer?**
   - Bees show both distance and ratio effects
   - Our model might show only one

5. **Can we learn both same/different AND numerosity?**
   - Multi-task learning might require separate output heads
   - Or the aggregate pathway might naturally support both

---

# Appendix A: Relation to Other Relational Concepts

If numerosity works, we can extend to:

## A.1 Magnitude (Size, Brightness)
- Stimulus varies in element size rather than count
- Larger elements → more total activity
- Should work with similar mechanism

## A.2 Spatial Relations (Above, Below, Left, Right)
- Requires spatially-organized input
- Random projection destroys spatial structure
- **May not work** without architectural changes
- Would need structured projections or separate spatial channels

## A.3 Temporal Relations (Before, After, Longer)
- Requires sequence processing
- Current architecture doesn't handle time within stimulus
- Would need recurrence or memory

**Prediction**: MBGN should generalize to relations computable from aggregate statistics (numerosity, magnitude) but not to relations requiring structural alignment (spatial, temporal).

---

# Appendix B: Quick Start Code

```python
# Numerosity stimulus generation
def make_numerosity_stimulus(n_elements, n_input=50):
    stimulus = np.zeros(n_input)
    indices = np.random.choice(n_input, n_elements, replace=False)
    stimulus[indices] = 1.0
    return stimulus

# Numerosity task trial
def run_numerosity_trial(model, n1, n2, target='more'):
    stim_a = make_numerosity_stimulus(n1)
    stim_b = make_numerosity_stimulus(n2)
    
    choice, info = model.compare_stimuli(stim_a, stim_b)
    
    if target == 'more':
        correct = 'A' if n1 > n2 else 'B'
    else:  # 'fewer'
        correct = 'A' if n1 < n2 else 'B'
    
    is_correct = (choice == correct)
    return is_correct, info

# Training loop
def train_numerosity(model, n_trials=100, numerosities=[2,3,5,6]):
    for trial in range(n_trials):
        n1, n2 = np.random.choice(numerosities, 2, replace=False)
        is_correct, info = run_numerosity_trial(model, n1, n2)
        model.update_numerosity(is_correct, info)
```

---

*This specification provides the roadmap for testing whether MBGN is a general relational learning architecture. The key question: Is same/different special, or can the same architectural principles support numerosity and beyond?*
