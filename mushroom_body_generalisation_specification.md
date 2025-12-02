# Mushroom Body-Inspired Generalisation Network (MBGN)
## A Specification for Building a Minimal Generalising System

---

# Part 1: The Deep Why

## 1.1 The Problem We're Trying to Solve

**Generalisation** is the ability to apply knowledge acquired in one context to novel situations. This is arguably the central challenge in machine learning—not just fitting training data, but extracting principles that transfer.

Current neural networks struggle with a specific kind of generalisation: **relational generalisation**. They can learn "this specific image contains a cat" but struggle with "these two things are the same kind" in a way that transfers to completely novel things.

Humans and even insects with tiny brains do this effortlessly. A bee can learn "choose the thing that matches what you just saw" and apply this rule to stimuli it has never encountered. This is remarkable because:

1. The rule is about a *relationship* (sameness), not about features
2. The rule transfers to novel stimuli without additional training
3. The learning happens in ~60 trials, not millions

## 1.2 Why Current Approaches Fall Short

Standard deep learning architectures have an implicit assumption: compress information through successive layers to extract hierarchical features. This works brilliantly for many tasks but has a fundamental limitation.

**The representation entanglement problem**: In a typical network, a layer's activation pattern conflates:
- What specific stimulus is present (stimulus identity)
- Relationships between stimuli (same/different, larger/smaller)
- Aggregate properties (how novel is this? how strong is the signal?)

When these are entangled, learning about relationships requires learning separate representations for every possible pair of stimuli—which doesn't generalise.

**The sparse coding paradox**: Very sparse representations are good for discrimination (telling things apart) but bad for generalisation (noticing what things have in common). Very dense representations are good for generalisation but cause interference. Biology seems to solve this with *dual pathways*—sparse coding for specifics, aggregate coding for relationships.

## 1.3 Why Look to the Insect Brain?

The insect mushroom body is interesting not because it's "natural" but because it represents a *minimal solution* to the generalisation problem. With only ~200,000 neurons in the entire brain, bees must be computationally efficient. They can't afford to learn millions of specific cases—they need architectural tricks that make generalisation automatic.

The mushroom body architecture has been refined by ~400 million years of evolution for exactly the problems we care about:
- Learning from few examples
- Transferring to novel situations  
- Balancing discrimination and generalisation
- Operating with limited computational resources

If we can identify the *computational principles* (not the biological details), we may be able to build systems that generalise better with less data.

## 1.4 The Core Hypothesis

**Our hypothesis**: The mushroom body's ability to learn abstract concepts emerges from the interaction of three architectural features:

1. **Expansion to sparse representations**: Inputs are projected to a much larger space where only a few units are active. This creates high-dimensional, separable codes.

2. **Dual readout pathways**: One pathway reads stimulus-specific patterns; another reads aggregate activity levels. Relationships are encoded in the *aggregate* pathway, which is stimulus-independent.

3. **Sensory accommodation**: Repeated stimuli produce weaker responses. This creates an implicit "novelty signal" that the aggregate pathway can use to distinguish "same" from "different" without knowing *what* is same or different.

**If this hypothesis is correct**, we should be able to build a simple network with these features that can:
- Learn "choose the matching stimulus" from a training set
- Transfer this rule to completely novel stimuli
- Do so with relatively few training examples

---

# Part 2: The Biological Architecture (Abstracted)

## 2.1 What the Mushroom Body Actually Does

Let's trace the flow of information through the bee's mushroom body:

```
Sensory Input (e.g., odor, visual pattern)
        ↓
[~150-200 Projection Neurons] — dense, overlapping representations
        ↓ (sparse, random-ish connectivity)
[~170,000 Kenyon Cells] — sparse, high-dimensional representations
        ↓                    ↓
[Stimulus-specific      [Aggregate pathway:
 readout via            PCT neurons sum
 plastic KC→EN          KC activity]
 synapses]                   ↓
        ↓               [Learns about
[Specific behaviors]     relationships]
```

### Key Ratios and Numbers

- **Expansion factor**: ~1000x (200 inputs → 170,000 Kenyon cells)
- **Connectivity**: Each Kenyon cell receives from ~7 projection neurons (very sparse)
- **Sparsity of activity**: Only ~5-10% of Kenyon cells active for any stimulus
- **Output neurons**: Very few (~20 types)

### The Accommodation Effect

When a stimulus is presented twice in quick succession:
- First presentation: Full Kenyon cell response
- Second presentation: ~50% reduction in Kenyon cell activity

This decay persists for several minutes but eventually recovers. This is a form of short-term habituation that happens automatically, without any learning signal.

### The Two Pathways

**Pathway 1: Stimulus-Specific (KC → EN)**
- Plastic synapses between Kenyon cells and extrinsic neurons
- Modified by reward signals (dopamine, octopamine)
- Learns "this specific pattern predicts reward"
- Does NOT generalise to novel stimuli

**Pathway 2: Aggregate/Relational (KC → PCT → EN)**
- PCT (protocerebral tract) neurons sum activity across many Kenyon cells
- Respond to *total* activity level, not specific patterns
- Because of accommodation, "same" stimuli produce low PCT activity; "different" stimuli produce high PCT activity
- Learning in PCT→EN synapses generalises because it's based on aggregate activity, not stimulus identity

## 2.2 Why This Architecture Works for Generalisation

The key insight is that the **accommodation + aggregate pathway** combination creates a stimulus-independent signal for "sameness" vs "difference":

| Condition | KC Activity | PCT Activity | Signal |
|-----------|-------------|--------------|--------|
| New stimulus | High | High | "DIFFERENT" |
| Repeated stimulus | Low (accommodated) | Low | "SAME" |

This works for *any* stimulus because:
1. Accommodation affects all stimuli equally
2. PCT neurons don't care *which* KCs are active, just *how many*
3. The "same/different" distinction is in the *magnitude*, not the pattern

The organism can then learn: "When PCT activity is low (same), do X" or "When PCT activity is high (different), do Y"—and this learning transfers to novel stimuli automatically.

---

# Part 3: The Computational Model

## 3.1 Design Principles

Based on the biological architecture, our model should embody these principles:

### Principle 1: Expand, Then Sparsify

**What**: Project low-dimensional inputs to a much higher-dimensional space, then enforce sparsity so only k units are active.

**Why**: This creates representations that are:
- High-dimensional (good for learning capacity)
- Sparse (good for discrimination, low interference)
- Overlapping in a structured way (similar inputs share some active units)

**Implementation**: Random (or learned) projection followed by winner-take-all or k-sparse activation.

### Principle 2: Separate Specific from Aggregate

**What**: Have two readout pathways—one that sees the full sparse pattern, another that only sees a scalar summary (total activation).

**Why**: 
- Specific pathway: Can learn stimulus-specific associations
- Aggregate pathway: Can learn relationship-based rules that generalise

**Implementation**: 
- Specific: Standard linear readout from sparse layer
- Aggregate: Sum of sparse layer activity → separate readout

### Principle 3: Build in Accommodation

**What**: When a pattern is presented, temporarily reduce the responsiveness of the units that fired.

**Why**: This creates an automatic "novelty detector"—same stimuli produce weaker responses, different stimuli produce full responses.

**Implementation**: Short-term depression of active units, with exponential recovery.

### Principle 4: Local Learning Rules

**What**: Use learning rules that depend only on locally available information (pre-synaptic activity, post-synaptic activity, reward signal).

**Why**: 
- Biological plausibility
- Forces the architecture to do the work (can't rely on clever gradient routing)
- May have regularisation benefits

**Implementation**: Reward-modulated Hebbian learning.

## 3.2 Network Architecture

```
INPUT LAYER (N_in units)
    │
    │ W_proj: random sparse projection matrix
    ↓
EXPANSION LAYER (N_exp units, where N_exp >> N_in)
    │
    │ k-WTA: keep only top k active
    ↓
SPARSE REPRESENTATION (same N_exp units, but only k active)
    │
    ├───────────────────────┬────────────────────────────┐
    │                       │                            │
    ↓                       ↓                            ↓
SPECIFIC PATHWAY      AGGREGATE PATHWAY            ACCOMMODATION
W_specific            sum(sparse_rep)              STATE (per-unit)
    │                       │                            │
    ↓                       ↓                            │
SPECIFIC OUTPUT       W_aggregate                        │
(N_out units)              │                            │
    │                       ↓                            │
    │                 AGGREGATE OUTPUT                   │
    │                 (N_out units)                      │
    │                       │                            │
    ↓                       ↓                            ↓
    └───────────────────────┴────────────────────────────┘
                            │
                     COMBINED OUTPUT
                     (decision/behavior)
```

### Layer Specifications

**Input Layer**
- Size: N_in (e.g., 50 for odor-like inputs, or flattened simple images)
- Activation: Continuous values [0, 1] or binary

**Projection (W_proj)**
- Shape: (N_exp, N_in)
- Initialisation: Sparse random, with ~p_conn probability of non-zero connection
- Values: Binary (0/1) or small positive weights
- Trainable: No (at least initially—this is the "random projection" assumption)

**Expansion Layer (pre-sparsification)**
- Size: N_exp (e.g., N_in × 40 = 2000)
- Computation: z = W_proj @ x (optionally with bias)

**Winner-Take-All / k-Sparse**
- Keep only the top k units active (e.g., k = 0.05 × N_exp = 100)
- Alternative: Use a global inhibition mechanism (like APL neuron)

**Sparse Representation**
- Size: N_exp, but only k non-zero
- This is the "Kenyon cell" layer

**Accommodation State**
- Shape: (N_exp,) — one value per expansion unit
- Dynamics: When a unit fires, its accommodation state increases; decays exponentially over time
- Effect: The effective activation is reduced by the accommodation state

**Specific Pathway**
- W_specific: (N_out, N_exp), dense, trainable
- Computes: out_specific = W_specific @ sparse_rep
- Learns via reward-modulated Hebbian rule or simple delta rule

**Aggregate Pathway**
- Aggregation: a_total = sum(sparse_rep) or mean(sparse_rep)
- W_aggregate: (N_out, 1), trainable
- Computes: out_aggregate = W_aggregate × a_total
- Learns via reward-modulated Hebbian rule

**Output Combination**
- Could be: out_final = out_specific + out_aggregate
- Or: learned gating between pathways
- Or: separate outputs for different types of tasks

## 3.3 Accommodation Dynamics

The accommodation mechanism is crucial for the same/different computation. Here's the formal specification:

**State variable**: For each expansion unit i, maintain accommodation state h_i(t)

**Update rule**:
```
On each stimulus presentation:
    # Compute raw activation
    z = W_proj @ x
    
    # Apply accommodation
    z_accommodated = z * (1 - h)  # element-wise
    
    # Apply k-WTA to get sparse representation
    sparse_rep = k_WTA(z_accommodated)
    
    # Update accommodation state
    h = h + α * sparse_rep  # increase for active units
    
Between presentations (or continuously):
    h = h * exp(-dt / τ)  # exponential decay with time constant τ
```

**Parameters**:
- α: accommodation increment (how much activation adds to accommodation)
- τ: accommodation time constant (how fast it decays)
- Typical values: α ≈ 0.5 (to get ~50% reduction on repeat), τ ≈ 60 seconds

## 3.4 Learning Rules

### Specific Pathway Learning

For the KC → EN (specific) pathway, we use reward-modulated Hebbian learning:

```
ΔW_specific[i,j] = η_s * (R - R_baseline) * post[i] * pre[j]
```

Where:
- η_s: learning rate for specific pathway
- R: reward signal (1 for reward, 0 for no reward, -1 for punishment)
- R_baseline: running average of reward (so we learn from deviation)
- post[i]: activation of output unit i
- pre[j]: activation of sparse representation unit j

This learns stimulus-specific associations.

### Aggregate Pathway Learning

For the PCT → EN (aggregate) pathway:

```
ΔW_aggregate[i] = η_a * (R - R_baseline) * post[i] * a_total
```

Where:
- η_a: learning rate for aggregate pathway
- a_total: sum of sparse representation activity

This learns relationships based on total activity level.

### Why This Works for Same/Different

Consider the same/different task:

**DMTS (Delayed Match to Sample) — reward for "same"**:
- Sample A presented → full KC response → high a_total
- Match A presented → accommodated KC response → low a_total
- If bee chooses Match A and gets reward, the aggregate pathway learns:
  "low a_total → GO" (because low a_total predicts reward)

**DNMTS (Delayed Non-Match to Sample) — reward for "different"**:
- Sample A presented → full KC response → high a_total
- Non-match B presented → full KC response (different units) → high a_total
- If bee chooses Non-match B and gets reward, the aggregate pathway learns:
  "high a_total → GO"

**Transfer to novel stimuli**:
- Novel sample C presented → full KC response → high a_total
- Novel match C presented → accommodated response → low a_total
- The already-learned rule "low a_total → GO" applies!

The key is that a_total is a *scalar* that doesn't depend on stimulus identity.

---

# Part 4: The Task Specification

## 4.1 The Same/Different Task

We will implement a delayed match-to-sample (DMTS) and delayed non-match-to-sample (DNMTS) task, following the experimental protocol used with bees.

### Task Structure

```
DMTS (Match to Sample):

  Sample Phase     →    Choice Phase     →    Outcome
  ───────────────────────────────────────────────────────
  Show stimulus A        Show A and B         Reward if
  (no choice)            (agent chooses)      agent chose A
                                              (the match)

DNMTS (Non-Match to Sample):

  Sample Phase     →    Choice Phase     →    Outcome
  ───────────────────────────────────────────────────────
  Show stimulus A        Show A and B         Reward if
  (no choice)            (agent chooses)      agent chose B
                                              (the non-match)
```

### Trial Sequence (detailed)

1. **Sample presentation**
   - Present stimulus S from training set
   - Network processes S, generating sparse representation
   - Accommodation is induced in active units
   - No decision required; no reward

2. **Delay** (optional)
   - Brief delay (modeled as discrete timesteps)
   - Accommodation decays slightly but persists

3. **Choice: First option**
   - Present stimulus C1 (either S or a different stimulus D)
   - Network processes C1
   - Network produces GO/NOGO output
   - If GO: trial ends, proceed to outcome
   - If NOGO: proceed to second option

4. **Choice: Second option** (if first was NOGO)
   - Present stimulus C2 (the other option)
   - Network produces GO/NOGO output
   - If GO: trial ends, proceed to outcome
   - If NOGO: forced choice (randomly assigned)

5. **Outcome**
   - DMTS: Reward if chosen stimulus matches sample
   - DNMTS: Reward if chosen stimulus differs from sample
   - Update weights based on outcome

### Training Protocol

Following the bee experiments:

1. **Pre-training familiarisation** (10 trials)
   - Expose to apparatus without learning
   - Establishes baseline accommodation

2. **Training phase** (60 trials)
   - Use a fixed set of 2-4 training stimuli
   - Balanced presentation (each stimulus appears as sample equally)
   - Balanced positioning (correct answer on left/right equally)

3. **Transfer test** (without additional training)
   - Use completely novel stimuli
   - No reward given (or reward given but not used for learning)
   - Measure: Does performance transfer?

### Success Criteria

- **Learning**: Performance improves from chance (50%) to ~75%+ during training
- **Transfer**: Performance on novel stimuli is significantly above chance
- **Comparison**: Transfer performance should be comparable to training performance

## 4.2 Stimulus Design

### Requirements

Stimuli must be:
- Distinguishable (produce different sparse representations)
- Not pre-associated with reward (learned during task)
- Generalisable (transfer stimuli should be "novel but fair")

### Option 1: Abstract Binary Patterns

Simple and interpretable:
- Input: Binary vector of length N_in (e.g., 50)
- Each stimulus: Random binary pattern with ~50% ones
- Training set: 4 patterns (A, B, C, D)
- Transfer set: 4 new patterns (E, F, G, H)

Advantage: Easy to analyse, clear separation between training and transfer.

### Option 2: Coloured Shapes

More similar to bee experiments:
- Input: Flattened low-resolution image (e.g., 8×8×3 = 192)
- Training set: {blue circle, yellow square, red triangle, green star}
- Transfer set: {purple hexagon, orange diamond, etc.}

Advantage: More intuitive, tests generalisation across visual categories.

### Option 3: Parametric Stimuli

Controlled similarity:
- Input: Points in a latent space, mapped to input via random projection
- Training set: Stimuli from one region of space
- Transfer set: Stimuli from a different region

Advantage: Can systematically vary similarity between training and transfer.

**Recommendation**: Start with Option 1 (binary patterns) for initial testing, then move to Option 2 for interpretability.

---

# Part 5: Implementation Plan

## 5.1 Phase 1: Core Architecture

Build the minimal network with:

1. **Input encoding**
   - Function to generate/load stimuli
   - Normalisation to [0, 1]

2. **Expansion layer**
   - Random sparse projection matrix
   - No training of projections initially

3. **k-Winner-Take-All**
   - Efficient implementation
   - Parameterised k as fraction of layer size

4. **Accommodation mechanism**
   - Per-unit state
   - Update on each forward pass
   - Decay between presentations

5. **Dual readout**
   - Specific pathway (sparse pattern → output)
   - Aggregate pathway (sum of activity → output)

6. **Output and decision**
   - Combine pathways
   - Softmax or threshold for GO/NOGO decision

**Verification**: 
- Check that different stimuli produce different sparse codes
- Check that repeated stimuli produce reduced total activity
- Check that similar stimuli have overlapping sparse codes

## 5.2 Phase 2: Learning Rules

Implement and verify:

1. **Reward-modulated Hebbian learning**
   - For both pathways
   - Reward baseline tracking

2. **Separate learning rates**
   - η_specific for specific pathway
   - η_aggregate for aggregate pathway
   - May need η_aggregate > η_specific (aggregate should learn faster)

3. **Weight constraints**
   - Non-negative weights (if biologically plausible)
   - Weight bounds to prevent explosion

**Verification**:
- Train on simple association task (stimulus A → reward)
- Verify that specific pathway learns
- Verify that aggregate pathway responds to novelty

## 5.3 Phase 3: Task Implementation

Build the DMTS/DNMTS task:

1. **Trial generator**
   - Sample phase, choice phase, outcome
   - Balanced stimulus/position assignment

2. **Episode runner**
   - Run through trial sequence
   - Track accommodation state across phases
   - Apply learning at outcome

3. **Metrics**
   - Accuracy per block (e.g., 10 trials)
   - Choice reaction (GO on first vs second option)
   - Transfer accuracy

**Verification**:
- Run DMTS with learning, check that accuracy improves
- Run DNMTS with learning, check that accuracy improves
- Compare learning curves to bee data

## 5.4 Phase 4: Transfer Testing

The critical test:

1. **Train on stimulus set A** (e.g., patterns 1-4)
2. **Test on stimulus set B** (patterns 5-8), without learning
3. **Measure transfer accuracy**

**Comparisons**:
- Model with accommodation → expect transfer
- Model without accommodation → expect no transfer
- Model without aggregate pathway → expect no transfer
- Baseline: Standard MLP → expect no transfer

## 5.5 Phase 5: Analysis and Ablations

Understand *why* the model works (or doesn't):

1. **Ablations**
   - Remove accommodation
   - Remove aggregate pathway
   - Remove specific pathway
   - Vary expansion ratio
   - Vary sparsity level

2. **Representations**
   - Visualise sparse codes for different stimuli
   - Measure overlap between similar/different stimuli
   - Track accommodation state dynamics

3. **Learning dynamics**
   - Which pathway learns faster?
   - How do weights evolve?
   - What's the role of the baseline?

---

# Part 6: Expected Challenges and Fallback Strategies

## 6.1 Anticipated Challenges

### Challenge 1: Sparsity-Learning Tradeoff

**Problem**: Very sparse representations may not have enough gradient signal for learning.

**Why it might happen**: If only 5% of units are active, and we're using local learning rules, there may be too little "credit" to go around.

**Fallback strategies**:
- Increase k (sparsity level) until learning works, then gradually decrease
- Use soft k-WTA (e.g., softmax with temperature) instead of hard k-WTA
- Add eligibility traces to spread credit over time

### Challenge 2: Accommodation Parameter Sensitivity

**Problem**: Accommodation might be too strong (everything looks "same") or too weak (everything looks "different").

**Why it might happen**: The τ and α parameters interact with the trial timing in non-obvious ways.

**Fallback strategies**:
- Systematic parameter sweep
- Adaptive accommodation (learn α from data)
- Multiple accommodation timescales

### Challenge 3: Pathway Interference

**Problem**: Specific and aggregate pathways might interfere with each other.

**Why it might happen**: If both pathways drive the same output, their gradients mix.

**Fallback strategies**:
- Separate outputs for each pathway, combined only at decision time
- Train pathways in alternating phases
- Different learning rates to let one pathway stabilise first

### Challenge 4: No Transfer

**Problem**: The model learns DMTS/DNMTS but doesn't transfer to novel stimuli.

**Why it might happen**: 
- Specific pathway dominates
- Aggregate pathway isn't learning the right thing
- Accommodation isn't producing a clean novelty signal

**Fallback strategies**:
- Block specific pathway during testing (force reliance on aggregate)
- Analyse aggregate pathway weights—what did it learn?
- Check accommodation dynamics—is repeated stimulus really producing lower total activity?

### Challenge 5: Biological Plausibility vs. Performance

**Problem**: Strict biological plausibility (local learning, non-negative weights) may limit performance.

**Why it might happen**: Backpropagation is powerful precisely because it routes credit through many layers.

**Fallback strategies**:
- Start with backprop for prototyping, then constrain
- Use "biologically plausible backprop" approximations
- Accept some non-biological elements if they don't change the key dynamics

## 6.2 Key Decision Points

At each phase, we need to decide whether to:
- **Proceed**: Results are promising, continue to next phase
- **Iterate**: Results are mixed, try variations on current phase
- **Pivot**: Results suggest fundamental issue, reconsider architecture

### Decision metrics:

| Phase | Proceed if... | Iterate if... | Pivot if... |
|-------|--------------|---------------|-------------|
| 1 (Architecture) | Different stimuli → different codes, repeated stimuli → reduced activity | Code separation or accommodation is weak | No code separation despite parameter tuning |
| 2 (Learning) | Simple association is learned | Learning is slow or unstable | No learning after extensive tuning |
| 3 (Task) | DMTS/DNMTS learned above chance | Learning is slow relative to bees | Cannot learn task at all |
| 4 (Transfer) | Transfer accuracy > 60% | Some transfer but weak | No transfer |

---

# Part 7: Success Criteria and Evaluation

## 7.1 Minimum Success

The experiment is a success if we demonstrate:

1. **Learning**: Model learns DMTS or DNMTS from chance to ~70%+ in ≤100 trials
2. **Transfer**: Performance on novel stimuli is significantly above chance (p < 0.05)
3. **Mechanism**: Ablation shows that accommodation + aggregate pathway are necessary for transfer

This would validate the core hypothesis that the mushroom body architecture enables relational generalisation.

## 7.2 Strong Success

A strong result would additionally show:

1. **Efficiency**: Learning in fewer trials than a comparable MLP
2. **Robustness**: Works across different stimulus types (patterns, images)
3. **Scalability**: Works with larger input dimensions
4. **Interpretability**: Clear explanation of *how* the architecture achieves transfer

## 7.3 Unexpected Outcomes

We should also be prepared for:

**Null result**: Model doesn't transfer. This would be informative—it might mean:
- Our abstraction missed a crucial biological detail
- The task doesn't capture what bees are actually doing
- Additional mechanisms are needed

**Surprising result**: Model transfers for reasons we didn't expect. For example:
- Specific pathway alone might be sufficient (if expansion creates systematic similarity structure)
- Accommodation might not be necessary (if sparsity alone is enough)

Either way, we learn something about what's necessary for generalisation.

---

# Part 8: Technical Appendix

## 8.1 Pseudocode for Core Forward Pass

```python
def forward(self, x, update_accommodation=True):
    """
    x: input stimulus, shape (N_in,)
    Returns: GO/NOGO decision and internal states
    """
    # 1. Random sparse projection to expansion layer
    z = self.W_proj @ x  # shape (N_exp,)
    
    # 2. Apply accommodation (reduce activity of recently-active units)
    z_accommodated = z * (1.0 - self.accommodation_state)
    
    # 3. k-Winner-Take-All: keep only top k units
    sparse_rep = self.k_wta(z_accommodated, k=self.k)  # shape (N_exp,), mostly zeros
    
    # 4. Update accommodation state for units that fired
    if update_accommodation:
        self.accommodation_state = self.accommodation_state + self.alpha * sparse_rep
    
    # 5. Specific pathway: pattern-based readout
    out_specific = self.W_specific @ sparse_rep  # shape (N_out,)
    
    # 6. Aggregate pathway: sum-based readout
    aggregate_activity = sparse_rep.sum()
    out_aggregate = self.W_aggregate * aggregate_activity  # shape (N_out,)
    
    # 7. Combine pathways
    out_combined = out_specific + out_aggregate  # shape (N_out,)
    
    # 8. Decision (e.g., GO if output > threshold)
    decision = out_combined > self.threshold
    
    return decision, {
        'sparse_rep': sparse_rep,
        'aggregate_activity': aggregate_activity,
        'out_specific': out_specific,
        'out_aggregate': out_aggregate,
        'out_combined': out_combined
    }

def decay_accommodation(self, dt):
    """Call between trials or during delays."""
    self.accommodation_state = self.accommodation_state * np.exp(-dt / self.tau)
```

## 8.2 Pseudocode for Learning Update

```python
def update_weights(self, reward, internal_states, prev_sparse_rep):
    """
    reward: scalar, 1 for reward, 0 for none, -1 for punishment
    internal_states: dict from forward pass
    prev_sparse_rep: sparse representation from previous stimulus (for eligibility)
    """
    # Update reward baseline (running average)
    self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
    reward_delta = reward - self.reward_baseline
    
    # Specific pathway update (Hebbian: post * pre * reward)
    # post = output activation, pre = sparse representation
    post = internal_states['out_specific']  # or decision
    pre = internal_states['sparse_rep']
    delta_W_specific = self.eta_specific * reward_delta * np.outer(post, pre)
    self.W_specific = self.W_specific + delta_W_specific
    
    # Aggregate pathway update
    agg = internal_states['aggregate_activity']
    delta_W_aggregate = self.eta_aggregate * reward_delta * post * agg
    self.W_aggregate = self.W_aggregate + delta_W_aggregate
    
    # Optional: clip weights to prevent explosion
    self.W_specific = np.clip(self.W_specific, -self.w_max, self.w_max)
    self.W_aggregate = np.clip(self.W_aggregate, -self.w_max, self.w_max)
```

## 8.3 Suggested Hyperparameters (Initial Values)

Based on biological data and preliminary estimates:

| Parameter | Symbol | Initial Value | Rationale |
|-----------|--------|---------------|-----------|
| Input dimension | N_in | 50 | Similar to # of olfactory glomeruli |
| Expansion dimension | N_exp | 2000 | ~40× expansion (as in fly/bee) |
| Sparsity (fraction) | k/N_exp | 0.05 | ~5% active (as in Kenyon cells) |
| Connection probability | p_conn | 0.02 | Each expansion unit from ~1-2% of inputs |
| Accommodation increment | α | 0.5 | To get ~50% reduction on repeat |
| Accommodation time constant | τ | 60.0 | Seconds; matches bee data |
| Specific pathway LR | η_specific | 0.01 | Standard |
| Aggregate pathway LR | η_aggregate | 0.02 | Slightly faster (to learn rule) |
| Reward baseline decay | - | 0.9 | Running average |
| Weight max | w_max | 10.0 | Prevent explosion |

## 8.4 Dependencies

Minimal requirements:
- Python 3.8+
- NumPy (core operations)
- Matplotlib (visualisation)

Optional:
- PyTorch (for GPU acceleration, gradient checking)
- Scipy (for statistical tests)

---

# Part 9: Open Questions

These are things we don't know yet and should track:

1. **Is random projection sufficient, or do we need learned projections?**
   - Fly/bee may tune projections during development
   - Could add slow learning to W_proj

2. **What's the right form of k-WTA?**
   - Hard k-WTA (exactly k active)
   - Soft k-WTA (top k get boost, others suppressed but not zero)
   - Threshold-based (all above threshold)

3. **Should accommodation be additive or multiplicative?**
   - Additive: z_accommodated = z - h
   - Multiplicative: z_accommodated = z * (1 - h)
   - Biological evidence suggests multiplicative

4. **How should pathways be combined?**
   - Simple addition
   - Learned gating
   - Separate outputs with combined decision

5. **What if we need more than binary output?**
   - Current design is GO/NOGO
   - Could extend to multiple output channels

6. **How does this scale?**
   - With input dimension?
   - With number of training stimuli?
   - With task complexity?

---

# Part 10: Next Steps

## Immediate (This Session)

1. Implement core architecture in Python/NumPy
2. Verify basic functionality (code generation, accommodation)
3. Implement DMTS task

## Short-Term (Next Sessions)

4. Train on DMTS, verify learning
5. Test transfer to novel stimuli
6. Run ablations (remove accommodation, remove aggregate pathway)

## Medium-Term

7. Try DNMTS task
8. Try different stimulus types
9. Compare to baseline MLP
10. Write up findings

---

*This specification is a living document. As we implement and test, we will update it with findings, parameter changes, and new insights.*
