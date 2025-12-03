"""
Mushroom Body-Inspired Generalisation Network (MBGN)

Core model implementation with:
- Random sparse projection (expansion layer)
- k-Winner-Take-All sparsification
- Accommodation mechanism (sensory adaptation)
- Dual readout pathways (specific + aggregate)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field


@dataclass
class MBGNConfig:
    """Configuration for MBGN model."""
    # Architecture
    n_input: int = 50  # Input dimension (like projection neurons)
    n_expansion: int = 2000  # Expansion layer size (like Kenyon cells)
    n_output: int = 1  # Output dimension (GO/NOGO decision)

    # Projection layer
    # ~7 connections per KC (as in bee), so for 50 inputs: 7/50 = 0.14
    connection_prob: float = 0.14  # Sparse connectivity probability
    projection_weight: float = 1.0  # Weight for non-zero connections (not used with new init)

    # Sparsity
    sparsity_fraction: float = 0.05  # Fraction of units active (k/N_exp)

    # Accommodation
    accommodation_alpha: float = 0.7  # How much activation increases accommodation
    accommodation_tau: float = 30.0  # Time constant for decay (seconds)

    # Learning rates
    lr_specific: float = 0.001  # Learning rate for specific pathway (slow for stimulus-specific)
    lr_aggregate: float = 0.1  # Learning rate for aggregate pathway (fast for rule learning)
    reward_baseline_decay: float = 0.8  # Decay for reward baseline

    # Weight constraints
    weight_max: float = 10.0  # Maximum weight magnitude

    # Decision
    decision_threshold: float = 0.0  # Threshold for GO decision

    # Reward baseline initialization
    # Starting at 0.5 (expected for random performance) reduces early trial variance
    reward_baseline_init: float = 0.5

    # Aggregate pathway
    # Baseline should be between "same" (~560) and "different" (~640) aggregate values
    aggregate_baseline: float = 600.0  # Expected aggregate activity (for deviation computation)

    # Relative comparison mode: compare to sample aggregate instead of fixed baseline
    use_relative_aggregate: bool = True  # If True, use sample aggregate as baseline

    # Random seed
    seed: Optional[int] = None


@dataclass
class ForwardResult:
    """Results from a forward pass through the network."""
    decision: bool  # GO (True) or NOGO (False)
    sparse_rep: np.ndarray  # Sparse representation (Kenyon cell activations)
    aggregate_activity: float  # Sum of sparse representation
    out_specific: np.ndarray  # Output from specific pathway
    out_aggregate: np.ndarray  # Output from aggregate pathway
    out_combined: np.ndarray  # Combined output
    pre_sparse: np.ndarray  # Activations before k-WTA
    accommodation_applied: np.ndarray  # Accommodation state that was applied


class MBGN:
    """
    Mushroom Body-Inspired Generalisation Network.

    This network implements the key architectural features of the insect
    mushroom body that enable learning of abstract relational concepts:

    1. Expansion to sparse representations via random projection
    2. Dual readout: specific (pattern-based) and aggregate (sum-based)
    3. Accommodation: recently active units have reduced responsiveness
    """

    def __init__(self, config: Optional[MBGNConfig] = None):
        """
        Initialize the MBGN.

        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or MBGNConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Compute k (number of active units in sparse layer)
        self.k = int(self.config.n_expansion * self.config.sparsity_fraction)

        # Initialize layers
        self._init_projection_layer()
        self._init_readout_layers()
        self._init_accommodation()

        # Learning state - start at expected value for random performance
        self.reward_baseline = self.config.reward_baseline_init

        # Reference aggregate for relative comparison (set by set_sample_reference)
        self._sample_aggregate = None

    def _init_projection_layer(self):
        """Initialize the random sparse projection matrix (Input → Expansion)."""
        # Create sparse random connectivity
        # Each expansion unit connects to ~connection_prob fraction of inputs
        # Following the spec: ~7 connections per KC in biology, we use ~7% connectivity
        n_connections_per_unit = max(1, int(self.config.n_input * self.config.connection_prob))

        self.W_proj = np.zeros(
            (self.config.n_expansion, self.config.n_input),
            dtype=np.float32
        )

        for i in range(self.config.n_expansion):
            # Each unit connects to a random subset of inputs
            connected_inputs = self.rng.choice(
                self.config.n_input,
                size=n_connections_per_unit,
                replace=False
            )
            # Use random positive weights for variety
            weights = self.rng.uniform(0.5, 1.5, size=n_connections_per_unit)
            self.W_proj[i, connected_inputs] = weights.astype(np.float32)

    def _init_readout_layers(self):
        """Initialize the readout weights for both pathways."""
        # Specific pathway: reads the full sparse pattern
        # Shape: (n_output, n_expansion)
        self.W_specific = self.rng.normal(
            0, 0.01,
            (self.config.n_output, self.config.n_expansion)
        ).astype(np.float32)

        # Aggregate pathway: reads the sum of activity
        # Shape: (n_output,)
        # Initialize near zero - will be biased by set_aggregate_bias() if needed
        self.W_aggregate = self.rng.normal(
            0, 0.01,
            self.config.n_output
        ).astype(np.float32)

    def set_aggregate_bias(self, task_type: str, strength: float = 0.3):
        """
        Bias W_aggregate initialization based on task type.

        This reduces variance by starting the aggregate pathway weight
        in the correct direction for the task.

        Args:
            task_type: 'DMTS' or 'DNMTS'
            strength: Initial weight magnitude (higher = stronger bias)
        """
        # For DMTS: same → low aggregate → negative deviation → need negative weight for GO
        # For DNMTS: different → high aggregate → positive deviation → need positive weight for GO
        if task_type == 'DMTS':
            # Low aggregate (same/match) should produce GO (positive output)
            # Deviation is negative, so weight should be negative
            self.W_aggregate = np.array([-strength], dtype=np.float32)
        elif task_type == 'DNMTS':
            # High aggregate (different/non-match) should produce GO (positive output)
            # Deviation is positive, so weight should be positive
            self.W_aggregate = np.array([strength], dtype=np.float32)
        # else: keep random initialization

    def _init_accommodation(self):
        """Initialize accommodation state (one value per expansion unit)."""
        self.accommodation_state = np.zeros(self.config.n_expansion, dtype=np.float32)

    def k_wta(self, activations: np.ndarray) -> np.ndarray:
        """
        k-Winner-Take-All: Keep only the top k activations.

        Args:
            activations: Pre-sparse activations (n_expansion,)

        Returns:
            Sparse activations with only top k non-zero
        """
        # Find the k-th largest value
        if self.k >= len(activations):
            return activations.copy()

        # Get indices of top k activations
        top_k_indices = np.argpartition(activations, -self.k)[-self.k:]

        # Create sparse output
        sparse = np.zeros_like(activations)
        sparse[top_k_indices] = activations[top_k_indices]

        # Ensure non-negative (biological plausibility)
        sparse = np.maximum(sparse, 0)

        return sparse

    def forward(
        self,
        x: np.ndarray,
        update_accommodation: bool = True
    ) -> ForwardResult:
        """
        Forward pass through the network.

        Args:
            x: Input stimulus vector (n_input,)
            update_accommodation: Whether to update accommodation state

        Returns:
            ForwardResult with decision and internal states
        """
        # 1. Random sparse projection to expansion layer
        z = self.W_proj @ x  # shape: (n_expansion,)

        # 2. Apply accommodation (reduce activity of recently-active units)
        accommodation_applied = self.accommodation_state.copy()
        z_accommodated = z * (1.0 - self.accommodation_state)

        # 3. k-Winner-Take-All: keep only top k units
        sparse_rep = self.k_wta(z_accommodated)

        # 4. Update accommodation state for units that fired
        if update_accommodation:
            # Use binary indicator: any unit that fired gets accommodation
            # This ensures consistent ~50% reduction regardless of activation magnitude
            active_mask = (sparse_rep > 0).astype(np.float32)
            self.accommodation_state = (
                self.accommodation_state +
                self.config.accommodation_alpha * active_mask
            )
            # Clip accommodation to [0, 1]
            self.accommodation_state = np.clip(self.accommodation_state, 0, 1)

        # 5. Specific pathway: pattern-based readout
        out_specific = self.W_specific @ sparse_rep  # shape: (n_output,)

        # 6. Aggregate pathway: deviation-based readout
        # Key insight: use deviation from baseline, not raw activity
        # Low aggregate (accommodated/same) → negative deviation
        # High aggregate (novel/different) → positive deviation
        aggregate_activity = sparse_rep.sum()

        # Choose baseline: use sample's aggregate if available (relative mode),
        # otherwise use fixed/calibrated baseline
        if self.config.use_relative_aggregate and self._sample_aggregate is not None:
            baseline = self._sample_aggregate
        else:
            baseline = self.config.aggregate_baseline

        aggregate_deviation = aggregate_activity - baseline
        out_aggregate = self.W_aggregate * aggregate_deviation  # shape: (n_output,)

        # 7. Combine pathways
        out_combined = out_specific + out_aggregate

        # 8. Decision (GO if output > threshold)
        decision = bool(out_combined[0] > self.config.decision_threshold)

        return ForwardResult(
            decision=decision,
            sparse_rep=sparse_rep,
            aggregate_activity=aggregate_activity,
            out_specific=out_specific,
            out_aggregate=out_aggregate,
            out_combined=out_combined,
            pre_sparse=z_accommodated,
            accommodation_applied=accommodation_applied
        )

    def decay_accommodation(self, dt: float):
        """
        Decay accommodation state over time.

        Call between trials or during delays.

        Args:
            dt: Time elapsed (in same units as tau, typically seconds)
        """
        decay_factor = np.exp(-dt / self.config.accommodation_tau)
        self.accommodation_state = self.accommodation_state * decay_factor

    def reset_accommodation(self):
        """Reset accommodation state to zero."""
        self.accommodation_state = np.zeros(self.config.n_expansion, dtype=np.float32)

    def set_sample_reference(self, aggregate: float):
        """
        Set the sample's aggregate activity as reference for relative comparison.

        Call this after the sample presentation, before choice presentations.

        Args:
            aggregate: The sample's aggregate activity
        """
        self._sample_aggregate = aggregate

    def clear_sample_reference(self):
        """Clear the sample reference (call at trial start)."""
        self._sample_aggregate = None

    def calibrate_baseline(self, stimuli: list, delay: float = 1.0):
        """
        Calibrate aggregate_baseline based on actual stimulus characteristics.

        This measures the midpoint between "same" (accommodated) and "different"
        (fresh) aggregate activity for the given stimuli, and sets the baseline
        accordingly. This improves generalization to different stimulus sets.

        Args:
            stimuli: List of Stimulus objects to use for calibration
            delay: Delay between sample and test presentation
        """
        same_aggregates = []
        diff_aggregates = []

        for i, stim1 in enumerate(stimuli):
            for j, stim2 in enumerate(stimuli):
                # Present sample
                self.reset_accommodation()
                self.forward(stim1.vector, update_accommodation=True)
                self.decay_accommodation(delay)

                # Present test (same or different)
                result = self.forward(stim2.vector, update_accommodation=False)

                if i == j:
                    same_aggregates.append(result.aggregate_activity)
                else:
                    diff_aggregates.append(result.aggregate_activity)

        # Set baseline to midpoint between same and different
        same_mean = np.mean(same_aggregates)
        diff_mean = np.mean(diff_aggregates)
        self.config.aggregate_baseline = (same_mean + diff_mean) / 2.0

        self.reset_accommodation()
        return {
            'same_mean': same_mean,
            'diff_mean': diff_mean,
            'baseline': self.config.aggregate_baseline
        }

    def update_weights(
        self,
        reward: float,
        result: ForwardResult,
        learn_specific: bool = True,
        learn_aggregate: bool = True
    ):
        """
        Update weights using reward-modulated Hebbian learning.

        Args:
            reward: Reward signal (1 for reward, 0 for none, -1 for punishment)
            result: ForwardResult from the forward pass
            learn_specific: Whether to update specific pathway
            learn_aggregate: Whether to update aggregate pathway
        """
        # Update reward baseline (running average)
        self.reward_baseline = (
            self.config.reward_baseline_decay * self.reward_baseline +
            (1 - self.config.reward_baseline_decay) * reward
        )
        reward_delta = reward - self.reward_baseline

        # Use decision signal (1 for GO, -1 for NOGO) as "post" activity
        # This ensures weight updates happen even when outputs are near zero
        decision_signal = 1.0 if result.decision else -1.0

        # Specific pathway update (Hebbian: post * pre * reward)
        if learn_specific:
            pre = result.sparse_rep
            # Scale by sparse representation for active units only
            delta_W_specific = (
                self.config.lr_specific *
                reward_delta *
                decision_signal *
                pre.reshape(1, -1)  # Outer product with scalar decision
            )
            self.W_specific = self.W_specific + delta_W_specific

        # Aggregate pathway update
        # Use deviation from baseline (same computation as in forward pass)
        if learn_aggregate:
            aggregate_deviation = result.aggregate_activity - self.config.aggregate_baseline
            # Normalize deviation for stable learning
            norm_deviation = aggregate_deviation / 100.0
            delta_W_aggregate = (
                self.config.lr_aggregate *
                reward_delta *
                decision_signal *
                norm_deviation
            )
            self.W_aggregate = self.W_aggregate + delta_W_aggregate

        # Clip weights to prevent explosion
        self.W_specific = np.clip(
            self.W_specific,
            -self.config.weight_max,
            self.config.weight_max
        )
        self.W_aggregate = np.clip(
            self.W_aggregate,
            -self.config.weight_max,
            self.config.weight_max
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current model state for analysis."""
        return {
            'W_specific': self.W_specific.copy(),
            'W_aggregate': self.W_aggregate.copy(),
            'accommodation_state': self.accommodation_state.copy(),
            'reward_baseline': self.reward_baseline
        }

    def set_state(self, state: Dict[str, Any]):
        """Set model state from saved state."""
        self.W_specific = state['W_specific'].copy()
        self.W_aggregate = state['W_aggregate'].copy()
        self.accommodation_state = state['accommodation_state'].copy()
        self.reward_baseline = state['reward_baseline']

    def clone(self) -> 'MBGN':
        """Create a copy of this model with the same state."""
        new_model = MBGN(self.config)
        new_model.W_proj = self.W_proj.copy()
        new_model.set_state(self.get_state())
        return new_model

    # === Numerosity comparison methods ===

    def compare_stimuli(
        self,
        stim_a: np.ndarray,
        stim_b: np.ndarray,
        use_accommodation: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compare two stimuli and return which has higher aggregate activity.

        This method is designed for numerosity comparison tasks where
        we want to determine which stimulus has more elements based on
        aggregate activity in the expansion layer.

        Args:
            stim_a: First stimulus vector (n_input,)
            stim_b: Second stimulus vector (n_input,)
            use_accommodation: Whether to apply accommodation (default False)
                             For numerosity, accommodation should typically be
                             disabled for clean comparison.

        Returns:
            Tuple of (choice, info) where:
                choice: 'A' if stim_a has higher aggregate, 'B' otherwise
                info: Dictionary with aggregate values and comparison details
        """
        # Save accommodation state if we're temporarily disabling it
        if not use_accommodation:
            old_accommodation = self.accommodation_state.copy()
            self.accommodation_state = np.zeros_like(self.accommodation_state)

        # Process stimulus A
        result_a = self.forward(stim_a, update_accommodation=use_accommodation)
        aggregate_a = result_a.aggregate_activity

        # Process stimulus B
        result_b = self.forward(stim_b, update_accommodation=use_accommodation)
        aggregate_b = result_b.aggregate_activity

        # Restore accommodation state if we disabled it
        if not use_accommodation:
            self.accommodation_state = old_accommodation

        # Decision based on aggregate comparison
        if aggregate_a > aggregate_b:
            choice = 'A'
        elif aggregate_b > aggregate_a:
            choice = 'B'
        else:
            # Tie (unlikely but possible)
            choice = 'A'  # Default to A on tie

        return choice, {
            'aggregate_a': aggregate_a,
            'aggregate_b': aggregate_b,
            'aggregate_diff': aggregate_a - aggregate_b,
            'sparse_rep_a': result_a.sparse_rep,
            'sparse_rep_b': result_b.sparse_rep,
        }

    def forward_numerosity(
        self,
        x: np.ndarray,
        use_accommodation: bool = False
    ) -> ForwardResult:
        """
        Forward pass optimized for numerosity comparison.

        This is a convenience method that temporarily disables accommodation
        for clean numerosity signal, then restores state.

        Args:
            x: Input stimulus vector (n_input,)
            use_accommodation: Whether to apply accommodation (default False)

        Returns:
            ForwardResult with decision and internal states
        """
        if use_accommodation:
            return self.forward(x, update_accommodation=True)

        # Disable accommodation temporarily
        old_accommodation = self.accommodation_state.copy()
        self.accommodation_state = np.zeros_like(self.accommodation_state)

        result = self.forward(x, update_accommodation=False)

        # Restore accommodation state
        self.accommodation_state = old_accommodation

        return result

    def verify_numerosity_signal(
        self,
        numerosities: list,
        stimulus_generator,
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Verify that aggregate activity correlates with numerosity.

        This is a diagnostic method to check whether the random projection
        preserves the numerosity signal before running experiments.

        Args:
            numerosities: List of numerosity values to test
            stimulus_generator: NumerosityStimulusGenerator instance
            n_samples: Number of samples per numerosity

        Returns:
            Dictionary with correlation statistics
        """
        aggregates = []
        nums = []

        for n in numerosities:
            for _ in range(n_samples):
                stim = stimulus_generator.make_stimulus(n)
                result = self.forward_numerosity(stim.vector, use_accommodation=False)
                aggregates.append(result.aggregate_activity)
                nums.append(n)

        aggregates = np.array(aggregates)
        nums = np.array(nums)

        # Compute correlation
        correlation = np.corrcoef(nums, aggregates)[0, 1]

        # Compute mean aggregate per numerosity
        mean_by_num = {n: np.mean(aggregates[nums == n]) for n in numerosities}
        std_by_num = {n: np.std(aggregates[nums == n]) for n in numerosities}

        # Check monotonicity
        means_ordered = [mean_by_num[n] for n in sorted(numerosities)]
        is_monotonic = all(
            means_ordered[i] <= means_ordered[i+1]
            for i in range(len(means_ordered)-1)
        )

        return {
            'correlation': correlation,
            'mean_by_numerosity': mean_by_num,
            'std_by_numerosity': std_by_num,
            'is_monotonic': is_monotonic,
            'n_samples': n_samples * len(numerosities)
        }


class AblatedMBGN(MBGN):
    """
    MBGN with ablation options for testing component contributions.
    """

    def __init__(
        self,
        config: Optional[MBGNConfig] = None,
        disable_accommodation: bool = False,
        disable_specific_pathway: bool = False,
        disable_aggregate_pathway: bool = False
    ):
        """
        Initialize ablated MBGN.

        Args:
            config: Model configuration
            disable_accommodation: If True, accommodation has no effect
            disable_specific_pathway: If True, specific pathway output is zero
            disable_aggregate_pathway: If True, aggregate pathway output is zero
        """
        super().__init__(config)
        self.disable_accommodation = disable_accommodation
        self.disable_specific_pathway = disable_specific_pathway
        self.disable_aggregate_pathway = disable_aggregate_pathway

    def forward(
        self,
        x: np.ndarray,
        update_accommodation: bool = True
    ) -> ForwardResult:
        """Forward pass with ablations applied."""
        # 1. Random sparse projection to expansion layer
        z = self.W_proj @ x

        # 2. Apply accommodation (or not, if disabled)
        if self.disable_accommodation:
            z_accommodated = z
            accommodation_applied = np.zeros_like(self.accommodation_state)
        else:
            accommodation_applied = self.accommodation_state.copy()
            z_accommodated = z * (1.0 - self.accommodation_state)

        # 3. k-Winner-Take-All
        sparse_rep = self.k_wta(z_accommodated)

        # 4. Update accommodation state (even if disabled, for consistency)
        if update_accommodation and not self.disable_accommodation:
            active_mask = (sparse_rep > 0).astype(np.float32)
            self.accommodation_state = (
                self.accommodation_state +
                self.config.accommodation_alpha * active_mask
            )
            self.accommodation_state = np.clip(self.accommodation_state, 0, 1)

        # 5. Specific pathway (or zero if disabled)
        if self.disable_specific_pathway:
            out_specific = np.zeros(self.config.n_output, dtype=np.float32)
        else:
            out_specific = self.W_specific @ sparse_rep

        # 6. Aggregate pathway (or zero if disabled)
        aggregate_activity = sparse_rep.sum()
        aggregate_deviation = aggregate_activity - self.config.aggregate_baseline
        if self.disable_aggregate_pathway:
            out_aggregate = np.zeros(self.config.n_output, dtype=np.float32)
        else:
            out_aggregate = self.W_aggregate * aggregate_deviation

        # 7. Combine pathways
        out_combined = out_specific + out_aggregate

        # 8. Decision
        decision = bool(out_combined[0] > self.config.decision_threshold)

        return ForwardResult(
            decision=decision,
            sparse_rep=sparse_rep,
            aggregate_activity=aggregate_activity,
            out_specific=out_specific,
            out_aggregate=out_aggregate,
            out_combined=out_combined,
            pre_sparse=z_accommodated,
            accommodation_applied=accommodation_applied
        )
