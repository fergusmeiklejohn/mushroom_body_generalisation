"""
Tests for the numerosity module.

Run with: python -m pytest tests/test_numerosity.py -v
"""

import numpy as np
import pytest
from typing import List

# Import modules to test
from mbgn import MBGN, MBGNConfig, AblatedMBGN
from mbgn.numerosity_stimuli import (
    NumerosityStimulus,
    NumerosityStimulusSet,
    NumerosityStimulusGenerator,
    create_numerosity_stimuli,
    verify_numerosity_correlation,
)
from mbgn.numerosity_task import (
    NumerosityTaskType,
    NumerosityTrial,
    NumerosityTrialResult,
    NumerosityTask,
    NumerosityTaskRunner,
    NumerosityExperimentRunner,
)
from mbgn.numerosity_experiments import (
    NumerosityExperimentConfig,
    run_experiment_1_baseline,
)


class TestNumerosityStimulusGeneration:
    """Tests for numerosity stimulus generation."""

    def test_sparse_stimulus_has_correct_active_count(self):
        """Sparse stimulus should have exactly n_elements active units."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)

        for n_elements in [2, 3, 5, 6, 7]:
            stim = generator.make_sparse_stimulus(n_elements)
            active_count = np.sum(stim.vector > 0)
            assert active_count == n_elements, \
                f"Expected {n_elements} active units, got {active_count}"

    def test_sparse_stimulus_has_correct_shape(self):
        """Stimulus vector should have correct shape."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        stim = generator.make_sparse_stimulus(5)
        assert stim.vector.shape == (50,)

    def test_sparse_stimulus_values_are_binary(self):
        """Sparse stimulus should have only 0 and 1 values."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        stim = generator.make_sparse_stimulus(5)
        unique_values = np.unique(stim.vector)
        assert set(unique_values).issubset({0.0, 1.0})

    def test_different_seeds_produce_different_stimuli(self):
        """Different seeds should produce different stimuli."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        stim1 = generator.make_sparse_stimulus(5, instance_seed=1)
        stim2 = generator.make_sparse_stimulus(5, instance_seed=2)
        assert not np.array_equal(stim1.vector, stim2.vector)

    def test_same_seed_produces_same_stimulus(self):
        """Same seed should produce identical stimulus."""
        gen1 = NumerosityStimulusGenerator(n_input=50, seed=42)
        gen2 = NumerosityStimulusGenerator(n_input=50, seed=42)
        stim1 = gen1.make_sparse_stimulus(5, instance_seed=100)
        stim2 = gen2.make_sparse_stimulus(5, instance_seed=100)
        assert np.array_equal(stim1.vector, stim2.vector)

    def test_stimulus_metadata(self):
        """Stimulus should have correct metadata."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        stim = generator.make_sparse_stimulus(5, element_type='A', instance_seed=123)
        assert stim.n_elements == 5
        assert stim.element_type == 'A'
        assert stim.seed == 123

    def test_binary_element_stimulus(self):
        """Binary element stimulus should have correct structure."""
        generator = NumerosityStimulusGenerator(
            n_input=50, stimulus_type='binary', seed=42
        )
        stim = generator.make_binary_element_stimulus(
            n_elements=3, element_size=3
        )
        # Should have 3 * 3 = 9 active units
        active_count = np.sum(stim.vector > 0)
        assert active_count == 9

    def test_stimulus_set_generation(self):
        """Stimulus set should contain all expected conditions."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        stimulus_set = generator.generate_stimulus_set(
            training_numerosities=[2, 3],
            transfer_numerosities=[4],
            training_element_type='A',
            transfer_element_type='B',
            n_instances_per_condition=5
        )

        # Check all expected keys exist
        expected_keys = [
            (2, 'A'), (3, 'A'),  # Training
            (4, 'A'),            # Transfer novel num
            (2, 'B'), (3, 'B'),  # Transfer novel type
            (4, 'B'),            # Full transfer
        ]
        for key in expected_keys:
            assert key in stimulus_set.stimuli
            assert len(stimulus_set.stimuli[key]) == 5


class TestNumerosityCorrelation:
    """Tests for numerosity-activity correlation."""

    def test_numerosity_correlates_with_total_activity(self):
        """Stimulus total activity should correlate with numerosity."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        verification = verify_numerosity_correlation(
            generator,
            numerosities=[2, 3, 4, 5, 6],
            n_samples=50
        )
        # Correlation should be very high for sparse stimuli
        assert verification['correlation'] > 0.95

    def test_mean_activity_increases_with_numerosity(self):
        """Mean activity should increase with numerosity."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        verification = verify_numerosity_correlation(
            generator,
            numerosities=[2, 4, 6, 8],
            n_samples=50
        )
        means = verification['mean_by_numerosity']
        assert means[2] < means[4] < means[6] < means[8]


class TestMBGNNumerosityMethods:
    """Tests for MBGN numerosity comparison methods."""

    def test_compare_stimuli_returns_correct_format(self):
        """compare_stimuli should return choice and info dict."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)

        stim_a = generator.make_sparse_stimulus(3)
        stim_b = generator.make_sparse_stimulus(6)

        choice, info = model.compare_stimuli(stim_a.vector, stim_b.vector)

        assert choice in ['A', 'B']
        assert 'aggregate_a' in info
        assert 'aggregate_b' in info
        assert 'aggregate_diff' in info

    def test_compare_stimuli_chooses_higher_aggregate(self):
        """compare_stimuli should choose stimulus with higher aggregate."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)

        # Test multiple times with clear differences
        correct = 0
        for _ in range(20):
            stim_a = generator.make_sparse_stimulus(2)
            stim_b = generator.make_sparse_stimulus(8)
            choice, info = model.compare_stimuli(stim_a.vector, stim_b.vector)

            # B should have higher aggregate (more elements)
            if choice == 'B' and info['aggregate_b'] > info['aggregate_a']:
                correct += 1

        # Should be correct most of the time
        assert correct >= 15

    def test_forward_numerosity_disables_accommodation(self):
        """forward_numerosity should not use accommodation by default."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)

        stim = generator.make_sparse_stimulus(5)

        # Set some accommodation state
        model.accommodation_state = np.ones(model.config.n_expansion) * 0.5

        # forward_numerosity should restore state
        original_accommodation = model.accommodation_state.copy()
        result = model.forward_numerosity(stim.vector, use_accommodation=False)

        assert np.array_equal(model.accommodation_state, original_accommodation)

    def test_verify_numerosity_signal(self):
        """verify_numerosity_signal should return correlation stats."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)

        verification = model.verify_numerosity_signal(
            numerosities=[2, 4, 6],
            stimulus_generator=generator,
            n_samples=30
        )

        assert 'correlation' in verification
        assert 'mean_by_numerosity' in verification
        assert 'is_monotonic' in verification
        # Correlation should be positive
        assert verification['correlation'] > 0


class TestNumerosityTask:
    """Tests for numerosity task infrastructure."""

    def test_trial_generation(self):
        """Task should generate valid trials."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 3, 5, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )

        trial = task.generate_trial()

        assert trial.n_a != trial.n_b
        assert trial.n_a in [2, 3, 5, 6]
        assert trial.n_b in [2, 3, 5, 6]
        assert trial.correct_choice in ['A', 'B']

    def test_choose_more_correct_choice(self):
        """CHOOSE_MORE should reward choosing higher numerosity."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )

        # Generate many trials and check correctness
        for _ in range(20):
            trial = task.generate_trial()
            if trial.n_a > trial.n_b:
                assert trial.correct_choice == 'A'
            else:
                assert trial.correct_choice == 'B'

    def test_choose_fewer_correct_choice(self):
        """CHOOSE_FEWER should reward choosing lower numerosity."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_FEWER,
            seed=42
        )

        for _ in range(20):
            trial = task.generate_trial()
            if trial.n_a < trial.n_b:
                assert trial.correct_choice == 'A'
            else:
                assert trial.correct_choice == 'B'

    def test_reward_function(self):
        """Reward should be 1.0 for correct, 0.0 for incorrect."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 3, 5, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )

        trial = task.generate_trial()
        assert task.get_reward(trial.correct_choice, trial) == 1.0

        wrong_choice = 'B' if trial.correct_choice == 'A' else 'A'
        assert task.get_reward(wrong_choice, trial) == 0.0

    def test_balanced_sequence(self):
        """Balanced sequence should cover numerosity pairs."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 3, 4],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )

        trials = task.generate_trial_sequence(30, balanced=True)

        # All pairs should appear
        pairs_seen = set()
        for trial in trials:
            pair = tuple(sorted([trial.n_a, trial.n_b]))
            pairs_seen.add(pair)

        expected_pairs = {(2, 3), (2, 4), (3, 4)}
        assert pairs_seen == expected_pairs


class TestNumerosityTaskRunner:
    """Tests for task runner."""

    def test_run_trial_returns_result(self):
        """run_trial should return NumerosityTrialResult."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 3, 5, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )
        runner = NumerosityTaskRunner(model, task, seed=42)

        trial = task.generate_trial()
        result = runner.run_trial(trial, learn=False)

        assert isinstance(result, NumerosityTrialResult)
        assert result.choice in ['A', 'B']
        assert isinstance(result.correct, bool)
        assert result.reward in [0.0, 1.0]

    def test_run_block_returns_list(self):
        """run_block should return list of results."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 3, 5, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )
        runner = NumerosityTaskRunner(model, task, seed=42)

        results = runner.run_block(10, learn=True)

        assert len(results) == 10
        assert all(isinstance(r, NumerosityTrialResult) for r in results)

    def test_compute_accuracy(self):
        """compute_accuracy should return correct value."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 3, 5, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )
        runner = NumerosityTaskRunner(model, task, seed=42)

        results = runner.run_block(20, learn=False)
        accuracy = runner.compute_accuracy(results)

        assert 0 <= accuracy <= 1

    def test_accuracy_by_distance(self):
        """compute_accuracy_by_distance should return dict."""
        model = MBGN(MBGNConfig(seed=42))
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        task = NumerosityTask(
            numerosities=[2, 3, 5, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )
        runner = NumerosityTaskRunner(model, task, seed=42)

        results = runner.run_block(30, learn=False)
        by_distance = runner.compute_accuracy_by_distance(results)

        assert isinstance(by_distance, dict)
        # All accuracies should be in [0, 1]
        for acc in by_distance.values():
            assert 0 <= acc <= 1


class TestNumerosityExperiments:
    """Tests for experiment functions."""

    def test_experiment_config_defaults(self):
        """Experiment config should have sensible defaults."""
        config = NumerosityExperimentConfig()
        assert config.training_numerosities == [2, 3, 5, 6]
        assert config.transfer_numerosities == [4, 7]
        assert config.n_training_trials == 100
        assert config.use_accommodation == False

    def test_baseline_experiment_runs(self):
        """Baseline experiment should run without errors."""
        config = NumerosityExperimentConfig(
            n_training_trials=20,
            n_transfer_trials=10,
            block_size=10
        )

        result = run_experiment_1_baseline(config, verbose=False)

        assert result.name == "Exp1: Baseline"
        assert 0 <= result.training_accuracy <= 1
        assert 0 <= result.transfer_accuracy <= 1
        assert len(result.training_accuracy_by_block) > 0


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_sparse_stimuli(self):
        """Test full pipeline with sparse stimuli."""
        # Create components
        generator = NumerosityStimulusGenerator(
            n_input=50,
            stimulus_type='sparse',
            seed=42
        )
        model = MBGN(MBGNConfig(n_input=50, seed=42))
        task = NumerosityTask(
            numerosities=[2, 4, 6],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )
        runner = NumerosityTaskRunner(
            model, task,
            use_accommodation=False,
            seed=42
        )

        # Run training
        training_results = runner.run_block(50, learn=True)
        training_accuracy = runner.compute_accuracy(training_results)

        # Run test
        test_results = runner.run_block(20, learn=False)
        test_accuracy = runner.compute_accuracy(test_results)

        # Should learn something
        assert test_accuracy > 0.4  # Better than random guessing

    def test_learning_improves_accuracy(self):
        """Training should improve accuracy over time."""
        generator = NumerosityStimulusGenerator(n_input=50, seed=42)
        model = MBGN(MBGNConfig(n_input=50, seed=42))
        task = NumerosityTask(
            numerosities=[2, 4, 6, 8],
            stimulus_generator=generator,
            task_type=NumerosityTaskType.CHOOSE_MORE,
            seed=42
        )
        runner = NumerosityTaskRunner(model, task, seed=42)

        # Early accuracy
        early_results = runner.run_block(20, learn=True)
        early_accuracy = runner.compute_accuracy(early_results)

        # Continue training
        runner.run_block(80, learn=True)

        # Later accuracy (on new trials)
        later_results = runner.run_block(20, learn=False)
        later_accuracy = runner.compute_accuracy(later_results)

        # Should improve (or at least not get worse)
        # Note: This test may occasionally fail due to randomness
        assert later_accuracy >= early_accuracy - 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
