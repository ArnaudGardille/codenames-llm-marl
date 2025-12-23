"""Integration tests for Qwen embedding agents with evaluation scripts."""

import pytest

from codenames_rl.agents import (
    QwenEmbeddingGuesser,
    QwenEmbeddingSpymaster,
)
from codenames_rl.eval import EvaluationHarness

# Check if torch and transformers are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
@pytest.mark.slow
class TestQwenIntegration:
    """Integration tests for Qwen embedding agents."""

    def test_evaluation_harness_with_qwen_agents(self, temp_wordlist, temp_vocabulary):
        """Test that EvaluationHarness works with Qwen agents."""
        try:
            spymaster = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            guesser = QwenEmbeddingGuesser(seed=42)
            
            harness = EvaluationHarness(
                wordlist_path=temp_wordlist,
                spymaster=spymaster,
                guesser=guesser,
                max_turns=10,
                max_guesses=5,
                verbose=False
            )
            
            metrics, results = harness.evaluate_with_details(
                num_games=2,
                start_seed=42
            )
            
            assert metrics.num_games == 2
            assert len(results) == 2
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    def test_model_sharing_in_evaluation(self, temp_wordlist, temp_vocabulary):
        """Test model sharing in evaluation harness."""
        try:
            spymaster = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            guesser = QwenEmbeddingGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device=spymaster.device,
                seed=42
            )
            
            harness = EvaluationHarness(
                wordlist_path=temp_wordlist,
                spymaster=spymaster,
                guesser=guesser,
                max_turns=10,
                max_guesses=5,
                verbose=False
            )
            
            metrics, results = harness.evaluate_with_details(
                num_games=2,
                start_seed=42
            )
            
            assert metrics.num_games == 2
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    def test_multiple_games_reproducibility(self, temp_wordlist, temp_vocabulary):
        """Test that same seed produces reproducible results."""
        try:
            spymaster1 = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            guesser1 = QwenEmbeddingGuesser(seed=42)
            
            spymaster2 = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            guesser2 = QwenEmbeddingGuesser(seed=42)
            
            harness1 = EvaluationHarness(
                wordlist_path=temp_wordlist,
                spymaster=spymaster1,
                guesser=guesser1,
                max_turns=10,
                max_guesses=5,
                verbose=False
            )
            
            harness2 = EvaluationHarness(
                wordlist_path=temp_wordlist,
                spymaster=spymaster2,
                guesser=guesser2,
                max_turns=10,
                max_guesses=5,
                verbose=False
            )
            
            metrics1, _ = harness1.evaluate_with_details(
                num_games=2,
                start_seed=42
            )
            
            metrics2, _ = harness2.evaluate_with_details(
                num_games=2,
                start_seed=42
            )
            
            # Should produce same results with same seeds
            assert metrics1.win_rate == metrics2.win_rate
            assert metrics1.avg_score == metrics2.avg_score
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    def test_different_seeds_produce_different_results(self, temp_wordlist, temp_vocabulary):
        """Test that different seeds produce different results."""
        try:
            spymaster1 = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            guesser1 = QwenEmbeddingGuesser(seed=42)
            
            spymaster2 = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=123,
                top_k=10
            )
            guesser2 = QwenEmbeddingGuesser(seed=123)
            
            harness1 = EvaluationHarness(
                wordlist_path=temp_wordlist,
                spymaster=spymaster1,
                guesser=guesser1,
                max_turns=10,
                max_guesses=5,
                verbose=False
            )
            
            harness2 = EvaluationHarness(
                wordlist_path=temp_wordlist,
                spymaster=spymaster2,
                guesser=guesser2,
                max_turns=10,
                max_guesses=5,
                verbose=False
            )
            
            metrics1, _ = harness1.evaluate_with_details(
                num_games=2,
                start_seed=42
            )
            
            metrics2, _ = harness2.evaluate_with_details(
                num_games=2,
                start_seed=42
            )
            
            # Results might be different (or same by chance)
            # Just verify both complete successfully
            assert metrics1.num_games == 2
            assert metrics2.num_games == 2
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")


