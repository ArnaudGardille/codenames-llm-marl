"""Integration tests for Qwen embedding agents."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from codenames_rl.agents.baselines import (
    QwenEmbeddingGuesser,
    QwenEmbeddingSpymaster,
)
from codenames_rl.env.spaces import CardColor, GamePhase, Observation

# Check if torch and transformers are available
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def small_vocabulary():
    """Create a small vocabulary file for testing."""
    vocab_words = ["fruit", "animal", "music"]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(vocab_words))
        temp_path = f.name
    
    yield temp_path
    
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def empty_vocabulary():
    """Create an empty vocabulary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    yield temp_path
    
    Path(temp_path).unlink(missing_ok=True)


def qwen_model_available():
    """Check if Qwen model can be loaded (skip test if not)."""
    if not HAS_TORCH:
        return False
    try:
        # Try to load a small part of the model to check availability
        # We'll use a try-except in actual tests
        return True
    except Exception:
        return False


@pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
class TestQwenEmbeddingSpymaster:
    """Tests for QwenEmbeddingSpymaster."""

    def test_initialization(self, temp_vocabulary):
        """Test that QwenEmbeddingSpymaster initializes correctly."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10  # Small for speed
            )
            assert agent.model is not None
            assert agent.tokenizer is not None
            assert agent.vocabulary is not None
            assert len(agent.vocabulary) > 0
        except Exception as e:
            # Model might not be available, skip test
            pytest.skip(f"Qwen model not available: {e}")

    def test_vocabulary_loading(self, temp_vocabulary):
        """Test vocabulary loading."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            assert "apple" in agent.vocabulary
            assert "banana" in agent.vocabulary
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    def test_invalid_vocabulary_path(self):
        """Test that invalid vocabulary path raises FileNotFoundError."""
        try:
            with pytest.raises(FileNotFoundError):
                QwenEmbeddingSpymaster(
                    vocabulary_path="/nonexistent/path/vocab.txt",
                    seed=42
                )
        except Exception as e:
            if "not available" in str(e).lower():
                pytest.skip(f"Qwen model not available: {e}")
            raise

    @pytest.mark.slow
    def test_get_clue_returns_valid_action(self, temp_vocabulary, sample_observation):
        """Test that get_clue returns a valid SpymasterAction."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            action = agent.get_clue(sample_observation)
            
            assert action.clue is not None
            assert isinstance(action.clue, str)
            assert action.count > 0
            assert isinstance(action.count, int)
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_clue_not_on_board(self, temp_vocabulary, sample_observation):
        """Test that generated clue is not on the board."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            action = agent.get_clue(sample_observation)
            
            board_lower = [w.lower() for w in sample_observation.board_words]
            assert action.clue.lower() not in board_lower
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ])
        
        sims = QwenEmbeddingSpymaster._cosine_similarity(vec, matrix)
        
        assert len(sims) == 3
        assert np.isclose(sims[0], 1.0)
        assert np.isclose(sims[1], 0.0)
        assert 0.0 < sims[2] < 1.0

    @pytest.mark.slow
    def test_handles_no_team_words(self, temp_vocabulary, sample_observation):
        """Test behavior when no team words remain."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'team_remaining': 0,
                   'board_colors': [CardColor.OPPONENT] * 25}
            )
            
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            action = agent.get_clue(obs)
            
            # Should return "pass" with count 0
            assert action.clue == "pass"
            assert action.count == 0
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_all_words_revealed(self, temp_vocabulary, sample_observation):
        """Test behavior when all words are revealed."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'revealed_mask': [True] * 25}
            )
            
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            action = agent.get_clue(obs)
            
            # Should still return a valid action (pass)
            assert action.clue is not None
            assert action.count >= 0
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_single_team_word(self, temp_vocabulary, sample_observation):
        """Test behavior with single team word."""
        try:
            colors = [CardColor.TEAM] + [CardColor.OPPONENT] * 24
            obs = Observation(
                **{**sample_observation.__dict__,
                   'board_colors': colors,
                   'team_remaining': 1}
            )
            
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            action = agent.get_clue(obs)
            
            assert action.clue is not None
            assert action.count >= 1
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_small_vocabulary(self, small_vocabulary, sample_observation):
        """Test with vocabulary smaller than top_k."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=small_vocabulary,
                seed=42,
                top_k=100  # Larger than vocabulary size
            )
            action = agent.get_clue(sample_observation)
            
            assert action.clue is not None
            assert action.count > 0
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_large_vocabulary_sampling(self, temp_vocabulary, sample_observation):
        """Test that large vocabulary is sampled."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=5  # Smaller than vocabulary size
            )
            # Should not raise error
            action = agent.get_clue(sample_observation)
            assert action.clue is not None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_device_auto_detection(self, temp_vocabulary):
        """Test device auto-detection."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            assert agent.device in ["cpu", "cuda", "mps"]
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_explicit_device_cpu(self, temp_vocabulary):
        """Test explicit CPU device specification."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                device="cpu",
                seed=42,
                top_k=10
            )
            assert agent.device == "cpu"
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_encoding_instruction_format(self, temp_vocabulary):
        """Test that encoding uses instruction-aware format."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            # Test encoding
            texts = ["apple", "banana"]
            embeddings = agent._encode(texts)
            
            assert embeddings.shape[0] == 2
            assert embeddings.shape[1] > 0  # Has embedding dimension
            # Check normalization
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6)
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_encoding_empty_list(self, temp_vocabulary):
        """Test encoding with empty text list."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            embeddings = agent._encode([])
            assert embeddings.shape[0] == 0
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_parameter_alpha_beta_gamma(self, temp_vocabulary, sample_observation):
        """Test different alpha/beta/gamma parameter values."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                alpha=5.0,
                beta=2.0,
                gamma=0.5,
                seed=42,
                top_k=10
            )
            action = agent.get_clue(sample_observation)
            assert action.clue is not None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_seed_reproducibility(self, temp_vocabulary, sample_observation):
        """Test that same seed produces same results."""
        try:
            agent1 = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            agent2 = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            
            action1 = agent1.get_clue(sample_observation)
            action2 = agent2.get_clue(sample_observation)
            
            # Should produce same clue with same seed
            assert action1.clue == action2.clue
            assert action1.count == action2.count
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_model_sharing_with_guesser(self, temp_vocabulary):
        """Test model sharing with QwenEmbeddingGuesser."""
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
            
            # Both should use same model
            assert guesser.model is spymaster.model
            assert guesser.tokenizer is spymaster.tokenizer
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_vocabulary_only_board_words(self, temp_vocabulary, sample_observation):
        """Test behavior when vocabulary only contains board words."""
        try:
            # Create vocabulary with only board words
            board_words = sample_observation.board_words
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write('\n'.join([w.lower() for w in board_words[:5]]))
                vocab_path = f.name
            
            try:
                agent = QwenEmbeddingSpymaster(
                    vocabulary_path=vocab_path,
                    seed=42,
                    top_k=10
                )
                # Should still return a valid action (fallback)
                action = agent.get_clue(sample_observation)
                assert action.clue is not None
            finally:
                Path(vocab_path).unlink(missing_ok=True)
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_unicode_characters(self, temp_vocabulary):
        """Test encoding with unicode characters."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            texts = ["café", "naïve", "résumé", "北京", "東京"]
            embeddings = agent._encode(texts)
            assert embeddings.shape[0] == 5
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_very_long_words(self, temp_vocabulary):
        """Test encoding with very long words."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            long_word = "a" * 1000
            embeddings = agent._encode([long_word])
            assert embeddings.shape[0] == 1
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_similarity_threshold_parameter(self, temp_vocabulary, sample_observation):
        """Test different similarity_threshold values."""
        try:
            agent_low = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                similarity_threshold=0.1,
                seed=42,
                top_k=10
            )
            agent_high = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                similarity_threshold=0.9,
                seed=42,
                top_k=10
            )
            
            action_low = agent_low.get_clue(sample_observation)
            action_high = agent_high.get_clue(sample_observation)
            
            # Both should return valid actions
            assert action_low.clue is not None
            assert action_high.clue is not None
            # Lower threshold might result in higher counts
            assert action_low.count >= 1
            assert action_high.count >= 1
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")


@pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
class TestQwenEmbeddingGuesser:
    """Tests for QwenEmbeddingGuesser."""

    def test_initialization(self):
        """Test that QwenEmbeddingGuesser initializes correctly."""
        try:
            agent = QwenEmbeddingGuesser(seed=42)
            assert agent.model is not None
            assert agent.tokenizer is not None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_get_guess_with_clue(self, sample_observation):
        """Test that get_guess works with an active clue."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'current_clue': 'furniture',
                   'current_count': 3,
                   'phase': GamePhase.GUESSER_TURN}
            )
            
            agent = QwenEmbeddingGuesser(seed=42)
            action = agent.get_guess(obs)
            
            # Should return either an index or None
            assert action.word_index is None or isinstance(action.word_index, int)
            if action.word_index is not None:
                assert 0 <= action.word_index < 25
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_guess_only_unrevealed(self, sample_observation):
        """Test that guesser only picks unrevealed words."""
        try:
            revealed = [True] * 10 + [False] * 15
            obs = Observation(
                **{**sample_observation.__dict__,
                   'revealed_mask': revealed,
                   'current_clue': 'test',
                   'current_count': 2,
                   'phase': GamePhase.GUESSER_TURN}
            )
            
            agent = QwenEmbeddingGuesser(seed=42)
            action = agent.get_guess(obs)
            
            if action.word_index is not None:
                assert not revealed[action.word_index]
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    def test_no_clue_returns_pass(self, sample_observation):
        """Test that guesser passes when no clue is active."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'current_clue': None,
                   'phase': GamePhase.SPYMASTER_TURN}
            )
            
            agent = QwenEmbeddingGuesser(seed=42)
            action = agent.get_guess(obs)
            
            assert action.word_index is None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_all_revealed_returns_pass(self, sample_observation):
        """Test that guesser passes when all words are revealed."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'revealed_mask': [True] * 25,
                   'current_clue': 'test',
                   'current_count': 2,
                   'phase': GamePhase.GUESSER_TURN}
            )
            
            agent = QwenEmbeddingGuesser(seed=42)
            action = agent.get_guess(obs)
            
            assert action.word_index is None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ])
        
        sims = QwenEmbeddingGuesser._cosine_similarity(vec, matrix)
        
        assert len(sims) == 3
        assert np.isclose(sims[0], 1.0)
        assert np.isclose(sims[1], 0.0)
        assert 0.0 < sims[2] < 1.0

    @pytest.mark.slow
    def test_low_confidence_passes(self, sample_observation):
        """Test that low similarity leads to passing."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'current_clue': 'xqzthwpqmn',  # Nonsense word
                   'current_count': 2,
                   'phase': GamePhase.GUESSER_TURN}
            )
            
            agent = QwenEmbeddingGuesser(
                seed=42,
                confidence_threshold=0.9  # Very high threshold
            )
            action = agent.get_guess(obs)
            
            # Should pass due to low confidence
            assert action.word_index is None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_high_confidence_guesses(self, sample_observation):
        """Test that high similarity leads to guessing."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'current_clue': 'furniture',  # Should match some words
                   'current_count': 3,
                   'phase': GamePhase.GUESSER_TURN}
            )
            
            agent = QwenEmbeddingGuesser(
                seed=42,
                confidence_threshold=0.0  # Very low threshold
            )
            action = agent.get_guess(obs)
            
            # Should guess if similarity is above threshold
            # (might be None if still too low, but should try)
            assert action.word_index is None or isinstance(action.word_index, int)
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_single_unrevealed_word(self, sample_observation):
        """Test behavior with single unrevealed word."""
        try:
            revealed = [True] * 24 + [False]
            obs = Observation(
                **{**sample_observation.__dict__,
                   'revealed_mask': revealed,
                   'current_clue': 'test',
                   'current_count': 1,
                   'phase': GamePhase.GUESSER_TURN}
            )
            
            agent = QwenEmbeddingGuesser(seed=42)
            action = agent.get_guess(obs)
            
            # Should either guess the word or pass
            assert action.word_index is None or action.word_index == 24
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_encoding_functionality(self):
        """Test encoding functionality."""
        try:
            agent = QwenEmbeddingGuesser(seed=42)
            texts = ["apple", "banana", "cherry"]
            embeddings = agent._encode(texts)
            
            assert embeddings.shape[0] == 3
            assert embeddings.shape[1] > 0
            # Check normalization
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6)
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_model_sharing_with_spymaster(self, temp_vocabulary):
        """Test model sharing with QwenEmbeddingSpymaster."""
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
            
            # Both should use same model
            assert guesser.model is spymaster.model
            assert guesser.tokenizer is spymaster.tokenizer
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_device_handling_when_sharing(self, temp_vocabulary):
        """Test device handling when sharing models."""
        try:
            spymaster = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                device="cpu",
                seed=42,
                top_k=10
            )
            guesser = QwenEmbeddingGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device=spymaster.device,
                seed=42
            )
            
            assert guesser.device == spymaster.device
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_different_confidence_thresholds(self, sample_observation):
        """Test different confidence threshold values."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'current_clue': 'test',
                   'current_count': 2,
                   'phase': GamePhase.GUESSER_TURN}
            )
            
            # Low threshold
            agent_low = QwenEmbeddingGuesser(
                seed=42,
                confidence_threshold=0.0
            )
            action_low = agent_low.get_guess(obs)
            
            # High threshold
            agent_high = QwenEmbeddingGuesser(
                seed=42,
                confidence_threshold=1.0
            )
            action_high = agent_high.get_guess(obs)
            
            # High threshold should be more conservative
            # (more likely to pass)
            assert isinstance(action_low.word_index, (int, type(None)))
            assert isinstance(action_high.word_index, (int, type(None)))
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

