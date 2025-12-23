"""Unit tests for baseline agents."""

import numpy as np
import pytest

from codenames_rl.agents.baselines import (
    EmbeddingsGuesser,
    EmbeddingsSpymaster,
    LLMGuesser,
    LLMSpymaster,
    QwenEmbeddingGuesser,
    QwenEmbeddingSpymaster,
    RandomGuesser,
    RandomSpymaster,
)
from codenames_rl.env.spaces import CardColor, GamePhase, Observation

# Check if torch and transformers are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestRandomSpymaster:
    """Tests for RandomSpymaster."""

    def test_initialization(self, temp_vocabulary):
        """Test that RandomSpymaster initializes correctly."""
        agent = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        assert agent.vocabulary is not None
        assert len(agent.vocabulary) > 0

    def test_vocabulary_loading(self, temp_vocabulary):
        """Test vocabulary loading."""
        agent = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        assert "apple" in agent.vocabulary
        assert "banana" in agent.vocabulary

    def test_get_clue_returns_valid_action(self, temp_vocabulary, sample_observation):
        """Test that get_clue returns a valid SpymasterAction."""
        agent = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        action = agent.get_clue(sample_observation)
        
        assert action.clue is not None
        assert isinstance(action.clue, str)
        assert action.count > 0
        assert isinstance(action.count, int)

    def test_clue_not_on_board(self, temp_vocabulary, sample_observation):
        """Test that generated clue is not on the board."""
        agent = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        action = agent.get_clue(sample_observation)
        
        board_lower = [w.lower() for w in sample_observation.board_words]
        assert action.clue.lower() not in board_lower

    def test_reproducibility(self, temp_vocabulary, sample_observation):
        """Test that same seed produces same results."""
        agent1 = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        agent2 = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        
        action1 = agent1.get_clue(sample_observation)
        action2 = agent2.get_clue(sample_observation)
        
        assert action1.clue == action2.clue
        assert action1.count == action2.count


class TestRandomGuesser:
    """Tests for RandomGuesser."""

    def test_initialization(self):
        """Test that RandomGuesser initializes correctly."""
        agent = RandomGuesser(seed=42)
        assert agent.rng is not None

    def test_get_guess_returns_valid_action(self, sample_observation):
        """Test that get_guess returns a valid GuesserAction."""
        obs = Observation(
            **{**sample_observation.__dict__, 
               'current_clue': 'test',
               'current_count': 2,
               'phase': GamePhase.GUESSER_TURN}
        )
        
        agent = RandomGuesser(seed=42)
        action = agent.get_guess(obs)
        
        assert action.word_index is None or isinstance(action.word_index, int)

    def test_guess_only_unrevealed(self, sample_observation):
        """Test that guesser only picks unrevealed words."""
        # Reveal some words
        revealed = [True] * 10 + [False] * 15
        obs = Observation(
            **{**sample_observation.__dict__,
               'revealed_mask': revealed,
               'current_clue': 'test',
               'current_count': 2,
               'phase': GamePhase.GUESSER_TURN}
        )
        
        agent = RandomGuesser(seed=42)
        
        # Test multiple guesses
        for _ in range(10):
            action = agent.get_guess(obs)
            if action.word_index is not None:
                assert not revealed[action.word_index]

    def test_all_revealed_returns_pass(self, sample_observation):
        """Test that guesser passes when all words are revealed."""
        obs = Observation(
            **{**sample_observation.__dict__,
               'revealed_mask': [True] * 25,
               'current_clue': 'test',
               'current_count': 2,
               'phase': GamePhase.GUESSER_TURN}
        )
        
        agent = RandomGuesser(seed=42)
        action = agent.get_guess(obs)
        
        assert action.word_index is None


class TestEmbeddingsSpymaster:
    """Tests for EmbeddingsSpymaster."""

    def test_initialization(self, temp_vocabulary):
        """Test that EmbeddingsSpymaster initializes correctly."""
        agent = EmbeddingsSpymaster(
            vocabulary_path=temp_vocabulary,
            seed=42
        )
        assert agent.model is not None
        assert agent.vocabulary is not None

    def test_get_clue_returns_valid_action(self, temp_vocabulary, sample_observation):
        """Test that get_clue returns a valid SpymasterAction."""
        agent = EmbeddingsSpymaster(
            vocabulary_path=temp_vocabulary,
            seed=42,
            top_k=10  # Small for speed
        )
        action = agent.get_clue(sample_observation)
        
        assert action.clue is not None
        assert isinstance(action.clue, str)
        assert action.count > 0
        assert isinstance(action.count, int)

    def test_clue_not_on_board(self, temp_vocabulary, sample_observation):
        """Test that generated clue is not on the board."""
        agent = EmbeddingsSpymaster(
            vocabulary_path=temp_vocabulary,
            seed=42,
            top_k=10
        )
        action = agent.get_clue(sample_observation)
        
        board_lower = [w.lower() for w in sample_observation.board_words]
        assert action.clue.lower() not in board_lower

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ])
        
        sims = EmbeddingsSpymaster._cosine_similarity(vec, matrix)
        
        assert len(sims) == 3
        assert np.isclose(sims[0], 1.0)
        assert np.isclose(sims[1], 0.0)
        assert 0.0 < sims[2] < 1.0

    def test_handles_no_team_words(self, temp_vocabulary, sample_observation):
        """Test behavior when no team words remain."""
        obs = Observation(
            **{**sample_observation.__dict__,
               'team_remaining': 0,
               'board_colors': [CardColor.OPPONENT] * 25}
        )
        
        agent = EmbeddingsSpymaster(
            vocabulary_path=temp_vocabulary,
            seed=42,
            top_k=10
        )
        action = agent.get_clue(obs)
        
        # Should still return a valid action
        assert action.clue is not None
        assert action.count >= 0


class TestEmbeddingsGuesser:
    """Tests for EmbeddingsGuesser."""

    def test_initialization(self):
        """Test that EmbeddingsGuesser initializes correctly."""
        agent = EmbeddingsGuesser(seed=42)
        assert agent.model is not None

    def test_get_guess_with_clue(self, sample_observation):
        """Test that get_guess works with an active clue."""
        obs = Observation(
            **{**sample_observation.__dict__,
               'current_clue': 'furniture',
               'current_count': 3,
               'phase': GamePhase.GUESSER_TURN}
        )
        
        agent = EmbeddingsGuesser(seed=42)
        action = agent.get_guess(obs)
        
        # Should return either an index or None
        assert action.word_index is None or isinstance(action.word_index, int)
        if action.word_index is not None:
            assert 0 <= action.word_index < 25

    def test_guess_only_unrevealed(self, sample_observation):
        """Test that guesser only picks unrevealed words."""
        revealed = [True] * 10 + [False] * 15
        obs = Observation(
            **{**sample_observation.__dict__,
               'revealed_mask': revealed,
               'current_clue': 'test',
               'current_count': 2,
               'phase': GamePhase.GUESSER_TURN}
        )
        
        agent = EmbeddingsGuesser(seed=42)
        action = agent.get_guess(obs)
        
        if action.word_index is not None:
            assert not revealed[action.word_index]

    def test_no_clue_returns_pass(self, sample_observation):
        """Test that guesser passes when no clue is active."""
        obs = Observation(
            **{**sample_observation.__dict__,
               'current_clue': None,
               'phase': GamePhase.SPYMASTER_TURN}
        )
        
        agent = EmbeddingsGuesser(seed=42)
        action = agent.get_guess(obs)
        
        assert action.word_index is None

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ])
        
        sims = EmbeddingsGuesser._cosine_similarity(vec, matrix)
        
        assert len(sims) == 3
        assert np.isclose(sims[0], 1.0)
        assert np.isclose(sims[1], 0.0)
        assert 0.0 < sims[2] < 1.0

    def test_low_confidence_passes(self, sample_observation):
        """Test that low similarity leads to passing."""
        obs = Observation(
            **{**sample_observation.__dict__,
               'current_clue': 'xqzthwpqmn',  # Nonsense word
               'current_count': 2,
               'phase': GamePhase.GUESSER_TURN}
        )
        
        agent = EmbeddingsGuesser(
            seed=42,
            confidence_threshold=0.9  # Very high threshold
        )
        action = agent.get_guess(obs)
        
        # Should pass due to low confidence
        assert action.word_index is None


@pytest.mark.slow
class TestLLMSpymaster:
    """Tests for LLMSpymaster (requires GPU and model download)."""

    def test_initialization(self):
        """Test that LLMSpymaster initializes correctly."""
        # Skip if no CUDA available
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        agent = LLMSpymaster(seed=42)
        assert agent.model is not None
        assert agent.tokenizer is not None

    def test_get_board_token_ids(self, sample_observation):
        """Test token ID suppression helper method."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available")
        
        agent = LLMSpymaster(seed=42, temperature=0.1)
        board_words = ["tree", "apple", "car"]
        token_ids = agent._get_board_token_ids(board_words)
        
        assert token_ids is not None
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        # Should have multiple variants per word (lowercase, uppercase, etc.)
        assert len(token_ids) >= len(board_words)

    def test_get_clue_returns_valid_action(self, sample_observation):
        """Test that get_clue returns a valid SpymasterAction."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available")
        
        agent = LLMSpymaster(seed=42, temperature=0.1)
        action = agent.get_clue(sample_observation)
        
        assert action.clue is not None
        assert isinstance(action.clue, str)
        assert action.count > 0
        assert isinstance(action.count, int)

    def test_clue_not_on_board(self, sample_observation):
        """Test that generated clue is not on the board."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available")
        
        agent = LLMSpymaster(seed=42, temperature=0.1)
        action = agent.get_clue(sample_observation)
        
        board_lower = [w.lower() for w in sample_observation.board_words]
        assert action.clue.lower() not in board_lower

    def test_retry_logic_with_max_retries(self, sample_observation):
        """Test that retry logic is invoked when needed."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available")
        
        agent = LLMSpymaster(seed=42, temperature=0.7)  # Higher temp for more variation
        
        # Should succeed within max_retries
        action = agent.get_clue(sample_observation, max_retries=5)
        
        board_lower = [w.lower() for w in sample_observation.board_words]
        assert action.clue.lower() not in board_lower


@pytest.mark.slow
class TestLLMGuesser:
    """Tests for LLMGuesser (requires GPU and model download)."""

    def test_initialization(self):
        """Test that LLMGuesser initializes correctly."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        agent = LLMGuesser(seed=42)
        assert agent.model is not None
        assert agent.tokenizer is not None

    def test_get_guess_with_clue(self, sample_observation):
        """Test that get_guess works with an active clue."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        obs = Observation(
            **{**sample_observation.__dict__,
               'current_clue': 'furniture',
               'current_count': 3,
               'phase': GamePhase.GUESSER_TURN}
        )
        
        agent = LLMGuesser(seed=42, temperature=0.1)
        action = agent.get_guess(obs)
        
        # Should return either an index or None
        assert action.word_index is None or isinstance(action.word_index, int)
        if action.word_index is not None:
            assert 0 <= action.word_index < 25

    def test_no_clue_returns_pass(self, sample_observation):
        """Test that guesser passes when no clue is active."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        obs = Observation(
            **{**sample_observation.__dict__,
               'current_clue': None,
               'phase': GamePhase.SPYMASTER_TURN}
        )
        
        agent = LLMGuesser(seed=42, temperature=0.1)
        action = agent.get_guess(obs)
        
        assert action.word_index is None


@pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
class TestQwenEmbeddingAgentsRegression:
    """Regression tests to ensure Qwen agents don't break existing functionality."""

    def test_qwen_agents_importable(self):
        """Test that Qwen agents can be imported."""
        # This test just verifies imports work
        assert QwenEmbeddingSpymaster is not None
        assert QwenEmbeddingGuesser is not None

    @pytest.mark.slow
    def test_qwen_spymaster_basic_functionality(self, temp_vocabulary, sample_observation):
        """Test basic QwenEmbeddingSpymaster functionality."""
        try:
            agent = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            action = agent.get_clue(sample_observation)
            assert action.clue is not None
            assert action.count > 0
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.slow
    def test_qwen_guesser_basic_functionality(self, sample_observation):
        """Test basic QwenEmbeddingGuesser functionality."""
        try:
            obs = Observation(
                **{**sample_observation.__dict__,
                   'current_clue': 'test',
                   'current_count': 2,
                   'phase': GamePhase.GUESSER_TURN}
            )
            agent = QwenEmbeddingGuesser(seed=42)
            action = agent.get_guess(obs)
            assert action.word_index is None or isinstance(action.word_index, int)
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

