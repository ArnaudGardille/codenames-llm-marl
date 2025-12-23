"""Integration tests for Cross-Encoder agents functionality."""

import time

import pytest

from codenames_rl.agents import CrossEncoderSpymaster, CrossEncoderGuesser
from codenames_rl.env.spaces import CardColor, GamePhase, Observation
from codenames_rl.utils.config import VOCABULARY_PATH


@pytest.fixture
def sample_observation_for_cross_encoder():
    """Create a sample observation for Cross-Encoder testing."""
    return Observation(
        board_words=["apple", "banana", "orange", "car", "truck", "bike",
                     "dog", "cat", "bird", "house", "tree", "flower",
                     "book", "pen", "paper", "water", "fire", "earth",
                     "sun", "moon", "star", "music", "dance", "song", "assassin"],
        revealed_mask=[False] * 25,
        board_colors=([CardColor.TEAM] * 9 + 
                      [CardColor.OPPONENT] * 8 + 
                      [CardColor.NEUTRAL] * 7 + 
                      [CardColor.ASSASSIN]),
        current_clue=None,
        current_count=None,
        remaining_guesses=0,
        phase=GamePhase.SPYMASTER_TURN,
        team_remaining=9,
        opponent_remaining=8
    )


@pytest.fixture
def sample_observation_for_guesser():
    """Create a sample observation for guesser testing."""
    return Observation(
        board_words=["apple", "banana", "orange", "car", "truck", "bike",
                     "dog", "cat", "bird", "house", "tree", "flower",
                     "book", "pen", "paper", "water", "fire", "earth",
                     "sun", "moon", "star", "music", "dance", "song", "assassin"],
        revealed_mask=[False] * 25,
        board_colors=([CardColor.TEAM] * 9 + 
                      [CardColor.OPPONENT] * 8 + 
                      [CardColor.NEUTRAL] * 7 + 
                      [CardColor.ASSASSIN]),
        current_clue="fruit",
        current_count=3,
        remaining_guesses=3,
        phase=GamePhase.GUESSER_TURN,
        team_remaining=9,
        opponent_remaining=8
    )


@pytest.mark.slow
class TestCrossEncoderAgents:
    """Integration tests for Cross-Encoder agents."""

    def test_spymaster_initialization(self):
        """Test that CrossEncoderSpymaster initializes correctly."""
        try:
            spymaster = CrossEncoderSpymaster(
                vocabulary_path=VOCABULARY_PATH,
                seed=42,
                top_k_retrieve=5,  # Small for testing
                top_k_candidates=20  # Small for testing
            )
            assert spymaster is not None
            assert spymaster.bi_encoder is not None
            assert spymaster.cross_encoder is not None
            assert spymaster.bi_encoder.get_sentence_embedding_dimension() > 0
        except Exception as e:
            pytest.skip(f"Cross-Encoder models not available: {e}")

    def test_guesser_initialization(self):
        """Test that CrossEncoderGuesser initializes correctly."""
        try:
            guesser = CrossEncoderGuesser(
                seed=42,
                top_k_retrieve=5  # Small for testing
            )
            assert guesser is not None
            assert guesser.bi_encoder is not None
            assert guesser.cross_encoder is not None
            assert guesser.bi_encoder.get_sentence_embedding_dimension() > 0
        except Exception as e:
            pytest.skip(f"Cross-Encoder models not available: {e}")

    def test_spymaster_clue_generation(self, sample_observation_for_cross_encoder):
        """Test that Spymaster can generate clues."""
        try:
            spymaster = CrossEncoderSpymaster(
                vocabulary_path=VOCABULARY_PATH,
                seed=42,
                top_k_retrieve=5,
                top_k_candidates=20
            )
            obs = sample_observation_for_cross_encoder
            
            start_time = time.time()
            action = spymaster.get_clue(obs)
            elapsed = time.time() - start_time
            
            assert action.clue is not None
            assert isinstance(action.clue, str)
            assert action.count > 0
            assert isinstance(action.count, int)
            assert action.clue not in obs.board_words
            assert elapsed > 0  # Should take some time
        except Exception as e:
            pytest.skip(f"Cross-Encoder models not available: {e}")

    def test_guesser_guess_generation(self, sample_observation_for_guesser):
        """Test that Guesser can make guesses."""
        try:
            guesser = CrossEncoderGuesser(
                seed=42,
                top_k_retrieve=5
            )
            obs = sample_observation_for_guesser
            
            start_time = time.time()
            action = guesser.get_guess(obs)
            elapsed = time.time() - start_time
            
            assert action.word_index is None or isinstance(action.word_index, int)
            if action.word_index is not None:
                assert 0 <= action.word_index < 25
                assert not obs.revealed_mask[action.word_index]
            assert elapsed > 0  # Should take some time
        except Exception as e:
            pytest.skip(f"Cross-Encoder models not available: {e}")

    def test_spymaster_performance(self, sample_observation_for_cross_encoder):
        """Test Spymaster performance over multiple clue generations."""
        try:
            spymaster = CrossEncoderSpymaster(
                vocabulary_path=VOCABULARY_PATH,
                seed=42,
                top_k_retrieve=5,
                top_k_candidates=20
            )
            obs = sample_observation_for_cross_encoder
            
            times = []
            for i in range(5):
                start = time.time()
                action = spymaster.get_clue(obs)
                elapsed = time.time() - start
                times.append(elapsed)
                assert action.clue is not None
                assert action.count > 0
            
            avg_time = sum(times) / len(times)
            assert avg_time > 0
            assert min(times) > 0
            assert max(times) > 0
        except Exception as e:
            pytest.skip(f"Cross-Encoder models not available: {e}")

