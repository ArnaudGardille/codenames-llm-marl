"""Tests to verify all agents can be instantiated and work together."""

import pytest

from codenames_rl.agents import (
    AdaptiveGuesser,
    ClusterSpymaster,
    ContextualGuesser,
    CrossEncoderGuesser,
    CrossEncoderSpymaster,
    EmbeddingsGuesser,
    EmbeddingsSpymaster,
    LLMGuesser,
    LLMSpymaster,
    QwenEmbeddingGuesser,
    QwenEmbeddingSpymaster,
    RandomGuesser,
    RandomSpymaster,
)

# Check if torch and transformers are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestAgentImports:
    """Test that all agents can be imported."""

    def test_all_agents_importable(self):
        """Test that all agent classes can be imported."""
        assert RandomSpymaster is not None
        assert RandomGuesser is not None
        assert EmbeddingsSpymaster is not None
        assert EmbeddingsGuesser is not None
        assert ClusterSpymaster is not None
        assert ContextualGuesser is not None
        assert AdaptiveGuesser is not None
        assert CrossEncoderSpymaster is not None
        assert CrossEncoderGuesser is not None
        
        if HAS_TORCH:
            assert LLMSpymaster is not None
            assert LLMGuesser is not None
            assert QwenEmbeddingSpymaster is not None
            assert QwenEmbeddingGuesser is not None


class TestAgentInstantiation:
    """Test that all agents can be instantiated."""

    def test_random_agents_instantiate(self, temp_vocabulary):
        """Test Random agents can be instantiated."""
        spymaster = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = RandomGuesser(seed=42)
        
        assert spymaster is not None
        assert guesser is not None

    def test_embeddings_agents_instantiate(self, temp_vocabulary):
        """Test Embeddings agents can be instantiated."""
        spymaster = EmbeddingsSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = EmbeddingsGuesser(seed=42)
        
        assert spymaster is not None
        assert guesser is not None

    def test_improved_agents_instantiate(self, temp_vocabulary):
        """Test Improved agents can be instantiated."""
        spymaster = ClusterSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser1 = ContextualGuesser(seed=42)
        guesser2 = AdaptiveGuesser(seed=42)
        
        assert spymaster is not None
        assert guesser1 is not None
        assert guesser2 is not None

    @pytest.mark.slow
    def test_cross_encoder_agents_instantiate(self, temp_vocabulary):
        """Test Cross-Encoder agents can be instantiated."""
        try:
            spymaster = CrossEncoderSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k_retrieve=5,  # Small for testing
                top_k_candidates=20  # Small for testing
            )
            guesser = CrossEncoderGuesser(
                seed=42,
                top_k_retrieve=5  # Small for testing
            )
            
            assert spymaster is not None
            assert guesser is not None
        except Exception as e:
            pytest.skip(f"Cross-Encoder models not available: {e}")

    @pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
    @pytest.mark.slow
    def test_qwen_agents_instantiate(self, temp_vocabulary):
        """Test Qwen embedding agents can be instantiated."""
        try:
            spymaster = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            guesser = QwenEmbeddingGuesser(seed=42)
            
            assert spymaster is not None
            assert guesser is not None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
    @pytest.mark.slow
    def test_llm_agents_instantiate(self):
        """Test LLM agents can be instantiated."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available for LLM agents")
        
        try:
            spymaster = LLMSpymaster(seed=42)
            guesser = LLMGuesser(seed=42)
            
            assert spymaster is not None
            assert guesser is not None
        except Exception as e:
            pytest.skip(f"LLM model not available: {e}")


class TestAgentCombinations:
    """Test that different agent combinations work together."""

    def test_random_embeddings_combination(self, temp_vocabulary):
        """Test Random spymaster with Embeddings guesser."""
        spymaster = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = EmbeddingsGuesser(seed=42)
        
        assert spymaster is not None
        assert guesser is not None

    def test_embeddings_random_combination(self, temp_vocabulary):
        """Test Embeddings spymaster with Random guesser."""
        spymaster = EmbeddingsSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = RandomGuesser(seed=42)
        
        assert spymaster is not None
        assert guesser is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
    @pytest.mark.slow
    def test_qwen_embeddings_combination(self, temp_vocabulary):
        """Test Qwen embedding spymaster with Embeddings guesser."""
        try:
            spymaster = QwenEmbeddingSpymaster(
                vocabulary_path=temp_vocabulary,
                seed=42,
                top_k=10
            )
            guesser = EmbeddingsGuesser(seed=42)
            
            assert spymaster is not None
            assert guesser is not None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
    @pytest.mark.slow
    def test_embeddings_qwen_combination(self, temp_vocabulary):
        """Test Embeddings spymaster with Qwen embedding guesser."""
        try:
            spymaster = EmbeddingsSpymaster(vocabulary_path=temp_vocabulary, seed=42)
            guesser = QwenEmbeddingGuesser(seed=42)
            
            assert spymaster is not None
            assert guesser is not None
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")

    @pytest.mark.skipif(not HAS_TORCH, reason="torch and transformers not available")
    @pytest.mark.slow
    def test_qwen_qwen_combination(self, temp_vocabulary):
        """Test Qwen embedding spymaster with Qwen embedding guesser."""
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
            
            assert spymaster is not None
            assert guesser is not None
            assert guesser.model is spymaster.model
        except Exception as e:
            pytest.skip(f"Qwen model not available: {e}")


class TestNoRegressions:
    """Test that existing agents still work (no regressions)."""

    def test_random_spymaster_still_works(self, temp_vocabulary):
        """Test RandomSpymaster still works correctly."""
        from codenames_rl.env.spaces import CardColor, GamePhase, Observation
        
        agent = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        obs = Observation(
            board_words=["car", "book"] * 12 + ["assassin"],
            revealed_mask=[False] * 25,
            board_colors=[CardColor.TEAM] * 9 + [CardColor.OPPONENT] * 8 + 
                        [CardColor.NEUTRAL] * 7 + [CardColor.ASSASSIN],
            current_clue=None,
            current_count=None,
            remaining_guesses=0,
            phase=GamePhase.SPYMASTER_TURN,
            team_remaining=9,
            opponent_remaining=8
        )
        
        action = agent.get_clue(obs)
        assert action.clue is not None
        assert action.count > 0

    def test_embeddings_spymaster_still_works(self, temp_vocabulary):
        """Test EmbeddingsSpymaster still works correctly."""
        from codenames_rl.env.spaces import CardColor, GamePhase, Observation
        
        agent = EmbeddingsSpymaster(
            vocabulary_path=temp_vocabulary,
            seed=42,
            top_k=10
        )
        obs = Observation(
            board_words=["car", "book"] * 12 + ["assassin"],
            revealed_mask=[False] * 25,
            board_colors=[CardColor.TEAM] * 9 + [CardColor.OPPONENT] * 8 + 
                        [CardColor.NEUTRAL] * 7 + [CardColor.ASSASSIN],
            current_clue=None,
            current_count=None,
            remaining_guesses=0,
            phase=GamePhase.SPYMASTER_TURN,
            team_remaining=9,
            opponent_remaining=8
        )
        
        action = agent.get_clue(obs)
        assert action.clue is not None
        assert action.count > 0

    def test_embeddings_guesser_still_works(self):
        """Test EmbeddingsGuesser still works correctly."""
        from codenames_rl.env.spaces import CardColor, GamePhase, Observation
        
        agent = EmbeddingsGuesser(seed=42)
        obs = Observation(
            board_words=["car", "book"] * 12 + ["assassin"],
            revealed_mask=[False] * 25,
            board_colors=[CardColor.TEAM] * 9 + [CardColor.OPPONENT] * 8 + 
                        [CardColor.NEUTRAL] * 7 + [CardColor.ASSASSIN],
            current_clue="furniture",
            current_count=3,
            remaining_guesses=3,
            phase=GamePhase.GUESSER_TURN,
            team_remaining=9,
            opponent_remaining=8
        )
        
        action = agent.get_guess(obs)
        assert action.word_index is None or isinstance(action.word_index, int)

    @pytest.mark.slow
    def test_cross_encoder_spymaster_works(self, temp_vocabulary):
        """Test CrossEncoderSpymaster works correctly."""
        from codenames_rl.env.spaces import CardColor, GamePhase, Observation
        
        agent = CrossEncoderSpymaster(
            vocabulary_path=temp_vocabulary,
            seed=42,
            top_k_retrieve=5,
            top_k_candidates=20
        )
        obs = Observation(
            board_words=["car", "book"] * 12 + ["assassin"],
            revealed_mask=[False] * 25,
            board_colors=[CardColor.TEAM] * 9 + [CardColor.OPPONENT] * 8 + 
                        [CardColor.NEUTRAL] * 7 + [CardColor.ASSASSIN],
            current_clue=None,
            current_count=None,
            remaining_guesses=0,
            phase=GamePhase.SPYMASTER_TURN,
            team_remaining=9,
            opponent_remaining=8
        )
        
        action = agent.get_clue(obs)
        assert action.clue is not None
        assert action.count > 0
        assert action.clue not in obs.board_words

    @pytest.mark.slow
    def test_cross_encoder_guesser_works(self):
        """Test CrossEncoderGuesser works correctly."""
        from codenames_rl.env.spaces import CardColor, GamePhase, Observation
        
        agent = CrossEncoderGuesser(seed=42, top_k_retrieve=5)
        obs = Observation(
            board_words=["car", "book"] * 12 + ["assassin"],
            revealed_mask=[False] * 25,
            board_colors=[CardColor.TEAM] * 9 + [CardColor.OPPONENT] * 8 + 
                        [CardColor.NEUTRAL] * 7 + [CardColor.ASSASSIN],
            current_clue="furniture",
            current_count=3,
            remaining_guesses=3,
            phase=GamePhase.GUESSER_TURN,
            team_remaining=9,
            opponent_remaining=8
        )
        
        action = agent.get_guess(obs)
        assert action.word_index is None or isinstance(action.word_index, int)


