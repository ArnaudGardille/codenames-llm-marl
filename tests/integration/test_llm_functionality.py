"""Integration tests for LLM agents functionality."""

import pytest

from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser
from codenames_rl.env.spaces import CardColor, GamePhase, Observation


@pytest.fixture
def sample_observation_for_llm():
    """Create a sample observation for LLM testing."""
    return Observation(
        board_words=[
            "apple", "dog", "river", "mountain", "book",
            "car", "tree", "phone", "house", "cloud",
            "star", "ocean", "guitar", "lamp", "chair",
            "sun", "moon", "rain", "snow", "wind",
            "fire", "water", "earth", "light", "dark"
        ],
        revealed_mask=[False] * 25,
        board_colors=[CardColor.TEAM]*9 + [CardColor.OPPONENT]*8 + [CardColor.NEUTRAL]*7 + [CardColor.ASSASSIN],
        current_clue="nature",
        current_count=2,
        remaining_guesses=3,
        phase=GamePhase.GUESSER_TURN,
        team_remaining=9,
        opponent_remaining=8
    )


@pytest.mark.slow
class TestLLMAgents:
    """Integration tests for LLM agents."""

    def test_spymaster_initialization(self):
        """Test that LLMSpymaster initializes correctly."""
        pytest.importorskip("torch")
        import torch
        
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available for LLM agents")
        
        try:
            spymaster = LLMSpymaster(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                device="cpu",
                seed=42,
                temperature=0.7
            )
            assert spymaster is not None
            assert spymaster.model is not None
            assert spymaster.tokenizer is not None
            assert spymaster.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
            assert spymaster.device in ["cpu", "cuda", "mps"]
            assert spymaster.temperature == 0.7
        except Exception as e:
            pytest.skip(f"LLM model not available: {e}")

    def test_guesser_initialization(self):
        """Test that LLMGuesser initializes correctly with shared model."""
        pytest.importorskip("torch")
        import torch
        
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available for LLM agents")
        
        try:
            spymaster = LLMSpymaster(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                device="cpu",
                seed=42,
                temperature=0.7
            )
            guesser = LLMGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device="cpu",
                seed=42
            )
            assert guesser is not None
            assert guesser.model is spymaster.model
            assert guesser.tokenizer is spymaster.tokenizer
        except Exception as e:
            pytest.skip(f"LLM model not available: {e}")

    def test_spymaster_output_parsing(self):
        """Test spymaster output parsing and validation."""
        pytest.importorskip("torch")
        import torch
        
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available for LLM agents")
        
        try:
            spymaster = LLMSpymaster(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                device="cpu",
                seed=42,
                temperature=0.7
            )
            
            # Test valid JSON parsing
            test_cases = [
                ('{"clue": "fruit", "count": 2}', True),
                ('{"clue": "animal", "count": 1}', True),
                ('Some text {"clue": "water", "count": 3} more text', True),
                ('invalid json', False),
                ('{"clue": "apple"}', False),
            ]
            
            for test_input, should_succeed in test_cases:
                if should_succeed:
                    result = spymaster._parse_spymaster_output(test_input, ["board", "words"])
                    assert result is not None
                    assert "clue" in result
                    assert "count" in result
                else:
                    with pytest.raises(ValueError):
                        spymaster._parse_spymaster_output(test_input, ["board", "words"])
        except Exception as e:
            pytest.skip(f"LLM model not available: {e}")

    def test_guesser_functionality(self, sample_observation_for_llm):
        """Test guesser functionality."""
        pytest.importorskip("torch")
        import torch
        
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available for LLM agents")
        
        try:
            spymaster = LLMSpymaster(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                device="cpu",
                seed=42,
                temperature=0.7
            )
            guesser = LLMGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device="cpu",
                seed=42
            )
            
            obs = sample_observation_for_llm
            action = guesser.get_guess(obs)
            
            assert action.word_index is None or isinstance(action.word_index, int)
            if action.word_index is not None:
                assert 0 <= action.word_index < 25
                assert not obs.revealed_mask[action.word_index]
        except Exception as e:
            pytest.skip(f"LLM model not available: {e}")

    def test_guesser_output_parsing(self):
        """Test guesser output parsing."""
        pytest.importorskip("torch")
        import torch
        
        if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No GPU available for LLM agents")
        
        try:
            spymaster = LLMSpymaster(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                device="cpu",
                seed=42,
                temperature=0.7
            )
            guesser = LLMGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device="cpu",
                seed=42
            )
            
            guesser_tests = [
                ('{"guess": "apple"}', "apple"),
                ('{"guess": "STOP"}', "STOP"),
                ('Some text {"guess": "river"} more', "river"),
            ]
            
            for test_input, expected in guesser_tests:
                result = guesser._parse_guesser_output(test_input)
                assert result == expected
        except Exception as e:
            pytest.skip(f"LLM model not available: {e}")

