"""Unit tests for Codenames environment."""

import pytest
import numpy as np
from pathlib import Path

from codenames_rl.env import (
    CodenamesEnv,
    CardColor,
    GamePhase,
    SpymasterAction,
    GuesserAction,
    is_valid_clue,
    load_wordlist,
)


@pytest.fixture
def wordlist_path():
    """Path to test wordlist."""
    return str(Path(__file__).parent.parent.parent / "configs" / "wordlist.txt")


@pytest.fixture
def env(wordlist_path):
    """Create a fresh environment."""
    return CodenamesEnv(wordlist_path=wordlist_path)


class TestWordlistLoading:
    """Tests for wordlist loading."""
    
    def test_load_wordlist(self, wordlist_path):
        """Test that wordlist loads correctly."""
        words = load_wordlist(wordlist_path)
        assert len(words) >= 25
        assert all(isinstance(w, str) for w in words)
        assert all(w.strip() == w for w in words)  # No trailing whitespace
    
    def test_wordlist_not_found(self):
        """Test error when wordlist doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_wordlist("/nonexistent/path.txt")


class TestClueValidation:
    """Tests for clue validation rules."""
    
    def test_valid_clue(self):
        """Test that valid clues pass."""
        board = ["CAT", "DOG", "HOUSE"]
        is_valid, msg = is_valid_clue("ANIMAL", board)
        assert is_valid
        assert msg == ""
    
    def test_empty_clue(self):
        """Test that empty clue is rejected."""
        board = ["CAT", "DOG"]
        is_valid, msg = is_valid_clue("", board)
        assert not is_valid
        assert "empty" in msg.lower()
    
    def test_clue_with_spaces(self):
        """Test that clues with spaces are rejected."""
        board = ["CAT", "DOG"]
        is_valid, msg = is_valid_clue("TWO WORDS", board)
        assert not is_valid
        assert "space" in msg.lower()
    
    def test_clue_with_hyphens(self):
        """Test that clues with hyphens are rejected."""
        board = ["CAT", "DOG"]
        is_valid, msg = is_valid_clue("HYPH-EN", board)
        assert not is_valid
        assert "hyphen" in msg.lower()
    
    def test_clue_with_special_chars(self):
        """Test that clues with special characters are rejected."""
        board = ["CAT", "DOG"]
        is_valid, msg = is_valid_clue("CLUE!", board)
        assert not is_valid
        assert "alphanumeric" in msg.lower()
    
    def test_clue_on_board(self):
        """Test that clue matching board word is rejected."""
        board = ["CAT", "DOG", "HOUSE"]
        is_valid, msg = is_valid_clue("cat", board)  # Case insensitive
        assert not is_valid
        assert "board" in msg.lower()
    
    def test_clue_case_insensitive(self):
        """Test that clue validation is case insensitive."""
        board = ["CAT", "DOG"]
        is_valid, msg = is_valid_clue("Cat", board)
        assert not is_valid
        
        is_valid, msg = is_valid_clue("DOG", board)
        assert not is_valid


class TestEnvironmentReset:
    """Tests for environment reset and initialization."""
    
    def test_reset_returns_observation(self, env):
        """Test that reset returns valid observation."""
        obs, info = env.reset(seed=42)
        
        assert len(obs.board_words) == 25
        assert len(obs.revealed_mask) == 25
        assert len(obs.board_colors) == 25
        assert all(not revealed for revealed in obs.revealed_mask)
        assert obs.phase == GamePhase.SPYMASTER_TURN
        assert obs.current_clue is None
        assert obs.team_remaining == 9
        assert obs.opponent_remaining == 8
    
    def test_reset_deterministic(self, env):
        """Test that same seed produces same board."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert obs1.board_words == obs2.board_words
        assert obs1.board_colors == obs2.board_colors
    
    def test_reset_different_seeds(self, env):
        """Test that different seeds produce different boards."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)
        
        # Extremely unlikely to be identical
        assert obs1.board_words != obs2.board_words
    
    def test_card_distribution(self, env):
        """Test that cards are distributed correctly."""
        obs, _ = env.reset(seed=42)
        
        color_counts = {}
        for color in obs.board_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        assert color_counts[CardColor.TEAM] == 9
        assert color_counts[CardColor.OPPONENT] == 8
        assert color_counts[CardColor.NEUTRAL] == 7
        assert color_counts[CardColor.ASSASSIN] == 1


class TestSpymasterActions:
    """Tests for spymaster turn."""
    
    def test_valid_spymaster_action(self, env):
        """Test that valid clue transitions to guesser phase."""
        obs, _ = env.reset(seed=42)
        
        action = SpymasterAction(clue="ANIMAL", count=2)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.phase == GamePhase.GUESSER_TURN
        assert obs.current_clue == "ANIMAL"
        assert obs.current_count == 2
        assert obs.remaining_guesses == 3  # count + 1
        assert reward == 0.0
        assert not terminated
        assert "clue_given" in info
    
    def test_invalid_clue_penalty(self, env):
        """Test that invalid clue gets penalty."""
        obs, _ = env.reset(seed=42)
        
        # Use a word that's on the board
        board_word = obs.board_words[0]
        action = SpymasterAction(clue=board_word, count=2)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward < 0
        assert "invalid_clue" in info
        # Should remain in spymaster turn
        assert obs.phase == GamePhase.SPYMASTER_TURN


class TestGuesserActions:
    """Tests for guesser turn."""
    
    def test_correct_team_guess(self, env):
        """Test guessing a team card."""
        obs, _ = env.reset(seed=42)
        
        # Give a clue first
        action = SpymasterAction(clue="TESTCLUE", count=1)
        obs, _, _, _, _ = env.step(action)
        
        # Find a team card
        team_idx = next(i for i, c in enumerate(obs.board_colors) if c == CardColor.TEAM)
        
        initial_team_remaining = obs.team_remaining
        action = GuesserAction(word_index=team_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward == 1.0
        assert obs.revealed_mask[team_idx]
        assert obs.team_remaining == initial_team_remaining - 1
        assert info["correct"]
        assert info["color"] == CardColor.TEAM.value
    
    def test_opponent_guess_ends_turn(self, env):
        """Test that guessing opponent card ends turn."""
        obs, _ = env.reset(seed=42)
        
        # Give a clue
        action = SpymasterAction(clue="TESTCLUE", count=2)
        obs, _, _, _, _ = env.step(action)
        
        # Find opponent card
        opp_idx = next(i for i, c in enumerate(obs.board_colors) if c == CardColor.OPPONENT)
        
        action = GuesserAction(word_index=opp_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward == -1.0
        assert obs.phase == GamePhase.SPYMASTER_TURN
        assert obs.current_clue is None
        assert not info["correct"]
    
    def test_neutral_guess_ends_turn(self, env):
        """Test that guessing neutral card ends turn."""
        obs, _ = env.reset(seed=42)
        
        # Give a clue
        action = SpymasterAction(clue="TESTCLUE", count=2)
        obs, _, _, _, _ = env.step(action)
        
        # Find neutral card
        neutral_idx = next(i for i, c in enumerate(obs.board_colors) if c == CardColor.NEUTRAL)
        
        action = GuesserAction(word_index=neutral_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward == 0.0
        assert obs.phase == GamePhase.SPYMASTER_TURN
        assert not info["correct"]
    
    def test_assassin_terminates_game(self, env):
        """Test that hitting assassin ends game with loss."""
        obs, _ = env.reset(seed=42)
        
        # Give a clue
        action = SpymasterAction(clue="TESTCLUE", count=1)
        obs, _, _, _, _ = env.step(action)
        
        # Find assassin card
        assassin_idx = next(i for i, c in enumerate(obs.board_colors) if c == CardColor.ASSASSIN)
        
        action = GuesserAction(word_index=assassin_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward == -10.0
        assert terminated
        assert obs.phase == GamePhase.GAME_OVER
        assert info["result"] == "loss_assassin"
    
    def test_pass_ends_turn(self, env):
        """Test that passing ends the turn."""
        obs, _ = env.reset(seed=42)
        
        # Give a clue
        action = SpymasterAction(clue="TESTCLUE", count=2)
        obs, _, _, _, _ = env.step(action)
        
        # Pass
        action = GuesserAction(word_index=None)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward == 0.0
        assert obs.phase == GamePhase.SPYMASTER_TURN
        assert obs.current_clue is None
        assert info["action"] == "pass"
    
    def test_invalid_word_index(self, env):
        """Test that invalid word index is rejected."""
        obs, _ = env.reset(seed=42)
        
        action = SpymasterAction(clue="TESTCLUE", count=1)
        obs, _, _, _, _ = env.step(action)
        
        # Try invalid index
        action = GuesserAction(word_index=999)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward < 0
        assert "error" in info
    
    def test_already_revealed(self, env):
        """Test that guessing already revealed card is rejected."""
        obs, _ = env.reset(seed=42)
        
        action = SpymasterAction(clue="TESTCLUE", count=2)
        obs, _, _, _, _ = env.step(action)
        
        # Find and reveal a card
        team_idx = next(i for i, c in enumerate(obs.board_colors) if c == CardColor.TEAM)
        action = GuesserAction(word_index=team_idx)
        obs, _, _, _, _ = env.step(action)
        
        # Try to guess the same card again (still in guesser phase)
        action = GuesserAction(word_index=team_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward < 0
        assert "already revealed" in info["error"].lower()


class TestTerminalConditions:
    """Tests for game end conditions."""
    
    def test_win_condition(self, env):
        """Test that revealing all team cards wins."""
        obs, _ = env.reset(seed=42)
        
        # Find all team cards
        team_indices = [i for i, c in enumerate(obs.board_colors) if c == CardColor.TEAM]
        
        for i, team_idx in enumerate(team_indices):
            # If in spymaster phase, give clue
            if obs.phase == GamePhase.SPYMASTER_TURN:
                action = SpymasterAction(clue=f"CLUE{i}", count=1)
                obs, _, terminated, _, _ = env.step(action)
                
                if terminated:
                    break
            
            # Guess team card
            action = GuesserAction(word_index=team_idx)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i == len(team_indices) - 1:
                # Last team card
                assert terminated
                assert obs.phase == GamePhase.GAME_OVER
                assert reward == 10.0
                assert info["result"] == "win"
    
    def test_game_over_raises_error(self, env):
        """Test that stepping after game over raises error."""
        obs, _ = env.reset(seed=42)
        
        # Give clue and hit assassin
        action = SpymasterAction(clue="TESTCLUE", count=1)
        env.step(action)
        
        assassin_idx = next(i for i, c in enumerate(obs.board_colors) if c == CardColor.ASSASSIN)
        action = GuesserAction(word_index=assassin_idx)
        env.step(action)
        
        # Try to take another action
        with pytest.raises(RuntimeError, match="Game is over"):
            env.step(SpymasterAction(clue="ANOTHER", count=1))


class TestActionTypeValidation:
    """Tests for action type validation."""
    
    def test_wrong_action_type_spymaster(self, env):
        """Test that guesser action during spymaster turn raises error."""
        env.reset(seed=42)
        
        with pytest.raises(TypeError, match="Expected SpymasterAction"):
            env.step(GuesserAction(word_index=0))
    
    def test_wrong_action_type_guesser(self, env):
        """Test that spymaster action during guesser turn raises error."""
        env.reset(seed=42)
        
        # Transition to guesser turn
        env.step(SpymasterAction(clue="TESTCLUE", count=1))
        
        with pytest.raises(TypeError, match="Expected GuesserAction"):
            env.step(SpymasterAction(clue="ANOTHER", count=1))

