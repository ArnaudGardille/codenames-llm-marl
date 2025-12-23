"""Shared pytest fixtures and configuration for all tests."""

import tempfile
from pathlib import Path

import pytest

from codenames_rl.env.spaces import CardColor, GamePhase, Observation

# Check if torch and transformers are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def temp_vocabulary():
    """Create a temporary vocabulary file for testing."""
    vocab_words = [
        "apple", "banana", "cherry", "dog", "elephant",
        "forest", "guitar", "house", "island", "jungle",
        "kitchen", "laptop", "mountain", "notebook", "ocean",
        "piano", "queen", "river", "sunset", "tree",
        "umbrella", "violin", "water", "xylophone", "yellow",
        "zebra", "animal", "fruit", "music", "nature"
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(vocab_words))
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_wordlist():
    """Create a temporary wordlist file for testing."""
    words = [
        "car", "book", "phone", "chair", "lamp",
        "desk", "window", "door", "wall", "floor",
        "ceiling", "table", "pen", "paper", "ink",
        "light", "dark", "sun", "moon", "star",
        "cloud", "rain", "snow", "wind", "storm",
        "fire", "ice", "metal", "wood", "stone"
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(words))
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_observation():
    """Create a sample observation for testing."""
    return Observation(
        board_words=[
            "car", "book", "phone", "chair", "lamp",
            "desk", "window", "door", "wall", "floor",
            "ceiling", "table", "pen", "paper", "ink",
            "light", "dark", "sun", "moon", "star",
            "cloud", "rain", "snow", "wind", "storm"
        ],
        revealed_mask=[False] * 25,
        board_colors=[
            CardColor.TEAM, CardColor.TEAM, CardColor.TEAM, CardColor.TEAM,
            CardColor.TEAM, CardColor.TEAM, CardColor.TEAM, CardColor.TEAM,
            CardColor.TEAM,  # 9 team
            CardColor.OPPONENT, CardColor.OPPONENT, CardColor.OPPONENT,
            CardColor.OPPONENT, CardColor.OPPONENT, CardColor.OPPONENT,
            CardColor.OPPONENT, CardColor.OPPONENT,  # 8 opponent
            CardColor.NEUTRAL, CardColor.NEUTRAL, CardColor.NEUTRAL,
            CardColor.NEUTRAL, CardColor.NEUTRAL, CardColor.NEUTRAL,
            CardColor.NEUTRAL,  # 7 neutral
            CardColor.ASSASSIN  # 1 assassin
        ],
        current_clue=None,
        current_count=None,
        remaining_guesses=0,
        phase=GamePhase.SPYMASTER_TURN,
        team_remaining=9,
        opponent_remaining=8
    )


@pytest.fixture
def wordlist_path():
    """Path to test wordlist."""
    return str(Path(__file__).parent.parent / "configs" / "wordlist.txt")

