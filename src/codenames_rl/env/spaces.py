"""Action spaces, observation spaces and game enums.

These are the *type* surfaces every agent and env layer agrees on. Keeping
them tiny and dataclass-based means nothing here imports torch/gym, so the
file is cheap to import from tests, the Streamlit app, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class CardColor(str, Enum):
    """Color of a card on the board.

    Values are the lowercase strings the harness compares against in
    ``info["color"]`` — keep them stable.
    """

    TEAM = "team"
    OPPONENT = "opponent"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"


class GamePhase(Enum):
    """Coarse phase of the game state machine."""

    SPYMASTER_TURN = "spymaster_turn"
    GUESSER_TURN = "guesser_turn"
    GAME_OVER = "game_over"


@dataclass
class SpymasterAction:
    """A clue given by the Spymaster."""

    clue: str
    count: int


@dataclass
class GuesserAction:
    """A guess (or STOP/pass when ``word_index is None``)."""

    word_index: Optional[int]


@dataclass
class Observation:
    """Observation seen by an agent.

    Notes
    -----
    * ``board_colors`` is from the perspective of the team that is currently
      acting (TEAM = your cards, OPPONENT = the other team's cards). In the
      adversarial env the underlying absolute colors are flipped for Team B.
    * In the cooperative env, ``board_colors`` is also exposed to the
      Guesser. Real Guessers should not look at it; agents that do are
      cheating by construction. This is a research codebase, not a referee.
    """

    board_words: List[str]
    board_colors: List[CardColor]
    revealed_mask: List[bool]
    phase: GamePhase
    current_clue: Optional[str]
    current_count: Optional[int]
    remaining_guesses: int
    team_remaining: int
    opponent_remaining: int
