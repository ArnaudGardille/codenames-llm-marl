"""Pure game logic for 4-player adversarial Codenames.

Two teams (``team_a`` and ``team_b``) each with their own Spymaster + Guesser
alternate turns. The board has *absolute* colors (TEAM = team_a's cards,
OPPONENT = team_b's cards); the PettingZoo and Gym wrappers flip them per
agent so each team sees TEAM = "mine" and OPPONENT = "theirs".

This module is wrapper-agnostic on purpose — both ``adversarial_pz`` and
``adversarial_gym`` drive it directly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .spaces import CardColor
from .validation import is_valid_clue, load_wordlist

_TEAM_A_COUNT = 9
_TEAM_B_COUNT = 8
_NEUTRAL_COUNT = 7
_ASSASSIN_COUNT = 1
_BOARD_SIZE = _TEAM_A_COUNT + _TEAM_B_COUNT + _NEUTRAL_COUNT + _ASSASSIN_COUNT  # 25

_REWARD_OWN = 1.0
_REWARD_OPP = -1.0
_REWARD_NEUTRAL = 0.0
_REWARD_ASSASSIN = -10.0
_REWARD_WIN = 10.0
_REWARD_INVALID = -2.0

_TEAMS = ("team_a", "team_b")


class CodenamesAdversarialCore:
    """4-player Codenames game state.

    Notes
    -----
    * ``board_colors_absolute[i] == CardColor.TEAM`` means card ``i``
      belongs to team_a; ``OPPONENT`` means it belongs to team_b. This is
      flipped per perspective by :meth:`get_board_colors_for_team`.
    * Phases and team identifiers are plain strings here (``'spymaster'``,
      ``'guesser'``, ``'team_a'``, ``'team_b'``) — the wrappers convert
      them to ``GamePhase`` enums in the observation they expose.
    """

    def __init__(self, wordlist_path: str, rng: np.random.Generator) -> None:
        self.wordlist_path = wordlist_path
        wordlist = load_wordlist(wordlist_path)
        if len(wordlist) < _BOARD_SIZE:
            raise ValueError(
                f"Wordlist needs ≥{_BOARD_SIZE} words, got {len(wordlist)}"
            )
        self.board_words: List[str] = rng.choice(
            wordlist, size=_BOARD_SIZE, replace=False
        ).tolist()

        colors = (
            [CardColor.TEAM] * _TEAM_A_COUNT
            + [CardColor.OPPONENT] * _TEAM_B_COUNT
            + [CardColor.NEUTRAL] * _NEUTRAL_COUNT
            + [CardColor.ASSASSIN] * _ASSASSIN_COUNT
        )
        rng.shuffle(colors)
        self.board_colors_absolute: List[CardColor] = colors
        self.revealed_mask: List[bool] = [False] * _BOARD_SIZE

        self.team_a_remaining = _TEAM_A_COUNT
        self.team_b_remaining = _TEAM_B_COUNT

        self.current_team = "team_a"
        self.current_phase = "spymaster"
        self.current_clue: Optional[str] = None
        self.current_count: Optional[int] = None
        self.remaining_guesses = 0

        self.game_over = False
        self.winner: Optional[str] = None  # 'team_a' / 'team_b' / None

    # ----------------------------------------------------------- accessors

    def get_board_colors_for_team(self, team: str) -> List[CardColor]:
        """Return colors flipped so that ``team`` sees TEAM = its own cards."""
        if team == "team_a":
            return list(self.board_colors_absolute)
        if team == "team_b":
            return [_flip(c) for c in self.board_colors_absolute]
        raise ValueError(f"Unknown team: {team!r}")

    def remaining_for(self, team: str) -> int:
        return self.team_a_remaining if team == "team_a" else self.team_b_remaining

    # -------------------------------------------------------------- actions

    def execute_spymaster_action(
        self, team: str, clue: str, count: int
    ) -> Tuple[float, bool, dict]:
        self._require_active(team, "spymaster")
        ok, err = is_valid_clue(clue, self.board_words)
        if not ok:
            return _REWARD_INVALID, False, {"invalid_clue": True, "error": err}
        self.current_clue = clue
        self.current_count = count
        self.remaining_guesses = max(count, 1) + 1
        self.current_phase = "guesser"
        return 0.0, False, {"clue_given": clue}

    def execute_guesser_action(
        self, team: str, word_index: Optional[int]
    ) -> Tuple[float, bool, dict]:
        self._require_active(team, "guesser")

        # Pass / STOP.
        if word_index is None:
            self._end_turn()
            return 0.0, False, {"action": "pass"}

        if (
            not isinstance(word_index, int)
            or word_index < 0
            or word_index >= _BOARD_SIZE
        ):
            return _REWARD_INVALID, False, {"error": f"invalid word_index: {word_index}"}
        if self.revealed_mask[word_index]:
            return (
                _REWARD_INVALID,
                False,
                {"error": f"word at {word_index} is already revealed"},
            )

        absolute = self.board_colors_absolute[word_index]
        self.revealed_mask[word_index] = True
        self.remaining_guesses -= 1
        relative = absolute if team == "team_a" else _flip(absolute)
        info = {"color": relative.value, "absolute_color": absolute.value}

        # Assassin → guessing team loses.
        if absolute == CardColor.ASSASSIN:
            other = _other(team)
            self._terminate(winner=other)
            info.update(correct=False, result="loss_assassin")
            return _REWARD_ASSASSIN, True, info

        # Own card.
        if relative == CardColor.TEAM:
            if team == "team_a":
                self.team_a_remaining -= 1
            else:
                self.team_b_remaining -= 1
            info["correct"] = True
            if self.remaining_for(team) == 0:
                self._terminate(winner=team)
                info["result"] = "win"
                return _REWARD_WIN, True, info
            if self.remaining_guesses <= 0:
                self._end_turn()
            return _REWARD_OWN, False, info

        # Opponent's card — turn ends, opponent's pool shrinks.
        if relative == CardColor.OPPONENT:
            if team == "team_a":
                self.team_b_remaining -= 1
            else:
                self.team_a_remaining -= 1
            info["correct"] = False
            self._end_turn()
            if self.remaining_for(_other(team)) == 0:
                self._terminate(winner=_other(team))
                info["result"] = "loss_opponent_won"
                return _REWARD_OPP, True, info
            return _REWARD_OPP, False, info

        # Neutral.
        info["correct"] = False
        self._end_turn()
        return _REWARD_NEUTRAL, False, info

    # ------------------------------------------------------------ internals

    def _require_active(self, team: str, phase: str) -> None:
        if self.game_over:
            raise RuntimeError("Game is over")
        if team != self.current_team:
            raise RuntimeError(
                f"Not {team}'s turn (current: {self.current_team})"
            )
        if self.current_phase != phase:
            raise RuntimeError(
                f"Wrong phase: expected {self.current_phase!r}, got {phase!r}"
            )

    def _end_turn(self) -> None:
        self.current_team = _other(self.current_team)
        self.current_phase = "spymaster"
        self.current_clue = None
        self.current_count = None
        self.remaining_guesses = 0

    def _terminate(self, winner: Optional[str]) -> None:
        self.game_over = True
        self.winner = winner
        self.current_clue = None
        self.current_count = None
        self.remaining_guesses = 0


def _other(team: str) -> str:
    return "team_b" if team == "team_a" else "team_a"


def _flip(color: CardColor) -> CardColor:
    if color == CardColor.TEAM:
        return CardColor.OPPONENT
    if color == CardColor.OPPONENT:
        return CardColor.TEAM
    return color
