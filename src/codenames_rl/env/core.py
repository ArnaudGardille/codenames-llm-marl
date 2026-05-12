"""Cooperative single-team Codenames environment.

A Gymnasium-style env: one team plays, the other team's cards sit on the
board purely as penalty targets and never act on their own. Designed to be
the simplest possible MDP for SFT/DPO/GRPO training.

Card distribution on a 25-card board: 9 TEAM / 8 OPPONENT / 7 NEUTRAL /
1 ASSASSIN. Per-turn guess budget is ``count + 1`` per the official rules.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from .spaces import (
    CardColor,
    GamePhase,
    GuesserAction,
    Observation,
    SpymasterAction,
)
from .validation import is_valid_clue, load_wordlist

# Standard Codenames distribution for the team that starts.
_TEAM_COUNT = 9
_OPPONENT_COUNT = 8
_NEUTRAL_COUNT = 7
_ASSASSIN_COUNT = 1
_BOARD_SIZE = _TEAM_COUNT + _OPPONENT_COUNT + _NEUTRAL_COUNT + _ASSASSIN_COUNT  # 25

# Reward magnitudes. The harness already reads `info["color"]` to compute
# its own win/loss outcomes, so these are mostly for training signal.
_REWARD_TEAM = 1.0
_REWARD_OPPONENT = -1.0
_REWARD_NEUTRAL = 0.0
_REWARD_ASSASSIN = -10.0
_REWARD_WIN = 10.0
_REWARD_INVALID = -2.0


class CodenamesEnv:
    """Cooperative single-team Codenames env."""

    def __init__(
        self,
        wordlist_path: str,
        max_guesses: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self.wordlist_path = wordlist_path
        # ``max_guesses`` caps the per-turn guess budget at clue time so a
        # Spymaster cannot grant the Guesser an unbounded number of tries.
        self.max_guesses = max_guesses if max_guesses is not None else _BOARD_SIZE
        self.render_mode = render_mode
        self._wordlist = load_wordlist(wordlist_path)
        if len(self._wordlist) < _BOARD_SIZE:
            raise ValueError(
                f"Wordlist needs at least {_BOARD_SIZE} unique words, "
                f"got {len(self._wordlist)} in {wordlist_path}"
            )
        # State filled by reset(); see _reset_state for the canonical layout.
        self._reset_state()

    # ------------------------------------------------------------------ API

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, dict]:
        rng = np.random.default_rng(seed)
        words = rng.choice(self._wordlist, size=_BOARD_SIZE, replace=False).tolist()
        colors = (
            [CardColor.TEAM] * _TEAM_COUNT
            + [CardColor.OPPONENT] * _OPPONENT_COUNT
            + [CardColor.NEUTRAL] * _NEUTRAL_COUNT
            + [CardColor.ASSASSIN] * _ASSASSIN_COUNT
        )
        rng.shuffle(colors)

        self.board_words = words
        self.board_colors = colors
        self.revealed_mask = [False] * _BOARD_SIZE
        self.team_remaining = _TEAM_COUNT
        self.opponent_remaining = _OPPONENT_COUNT
        self.phase = GamePhase.SPYMASTER_TURN
        self.current_clue = None
        self.current_count = None
        self.remaining_guesses = 0
        self.game_over = False
        return self._observation(), {}

    def step(
        self, action: Union[SpymasterAction, GuesserAction]
    ) -> Tuple[Observation, float, bool, bool, dict]:
        if self.game_over:
            raise RuntimeError("Game is over; call reset() before stepping again")

        if self.phase == GamePhase.SPYMASTER_TURN:
            if not isinstance(action, SpymasterAction):
                raise TypeError(
                    f"Expected SpymasterAction during spymaster turn, got {type(action).__name__}"
                )
            return self._step_spymaster(action)

        if self.phase == GamePhase.GUESSER_TURN:
            if not isinstance(action, GuesserAction):
                raise TypeError(
                    f"Expected GuesserAction during guesser turn, got {type(action).__name__}"
                )
            return self._step_guesser(action)

        # GAME_OVER (defensive — already guarded by self.game_over above).
        raise RuntimeError("Game is over; call reset() before stepping again")

    def render(self) -> None:
        if self.render_mode != "human":
            return
        rows = []
        for row in range(5):
            cells = []
            for col in range(5):
                i = row * 5 + col
                word = self.board_words[i]
                color = self.board_colors[i].value[0].upper()
                marker = "✓" if self.revealed_mask[i] else " "
                cells.append(f"{marker}{color} {word:<14}")
            rows.append(" | ".join(cells))
        print("\n".join(rows))
        print(
            f"phase={self.phase.value} team_left={self.team_remaining} "
            f"opp_left={self.opponent_remaining} guesses={self.remaining_guesses}"
        )

    def close(self) -> None:
        # No external resources to release; method exists for Gym-API symmetry.
        pass

    # -------------------------------------------------------------- internals

    def _reset_state(self) -> None:
        self.board_words: List[str] = []
        self.board_colors: List[CardColor] = []
        self.revealed_mask: List[bool] = []
        self.team_remaining = 0
        self.opponent_remaining = 0
        self.phase = GamePhase.SPYMASTER_TURN
        self.current_clue: Optional[str] = None
        self.current_count: Optional[int] = None
        self.remaining_guesses = 0
        self.game_over = False

    def _observation(self) -> Observation:
        return Observation(
            board_words=list(self.board_words),
            board_colors=list(self.board_colors),
            revealed_mask=list(self.revealed_mask),
            phase=self.phase,
            current_clue=self.current_clue,
            current_count=self.current_count,
            remaining_guesses=self.remaining_guesses,
            team_remaining=self.team_remaining,
            opponent_remaining=self.opponent_remaining,
        )

    def _step_spymaster(
        self, action: SpymasterAction
    ) -> Tuple[Observation, float, bool, bool, dict]:
        ok, err = is_valid_clue(action.clue, self.board_words)
        if not ok:
            return (
                self._observation(),
                _REWARD_INVALID,
                False,
                False,
                {"invalid_clue": True, "error": err},
            )
        self.current_clue = action.clue
        self.current_count = action.count
        self.remaining_guesses = min(max(action.count, 1), self.max_guesses) + 1
        self.phase = GamePhase.GUESSER_TURN
        return self._observation(), 0.0, False, False, {"clue_given": action.clue}

    def _step_guesser(
        self, action: GuesserAction
    ) -> Tuple[Observation, float, bool, bool, dict]:
        # Pass / STOP.
        if action.word_index is None:
            self._end_turn()
            return self._observation(), 0.0, False, False, {"action": "pass"}

        idx = action.word_index
        if not isinstance(idx, int) or idx < 0 or idx >= _BOARD_SIZE:
            return (
                self._observation(),
                _REWARD_INVALID,
                False,
                False,
                {"error": f"invalid word_index: {idx}"},
            )
        if self.revealed_mask[idx]:
            return (
                self._observation(),
                _REWARD_INVALID,
                False,
                False,
                {"error": f"word at {idx} is already revealed"},
            )

        color = self.board_colors[idx]
        self.revealed_mask[idx] = True
        self.remaining_guesses -= 1

        if color == CardColor.TEAM:
            self.team_remaining -= 1
            info = {"correct": True, "color": color.value}
            if self.team_remaining == 0:
                self._terminate("win")
                return self._observation(), _REWARD_WIN, True, False, {**info, "result": "win"}
            # Allow another guess if budget remains; otherwise end turn.
            if self.remaining_guesses <= 0:
                self._end_turn()
            return self._observation(), _REWARD_TEAM, False, False, info

        if color == CardColor.ASSASSIN:
            self._terminate("loss_assassin")
            return (
                self._observation(),
                _REWARD_ASSASSIN,
                True,
                False,
                {"correct": False, "color": color.value, "result": "loss_assassin"},
            )

        if color == CardColor.OPPONENT:
            self.opponent_remaining -= 1
            self._end_turn()
            info = {"correct": False, "color": color.value}
            if self.opponent_remaining == 0:
                self._terminate("loss_opponent_won")
                info["result"] = "loss_opponent_won"
                return self._observation(), _REWARD_OPPONENT, True, False, info
            return self._observation(), _REWARD_OPPONENT, False, False, info

        # NEUTRAL.
        self._end_turn()
        return (
            self._observation(),
            _REWARD_NEUTRAL,
            False,
            False,
            {"correct": False, "color": color.value},
        )

    def _end_turn(self) -> None:
        self.phase = GamePhase.SPYMASTER_TURN
        self.current_clue = None
        self.current_count = None
        self.remaining_guesses = 0

    def _terminate(self, _outcome: str) -> None:
        self.phase = GamePhase.GAME_OVER
        self.game_over = True
        self.current_clue = None
        self.current_count = None
        self.remaining_guesses = 0
