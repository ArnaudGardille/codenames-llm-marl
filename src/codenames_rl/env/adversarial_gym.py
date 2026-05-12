"""Single-agent Gymnasium wrapper for adversarial training.

This is what TRL sees: a standard Gymnasium-style env where the trainable
team is ``team_a`` and ``team_b`` is driven by the supplied frozen callables
(``opponent_spymaster_policy``, ``opponent_guesser_policy``). The wrapper
runs the opponent's actions automatically between Team A's steps, so the
caller only needs to act when ``obs.phase == GamePhase.{SPYMASTER,GUESSER}_TURN``
and the current team is Team A.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .adversarial_core import CodenamesAdversarialCore
from .spaces import (
    GamePhase,
    GuesserAction,
    Observation,
    SpymasterAction,
)

OpponentPolicy = Callable[[Observation], object]


class CodenamesAdversarialGym:
    """Gymnasium-style wrapper around :class:`CodenamesAdversarialCore`."""

    def __init__(
        self,
        wordlist_path: str,
        opponent_spymaster_policy: OpponentPolicy,
        opponent_guesser_policy: OpponentPolicy,
        render_mode: Optional[str] = None,
    ) -> None:
        self.wordlist_path = wordlist_path
        self.opponent_spymaster_policy = opponent_spymaster_policy
        self.opponent_guesser_policy = opponent_guesser_policy
        self.render_mode = render_mode
        self.core: Optional[CodenamesAdversarialCore] = None
        self._episode_reward = 0.0

    # ------------------------------------------------------------- API

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, dict]:
        rng = np.random.default_rng(seed)
        self.core = CodenamesAdversarialCore(self.wordlist_path, rng)
        self._episode_reward = 0.0
        # Team A goes first by construction, so no opponent rollout needed here.
        return self._observation_for("team_a"), {}

    def step(self, action) -> Tuple[Observation, float, bool, bool, dict]:
        assert self.core is not None, "call reset() first"
        if self.core.game_over:
            raise RuntimeError("Game is over; call reset() before stepping again")
        if self.core.current_team != "team_a":
            raise RuntimeError(
                "Gym wrapper expects Team A to act; opponent rollout is handled "
                "internally."
            )

        if self.core.current_phase == "spymaster":
            if not isinstance(action, SpymasterAction):
                raise TypeError(
                    f"Expected SpymasterAction, got {type(action).__name__}"
                )
            reward, _done, info = self.core.execute_spymaster_action(
                "team_a", action.clue, action.count
            )
        else:
            if not isinstance(action, GuesserAction):
                raise TypeError(
                    f"Expected GuesserAction, got {type(action).__name__}"
                )
            reward, _done, info = self.core.execute_guesser_action(
                "team_a", action.word_index
            )

        self._episode_reward += reward

        # Roll the opponent forward until either the game ends or it is
        # Team A's turn again.
        while not self.core.game_over and self.core.current_team == "team_b":
            self._opponent_step()

        done = self.core.game_over
        if done:
            info = dict(info)
            info.setdefault("episode_reward", self._episode_reward)
            info.setdefault(
                "result",
                "win" if self.core.winner == "team_a" else "loss",
            )
        return self._observation_for("team_a"), reward, done, False, info

    def render(self) -> None:
        if self.render_mode != "human" or self.core is None:
            return
        rows = []
        for row in range(5):
            cells = []
            for col in range(5):
                i = row * 5 + col
                word = self.core.board_words[i]
                color = self.core.board_colors_absolute[i].value[0].upper()
                marker = "✓" if self.core.revealed_mask[i] else " "
                cells.append(f"{marker}{color} {word:<14}")
            rows.append(" | ".join(cells))
        print("\n".join(rows))
        print(
            f"turn={self.core.current_team}/{self.core.current_phase} "
            f"a_left={self.core.team_a_remaining} b_left={self.core.team_b_remaining}"
        )

    def close(self) -> None:
        pass

    # -------------------------------------------------------- internals

    def _observation_for(self, team: str) -> Observation:
        assert self.core is not None
        phase = (
            GamePhase.GAME_OVER
            if self.core.game_over
            else (
                GamePhase.SPYMASTER_TURN
                if self.core.current_phase == "spymaster"
                else GamePhase.GUESSER_TURN
            )
        )
        return Observation(
            board_words=list(self.core.board_words),
            board_colors=self.core.get_board_colors_for_team(team),
            revealed_mask=list(self.core.revealed_mask),
            phase=phase,
            current_clue=self.core.current_clue,
            current_count=self.core.current_count,
            remaining_guesses=self.core.remaining_guesses,
            team_remaining=self.core.remaining_for(team),
            opponent_remaining=self.core.remaining_for(_other(team)),
        )

    def _opponent_step(self) -> None:
        assert self.core is not None
        obs = self._observation_for("team_b")
        if self.core.current_phase == "spymaster":
            action = self.opponent_spymaster_policy(obs)
            self.core.execute_spymaster_action("team_b", action.clue, action.count)
        else:
            action = self.opponent_guesser_policy(obs)
            self.core.execute_guesser_action("team_b", action.word_index)


def _other(team: str) -> str:
    return "team_b" if team == "team_a" else "team_a"
