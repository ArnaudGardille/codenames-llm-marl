"""PettingZoo AEC wrapper around :class:`CodenamesAdversarialCore`.

Four agents — ``team_a_spymaster``, ``team_a_guesser``, ``team_b_spymaster``,
``team_b_guesser`` — alternate in the order dictated by the core's state
machine. ``observe(agent)`` returns the same :class:`Observation` dataclass
the cooperative env uses, with colors flipped per team.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np

from .adversarial_core import CodenamesAdversarialCore
from .spaces import (
    GamePhase,
    GuesserAction,
    Observation,
    SpymasterAction,
)

_AGENT_NAMES = (
    "team_a_spymaster",
    "team_a_guesser",
    "team_b_spymaster",
    "team_b_guesser",
)


class CodenamesAdversarialPZ:
    """PettingZoo-style AEC env (subset of the API actually used here).

    We don't depend on PettingZoo at runtime — only on the conventional
    surface it exposes (``agents``, ``agent_selection``, ``observe``,
    ``step``, ``last``, ``agent_iter``, ``reset``). Keeping it dependency-
    light makes the test suite cheap.
    """

    def __init__(self, wordlist_path: str) -> None:
        self.wordlist_path = wordlist_path
        self.core: Optional[CodenamesAdversarialCore] = None
        self.agents: List[str] = list(_AGENT_NAMES)
        self.agent_selection: Optional[str] = None
        self.rewards: Dict[str, float] = {a: 0.0 for a in self.agents}
        self.terminations: Dict[str, bool] = {a: False for a in self.agents}
        self.truncations: Dict[str, bool] = {a: False for a in self.agents}
        self.infos: Dict[str, dict] = {a: {} for a in self.agents}
        self._last_reward = 0.0
        self._last_info: dict = {}

    # ------------------------------------------------------------- API

    def reset(self, seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        self.core = CodenamesAdversarialCore(self.wordlist_path, rng)
        self.agents = list(_AGENT_NAMES)
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self._active_agent()
        self._last_reward = 0.0
        self._last_info = {}

    def observe(self, agent: str) -> Observation:
        assert self.core is not None, "call reset() first"
        team, _role = _split_agent(agent)
        colors = self.core.get_board_colors_for_team(team)
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
            board_colors=colors,
            revealed_mask=list(self.core.revealed_mask),
            phase=phase,
            current_clue=self.core.current_clue,
            current_count=self.core.current_count,
            remaining_guesses=self.core.remaining_guesses,
            team_remaining=self.core.remaining_for(team),
            opponent_remaining=self.core.remaining_for(_other(team)),
        )

    def last(self):
        agent = self.agent_selection
        obs = self.observe(agent) if agent is not None and self.core is not None else None
        terminated = self.core.game_over if self.core is not None else True
        truncated = False
        return obs, self._last_reward, terminated, truncated, dict(self._last_info)

    def step(self, action) -> None:
        assert self.core is not None, "call reset() first"
        if self.core.game_over:
            raise RuntimeError("Game is over; call reset() before stepping again")

        agent = self.agent_selection
        assert agent is not None
        team, role = _split_agent(agent)

        if role == "spymaster":
            if not isinstance(action, SpymasterAction):
                raise TypeError(
                    f"Expected SpymasterAction for {agent}, got {type(action).__name__}"
                )
            reward, _done, info = self.core.execute_spymaster_action(
                team, action.clue, action.count
            )
        else:
            if not isinstance(action, GuesserAction):
                raise TypeError(
                    f"Expected GuesserAction for {agent}, got {type(action).__name__}"
                )
            reward, _done, info = self.core.execute_guesser_action(
                team, action.word_index
            )

        self.rewards[agent] = reward
        self.infos[agent] = info
        self._last_reward = reward
        self._last_info = info

        if self.core.game_over:
            for a in self.agents:
                self.terminations[a] = True
            self.agent_selection = None
        else:
            self.agent_selection = self._active_agent()

    def agent_iter(self, max_iter: int = 10**6) -> Iterator[str]:
        """Yield the active agent until the game ends or ``max_iter`` is hit."""
        n = 0
        while (
            self.agent_selection is not None
            and self.core is not None
            and not self.core.game_over
            and n < max_iter
        ):
            yield self.agent_selection
            n += 1

    def close(self) -> None:
        pass

    # -------------------------------------------------------- internals

    def _active_agent(self) -> Optional[str]:
        if self.core is None or self.core.game_over:
            return None
        return f"{self.core.current_team}_{self.core.current_phase}"


def env(wordlist_path: str) -> CodenamesAdversarialPZ:
    """Factory matching the PettingZoo convention (``env()`` constructor)."""
    return CodenamesAdversarialPZ(wordlist_path)


def _split_agent(agent: str):
    """``"team_a_spymaster"`` → ``("team_a", "spymaster")``."""
    if agent.startswith("team_a_"):
        return "team_a", agent[len("team_a_"):]
    if agent.startswith("team_b_"):
        return "team_b", agent[len("team_b_"):]
    raise ValueError(f"Unknown agent name: {agent!r}")


def _other(team: str) -> str:
    return "team_b" if team == "team_a" else "team_a"
