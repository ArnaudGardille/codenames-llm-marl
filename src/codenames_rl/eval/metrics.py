"""Metrics and data structures for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GameResult:
    """Result of a single game episode."""
    
    seed: int
    outcome: str  # "win", "loss_assassin", "loss_opponent_won"
    total_turns: int  # Number of spymaster turns (clues given)
    total_guesses: int  # Total guesses made
    score: int  # Team cards found
    illegal_clues: int  # Number of invalid clues attempted
    clue_history: List[Tuple[str, int]] = field(default_factory=list)  # (clue, count) pairs
    cards_per_turn: List[int] = field(default_factory=list)  # Cards found per turn
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "seed": self.seed,
            "outcome": self.outcome,
            "total_turns": self.total_turns,
            "total_guesses": self.total_guesses,
            "score": self.score,
            "illegal_clues": self.illegal_clues,
            "clue_history": self.clue_history,
            "cards_per_turn": self.cards_per_turn
        }


@dataclass
class EvaluationMetrics:
    """Aggregated metrics across multiple games."""
    
    num_games: int
    win_rate: float
    avg_score: float
    avg_turns: float
    avg_cards_per_turn: float
    std_cards_per_turn: float
    illegal_clue_rate: float
    assassin_rate: float
    opponent_win_rate: float
    truncation_rate: float
    avg_guesses: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "num_games": self.num_games,
            "win_rate": self.win_rate,
            "avg_score": self.avg_score,
            "avg_turns": self.avg_turns,
            "avg_cards_per_turn": self.avg_cards_per_turn,
            "std_cards_per_turn": self.std_cards_per_turn,
            "illegal_clue_rate": self.illegal_clue_rate,
            "assassin_rate": self.assassin_rate,
            "opponent_win_rate": self.opponent_win_rate,
            "truncation_rate": self.truncation_rate,
            "avg_guesses": self.avg_guesses
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return f"""
Evaluation Results ({self.num_games} games):
  Win Rate:            {self.win_rate:.1%}
  Avg Score:           {self.avg_score:.2f} / 9 cards
  Avg Turns:           {self.avg_turns:.2f}
  Avg Cards/Turn:      {self.avg_cards_per_turn:.2f} ± {self.std_cards_per_turn:.2f}
  Avg Guesses:         {self.avg_guesses:.2f}
  Illegal Clue Rate:   {self.illegal_clue_rate:.1%}
  Assassin Rate:       {self.assassin_rate:.1%}
  Opponent Win Rate:   {self.opponent_win_rate:.1%}
  Truncation Rate:     {self.truncation_rate:.1%}
        """.strip()


def compute_metrics(results: List[GameResult]) -> EvaluationMetrics:
    """Compute aggregated metrics from game results.
    
    Args:
        results: List of GameResult objects
        
    Returns:
        EvaluationMetrics with aggregated statistics
    """
    if not results:
        return EvaluationMetrics(
            num_games=0,
            win_rate=0.0,
            avg_score=0.0,
            avg_turns=0.0,
            avg_cards_per_turn=0.0,
            std_cards_per_turn=0.0,
            illegal_clue_rate=0.0,
            assassin_rate=0.0,
            opponent_win_rate=0.0,
            truncation_rate=0.0,
            avg_guesses=0.0
        )
    
    num_games = len(results)
    wins = sum(1 for r in results if r.outcome == "win")
    assassin_losses = sum(1 for r in results if r.outcome == "loss_assassin")
    opponent_wins = sum(1 for r in results if r.outcome == "loss_opponent_won")
    truncated = sum(1 for r in results if r.outcome == "truncated")
    total_illegal = sum(r.illegal_clues for r in results)
    
    # Compute cards per turn statistics
    all_cards_per_turn = [card_count for r in results for card_count in r.cards_per_turn]
    avg_cards_per_turn = sum(all_cards_per_turn) / len(all_cards_per_turn) if all_cards_per_turn else 0.0
    
    # Compute standard deviation
    if len(all_cards_per_turn) > 1:
        variance = sum((x - avg_cards_per_turn) ** 2 for x in all_cards_per_turn) / len(all_cards_per_turn)
        std_cards_per_turn = variance ** 0.5
    else:
        std_cards_per_turn = 0.0
    
    return EvaluationMetrics(
        num_games=num_games,
        win_rate=wins / num_games,
        avg_score=sum(r.score for r in results) / num_games,
        avg_turns=sum(r.total_turns for r in results) / num_games,
        avg_cards_per_turn=avg_cards_per_turn,
        std_cards_per_turn=std_cards_per_turn,
        illegal_clue_rate=total_illegal / sum(r.total_turns for r in results) if sum(r.total_turns for r in results) > 0 else 0.0,
        assassin_rate=assassin_losses / num_games,
        opponent_win_rate=opponent_wins / num_games,
        truncation_rate=truncated / num_games,
        avg_guesses=sum(r.total_guesses for r in results) / num_games
    )

