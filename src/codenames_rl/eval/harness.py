"""Evaluation harness for running and evaluating agent combinations."""

from __future__ import annotations

from typing import Dict, List, Optional

from tqdm import tqdm

from ..agents.baselines import BaseGuesser, BaseSpymaster
from ..env.core import CodenamesEnv
from ..env.spaces import CardColor, GamePhase, GuesserAction, SpymasterAction
from ..utils.config import MAX_GUESSES, MAX_TURNS
from .metrics import GameResult, EvaluationMetrics, compute_metrics


class EvaluationHarness:
    """Harness to evaluate Spymaster/Guesser combinations across multiple games."""

    def __init__(
        self,
        wordlist_path: str,
        spymaster: BaseSpymaster,
        guesser: BaseGuesser,
        max_guesses: Optional[int] = None,
        max_turns: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize the evaluation harness.
        
        Args:
            wordlist_path: Path to wordlist file for board generation
            spymaster: Spymaster agent
            guesser: Guesser agent
            max_guesses: Maximum guesses per turn (defaults to MAX_GUESSES from config)
            max_turns: Maximum turns before truncating game (defaults to MAX_TURNS from config)
            verbose: Print game progress
        """
        self.wordlist_path = wordlist_path
        self.spymaster = spymaster
        self.guesser = guesser
        self.max_guesses = max_guesses if max_guesses is not None else MAX_GUESSES
        self.max_turns = max_turns if max_turns is not None else MAX_TURNS
        self.verbose = verbose

    def run_episode(self, seed: int) -> GameResult:
        """Run a single game episode.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            GameResult with game statistics
        """
        env = CodenamesEnv(
            wordlist_path=self.wordlist_path,
            max_guesses=self.max_guesses,
            render_mode="human" if self.verbose else None
        )
        
        obs, info = env.reset(seed=seed)
        
        # Reset agents
        self.spymaster.reset()
        self.guesser.reset()
        
        # Track metrics
        total_turns = 0
        total_guesses = 0
        score = 0
        illegal_clues = 0
        clue_history = []
        cards_per_turn = []
        outcome = "truncated"
        
        # Track cards found in current turn
        cards_this_turn = 0
        
        terminated = False
        
        while not terminated and total_turns < self.max_turns:
            if self.verbose:
                env.render()
            
            if obs.phase == GamePhase.SPYMASTER_TURN:
                # Save cards from previous turn (if any)
                if total_turns > 0:
                    cards_per_turn.append(cards_this_turn)
                    cards_this_turn = 0
                
                if self.verbose:
                    # Show team words and danger words
                    team_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                                 if c == CardColor.TEAM and not r]
                    opponent_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                                     if c == CardColor.OPPONENT and not r]
                    assassin_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                                     if c == CardColor.ASSASSIN and not r]
                    print(f"\n[Turn {total_turns + 1}] Spymaster's turn")
                    print(f"  Team words to find: {', '.join(team_words)} ({len(team_words)} remaining)")
                    print(f"  Opponent words: {', '.join(opponent_words)} ({len(opponent_words)} remaining)")
                    if assassin_words:
                        print(f"  ⚠️  ASSASSIN: {', '.join(assassin_words)}")
                    print(f"  Score: {score}/9")
                
                # Spymaster gives clue
                action = self.spymaster.get_clue(obs)
                
                if self.verbose:
                    print(f"[Spymaster] Clue: '{action.clue}' (count: {action.count})")
                    # Show raw LLM output if available
                    if hasattr(self.spymaster, 'last_raw_output') and self.spymaster.last_raw_output:
                        print(f"  [LLM raw output]: {self.spymaster.last_raw_output[:200]}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track invalid clues
                if info.get("invalid_clue", False):
                    illegal_clues += 1
                    if self.verbose:
                        print(f"  ❌ [Invalid Clue] {info.get('error', 'Unknown error')}")
                else:
                    clue_history.append((action.clue, action.count))
                    total_turns += 1
                
            elif obs.phase == GamePhase.GUESSER_TURN:
                # Guesser makes guesses
                if self.verbose:
                    print(f"\n[Guesser's turn] Clue: '{obs.current_clue}' ({obs.current_count}), {obs.remaining_guesses} guesses remaining")
                    unrevealed = [w for w, r in zip(obs.board_words, obs.revealed_mask) if not r]
                    print(f"  Unrevealed words: {', '.join(unrevealed)}")
                
                while obs.phase == GamePhase.GUESSER_TURN and not terminated:
                    action = self.guesser.get_guess(obs)
                    
                    if action.word_index is None:
                        if self.verbose:
                            print(f"[Guesser] STOP (passing)")
                            if hasattr(self.guesser, 'last_raw_output') and self.guesser.last_raw_output:
                                print(f"  [LLM raw output]: {self.guesser.last_raw_output[:200]}")
                    else:
                        word = obs.board_words[action.word_index]
                        if self.verbose:
                            print(f"[Guesser] Guessing: '{word}'")
                            if hasattr(self.guesser, 'last_raw_output') and self.guesser.last_raw_output:
                                print(f"  [LLM raw output]: {self.guesser.last_raw_output[:200]}")
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_guesses += 1
                    
                    if info.get("correct", False) and info.get("color") == "team":
                        score += 1
                        cards_this_turn += 1
                        if self.verbose:
                            print(f"  ✓ Correct! Team card found ({score}/9 remaining)")
                    elif self.verbose and "color" in info:
                        color = info['color']
                        if color == "assassin":
                            print(f"  💀 ASSASSIN! Game lost!")
                        elif color == "opponent":
                            print(f"  ✗ Opponent card (they now have {obs.opponent_remaining} remaining)")
                        elif color == "neutral":
                            print(f"  ○ Neutral card")
                    
                    # Check outcome
                    if "result" in info:
                        outcome = info["result"]
                        if self.verbose:
                            print(f"\n[Game Over] Result: {outcome}")
                        
            else:
                # Game over
                break
        
        # Save cards from final turn
        if total_turns > 0:
            cards_per_turn.append(cards_this_turn)
        
        return GameResult(
            seed=seed,
            outcome=outcome,
            total_turns=total_turns,
            total_guesses=total_guesses,
            score=score,
            illegal_clues=illegal_clues,
            clue_history=clue_history,
            cards_per_turn=cards_per_turn
        )

    def evaluate(
        self,
        num_games: int,
        seeds: Optional[List[int]] = None,
        start_seed: int = 0
    ) -> EvaluationMetrics:
        """Evaluate agents over multiple games.
        
        Args:
            num_games: Number of games to run
            seeds: Optional list of specific seeds to use
            start_seed: Starting seed if seeds not provided
            
        Returns:
            EvaluationMetrics with aggregated statistics
        """
        if seeds is None:
            seeds = list(range(start_seed, start_seed + num_games))
        elif len(seeds) != num_games:
            raise ValueError(f"Number of seeds ({len(seeds)}) must match num_games ({num_games})")
        
        results = []
        
        # Use tqdm for progress bar (disable if verbose mode is on)
        progress_bar = tqdm(
            enumerate(seeds),
            total=num_games,
            desc="Evaluating",
            disable=self.verbose,
            unit="game"
        )
        
        for i, seed in progress_bar:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Game {i+1}/{num_games} (seed={seed})")
                print(f"{'='*60}")
            
            result = self.run_episode(seed)
            results.append(result)
            
            # Update progress bar with current stats
            if not self.verbose:
                current_metrics = compute_metrics(results)
                progress_bar.set_postfix({
                    'win_rate': f'{current_metrics.win_rate:.1%}',
                    'avg_score': f'{current_metrics.avg_score:.1f}'
                })
            
            if self.verbose:
                print(f"\nResult: {result.outcome}")
                print(f"Score: {result.score}/9")
                print(f"Turns: {result.total_turns}")
        
        metrics = compute_metrics(results)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("EVALUATION COMPLETE")
            print(f"{'='*60}")
            print(metrics)
        
        return metrics

    def evaluate_with_details(
        self,
        num_games: int,
        seeds: Optional[List[int]] = None,
        start_seed: int = 0
    ) -> tuple[EvaluationMetrics, List[GameResult]]:
        """Evaluate agents and return both metrics and detailed results.
        
        Args:
            num_games: Number of games to run
            seeds: Optional list of specific seeds to use
            start_seed: Starting seed if seeds not provided
            
        Returns:
            Tuple of (EvaluationMetrics, List[GameResult])
        """
        if seeds is None:
            seeds = list(range(start_seed, start_seed + num_games))
        elif len(seeds) != num_games:
            raise ValueError(f"Number of seeds ({len(seeds)}) must match num_games ({num_games})")
        
        results = []
        
        # Use tqdm for progress bar (disable if verbose mode is on)
        progress_bar = tqdm(
            enumerate(seeds),
            total=num_games,
            desc="Evaluating",
            disable=self.verbose,
            unit="game"
        )
        
        for i, seed in progress_bar:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Game {i+1}/{num_games} (seed={seed})")
                print(f"{'='*60}")
            
            result = self.run_episode(seed)
            results.append(result)
            
            # Update progress bar with current stats
            if not self.verbose:
                current_metrics = compute_metrics(results)
                progress_bar.set_postfix({
                    'win_rate': f'{current_metrics.win_rate:.1%}',
                    'avg_score': f'{current_metrics.avg_score:.1f}'
                })
            
            if self.verbose:
                print(f"\nResult: {result.outcome}")
                print(f"Score: {result.score}/9")
                print(f"Turns: {result.total_turns}")
        
        metrics = compute_metrics(results)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("EVALUATION COMPLETE")
            print(f"{'='*60}")
            print(metrics)
        
        return metrics, results


def compare_agents(
    agent_configs: Dict[str, Dict],
    wordlist_path: str,
    num_games: int = 100,
    seeds: Optional[List[int]] = None,
    verbose: bool = False
) -> Dict[str, EvaluationMetrics]:
    """Compare multiple agent configurations.
    
    Args:
        agent_configs: Dict mapping name -> {"spymaster": agent, "guesser": agent}
        wordlist_path: Path to wordlist
        num_games: Number of games per configuration
        seeds: Optional list of seeds (same seeds used for all configs)
        verbose: Print progress
        
    Returns:
        Dict mapping configuration name -> EvaluationMetrics
    """
    results = {}
    
    for name, config in agent_configs.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating: {name}")
            print(f"{'='*70}")
        
        harness = EvaluationHarness(
            wordlist_path=wordlist_path,
            spymaster=config["spymaster"],
            guesser=config["guesser"],
            verbose=verbose
        )
        
        metrics = harness.evaluate(num_games=num_games, seeds=seeds)
        results[name] = metrics
        
        if not verbose:
            print(f"{name}: {metrics.win_rate:.1%} win rate")
    
    return results

