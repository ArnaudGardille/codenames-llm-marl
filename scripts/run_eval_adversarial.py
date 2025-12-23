#!/usr/bin/env python3
"""CLI script for running adversarial (4-player) agent evaluations.

This script evaluates agents in the competitive 4-player adversarial mode
where two teams (Red and Blue), each with a Spymaster and Guesser, compete
to find all their words first.
"""

import argparse
import json
from pathlib import Path

from codenames_rl.agents import (
    EmbeddingsGuesser,
    EmbeddingsSpymaster,
    LLMGuesser,
    LLMSpymaster,
    QwenEmbeddingGuesser,
    QwenEmbeddingSpymaster,
    RandomGuesser,
    RandomSpymaster,
)
from codenames_rl.utils.config import (
    NUM_GAMES,
    START_SEED,
    AGENT_SEED,
    WORDLIST_PATH,
    VOCABULARY_PATH,
)


def create_agent(agent_type: str, vocabulary_path: str = None, seed: int = None):
    """Create an agent based on type string.
    
    Args:
        agent_type: One of "random", "embeddings", "qwen_embedding", "llm"
        vocabulary_path: Path to vocabulary file (for spymasters)
        seed: Random seed
        
    Returns:
        Agent instance
    """
    if agent_type == "random_spymaster":
        if vocabulary_path is None:
            raise ValueError("vocabulary_path required for random spymaster")
        return RandomSpymaster(vocabulary_path=vocabulary_path, seed=seed)
    elif agent_type == "embeddings_spymaster":
        if vocabulary_path is None:
            raise ValueError("vocabulary_path required for embeddings spymaster")
        return EmbeddingsSpymaster(vocabulary_path=vocabulary_path, seed=seed)
    elif agent_type == "qwen_embedding_spymaster":
        if vocabulary_path is None:
            raise ValueError("vocabulary_path required for qwen_embedding spymaster")
        return QwenEmbeddingSpymaster(vocabulary_path=vocabulary_path, seed=seed)
    elif agent_type == "llm_spymaster":
        return LLMSpymaster(seed=seed)
    elif agent_type == "random_guesser":
        return RandomGuesser(seed=seed)
    elif agent_type == "embeddings_guesser":
        return EmbeddingsGuesser(seed=seed)
    elif agent_type == "qwen_embedding_guesser":
        return QwenEmbeddingGuesser(seed=seed)
    elif agent_type == "llm_guesser":
        return LLMGuesser(seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_adversarial_game(
    red_spymaster,
    red_guesser,
    blue_spymaster,
    blue_guesser,
    wordlist_path: str,
    seed: int,
    verbose: bool = False
):
    """Run a single adversarial game between two teams.
    
    Args:
        red_spymaster: Red team spymaster agent
        red_guesser: Red team guesser agent
        blue_spymaster: Blue team spymaster agent
        blue_guesser: Blue team guesser agent
        wordlist_path: Path to wordlist file
        seed: Random seed for game
        verbose: Print detailed game progress
        
    Returns:
        Dictionary with game results
    """
    try:
        from codenames_rl.env.adversarial_pz import env as make_env
    except ImportError:
        raise NotImplementedError(
            "Adversarial environment not yet implemented. "
            "Please implement CodenamesAdversarialPZ first."
        )
    
    env = make_env(wordlist_path=wordlist_path)
    env.reset(seed=seed)
    
    agents = {
        'red_spymaster': red_spymaster,
        'red_guesser': red_guesser,
        'blue_spymaster': blue_spymaster,
        'blue_guesser': blue_guesser,
    }
    
    game_log = []
    turns = 0
    
    for agent_id in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        
        if terminated or truncated:
            break
        
        agent = agents[agent_id]
        
        # Get action based on agent role
        if 'spymaster' in agent_id:
            action = agent.get_clue(obs)
        else:
            action = agent.get_guess(obs)
        
        if verbose:
            print(f"[Turn {turns}] {agent_id}: {action}")
        
        game_log.append({
            'turn': turns,
            'agent': agent_id,
            'action': str(action),
            'reward': reward
        })
        
        env.step(action)
        turns += 1
    
    # Get final results
    winner = info.get('winner', None)  # 'red' or 'blue'
    red_score = info.get('red_score', 0)
    blue_score = info.get('blue_score', 0)
    
    return {
        'seed': seed,
        'winner': winner,
        'red_score': red_score,
        'blue_score': blue_score,
        'turns': turns,
        'game_log': game_log if verbose else None
    }


def evaluate_adversarial(
    red_spymaster,
    red_guesser,
    blue_spymaster,
    blue_guesser,
    wordlist_path: str,
    num_games: int,
    start_seed: int,
    verbose: bool = False
):
    """Evaluate teams over multiple games.
    
    Args:
        red_spymaster: Red team spymaster agent
        red_guesser: Red team guesser agent
        blue_spymaster: Blue team spymaster agent
        blue_guesser: Blue team guesser agent
        wordlist_path: Path to wordlist file
        num_games: Number of games to run
        start_seed: Starting seed
        verbose: Print detailed progress
        
    Returns:
        Tuple of (metrics_dict, game_results_list)
    """
    results = []
    red_wins = 0
    blue_wins = 0
    
    for i in range(num_games):
        seed = start_seed + i
        
        if verbose or (i + 1) % 10 == 0:
            print(f"Game {i+1}/{num_games}...")
        
        result = run_adversarial_game(
            red_spymaster=red_spymaster,
            red_guesser=red_guesser,
            blue_spymaster=blue_spymaster,
            blue_guesser=blue_guesser,
            wordlist_path=wordlist_path,
            seed=seed,
            verbose=verbose
        )
        
        results.append(result)
        
        if result['winner'] == 'red':
            red_wins += 1
        elif result['winner'] == 'blue':
            blue_wins += 1
    
    metrics = {
        'num_games': num_games,
        'red_wins': red_wins,
        'blue_wins': blue_wins,
        'red_win_rate': red_wins / num_games if num_games > 0 else 0,
        'blue_win_rate': blue_wins / num_games if num_games > 0 else 0,
        'avg_red_score': sum(r['red_score'] for r in results) / num_games if num_games > 0 else 0,
        'avg_blue_score': sum(r['blue_score'] for r in results) / num_games if num_games > 0 else 0,
        'avg_turns': sum(r['turns'] for r in results) / num_games if num_games > 0 else 0,
    }
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Codenames agents in adversarial (4-player) mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LLM vs Embeddings teams
  python scripts/run_eval_adversarial.py \\
      --wordlist configs/wordlist_en.txt \\
      --vocabulary configs/vocabulary_en.txt \\
      --red-spymaster llm \\
      --red-guesser llm \\
      --blue-spymaster embeddings \\
      --blue-guesser embeddings \\
      --num-games 50

  # Random vs LLM teams
  python scripts/run_eval_adversarial.py \\
      --wordlist configs/wordlist_en.txt \\
      --vocabulary configs/vocabulary_en.txt \\
      --red-spymaster random \\
      --red-guesser random \\
      --blue-spymaster llm \\
      --blue-guesser llm \\
      --num-games 100 \\
      --output results/random_vs_llm.json
        """
    )
    
    parser.add_argument(
        "--wordlist",
        type=str,
        default=WORDLIST_PATH,
        help=f"Path to wordlist file for board generation (default: {WORDLIST_PATH})"
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        default=VOCABULARY_PATH,
        help=f"Path to vocabulary file for clue generation (default: {VOCABULARY_PATH})"
    )
    
    # Red team agents
    parser.add_argument(
        "--red-spymaster",
        type=str,
        choices=["random", "embeddings", "qwen_embedding", "llm"],
        default="embeddings",
        help="Red team spymaster agent type"
    )
    parser.add_argument(
        "--red-guesser",
        type=str,
        choices=["random", "embeddings", "qwen_embedding", "llm"],
        default="embeddings",
        help="Red team guesser agent type"
    )
    
    # Blue team agents
    parser.add_argument(
        "--blue-spymaster",
        type=str,
        choices=["random", "embeddings", "qwen_embedding", "llm"],
        default="embeddings",
        help="Blue team spymaster agent type"
    )
    parser.add_argument(
        "--blue-guesser",
        type=str,
        choices=["random", "embeddings", "qwen_embedding", "llm"],
        default="embeddings",
        help="Blue team guesser agent type"
    )
    
    parser.add_argument(
        "--num-games",
        type=int,
        default=NUM_GAMES,
        help=f"Number of games to run (default: {NUM_GAMES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=AGENT_SEED,
        help=f"Random seed for agent initialization (default: {AGENT_SEED})"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=START_SEED,
        help=f"Starting seed for game episodes (default: {START_SEED})"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game progress"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("ADVERSARIAL MODE EVALUATION")
    print("="*70)
    print(f"\nRed Team:  {args.red_spymaster.title()} Spymaster + {args.red_guesser.title()} Guesser")
    print(f"Blue Team: {args.blue_spymaster.title()} Spymaster + {args.blue_guesser.title()} Guesser")
    print(f"\nWordlist: {args.wordlist}")
    if args.vocabulary:
        print(f"Vocabulary: {args.vocabulary}")
    print(f"Games: {args.num_games}")
    print(f"Starting seed: {args.start_seed}\n")
    
    # Create Red team agents
    red_spymaster = create_agent(
        f"{args.red_spymaster}_spymaster",
        vocabulary_path=args.vocabulary,
        seed=args.seed
    )
    # Share model if both red agents are LLM or qwen_embedding to save memory
    if args.red_spymaster == "llm" and args.red_guesser == "llm":
        red_guesser = LLMGuesser(
            model=red_spymaster.model,
            tokenizer=red_spymaster.tokenizer,
            device=red_spymaster.device,
            seed=args.seed
        )
    elif args.red_spymaster == "qwen_embedding" and args.red_guesser == "qwen_embedding":
        red_guesser = QwenEmbeddingGuesser(
            model=red_spymaster.model,
            tokenizer=red_spymaster.tokenizer,
            device=red_spymaster.device,
            seed=args.seed
        )
    else:
        red_guesser = create_agent(
            f"{args.red_guesser}_guesser",
            seed=args.seed
        )
    
    # Create Blue team agents
    blue_spymaster = create_agent(
        f"{args.blue_spymaster}_spymaster",
        vocabulary_path=args.vocabulary,
        seed=args.seed + 1000  # Different seed for opponent
    )
    # Share model if both blue agents are LLM or qwen_embedding to save memory
    if args.blue_spymaster == "llm" and args.blue_guesser == "llm":
        blue_guesser = LLMGuesser(
            model=blue_spymaster.model,
            tokenizer=blue_spymaster.tokenizer,
            device=blue_spymaster.device,
            seed=args.seed + 1000
        )
    elif args.blue_spymaster == "qwen_embedding" and args.blue_guesser == "qwen_embedding":
        blue_guesser = QwenEmbeddingGuesser(
            model=blue_spymaster.model,
            tokenizer=blue_spymaster.tokenizer,
            device=blue_spymaster.device,
            seed=args.seed + 1000
        )
    else:
        blue_guesser = create_agent(
            f"{args.blue_guesser}_guesser",
            seed=args.seed + 1000
        )
    
    # Run evaluation
    try:
        metrics, results = evaluate_adversarial(
            red_spymaster=red_spymaster,
            red_guesser=red_guesser,
            blue_spymaster=blue_spymaster,
            blue_guesser=blue_guesser,
            wordlist_path=args.wordlist,
            num_games=args.num_games,
            start_seed=args.start_seed,
            verbose=args.verbose
        )
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Total Games:       {metrics['num_games']}")
        print(f"\nRed Team Wins:     {metrics['red_wins']} ({metrics['red_win_rate']:.1%})")
        print(f"Blue Team Wins:    {metrics['blue_wins']} ({metrics['blue_win_rate']:.1%})")
        print(f"\nAvg Red Score:     {metrics['avg_red_score']:.2f}/9")
        print(f"Avg Blue Score:    {metrics['avg_blue_score']:.2f}/8")
        print(f"\nAvg Game Length:   {metrics['avg_turns']:.1f} turns")
        print("="*70)
        
        # Save results if output specified
        if args.output:
            output_data = {
                "config": {
                    "red_spymaster": args.red_spymaster,
                    "red_guesser": args.red_guesser,
                    "blue_spymaster": args.blue_spymaster,
                    "blue_guesser": args.blue_guesser,
                    "wordlist": args.wordlist,
                    "vocabulary": args.vocabulary,
                    "num_games": args.num_games,
                    "start_seed": args.start_seed,
                    "agent_seed": args.seed
                },
                "metrics": metrics,
                "games": results
            }
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to: {args.output}")
    
    except NotImplementedError as e:
        print(f"\nERROR: {e}")
        print("\nTo use adversarial mode, you need to implement:")
        print("1. src/codenames_rl/env/adversarial_core.py")
        print("2. src/codenames_rl/env/adversarial_pz.py")
        print("3. src/codenames_rl/env/adversarial_gym.py")
        print("\nSee README.md for design details.")
        return 1


if __name__ == "__main__":
    exit(main() or 0)

