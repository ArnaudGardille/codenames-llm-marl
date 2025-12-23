#!/usr/bin/env python3
"""Compare baseline vs improved agents."""

import argparse
import json
from pathlib import Path

from codenames_rl.agents import (
    AdaptiveGuesser,
    ClusterSpymaster,
    ContextualGuesser,
    EmbeddingsGuesser,
    EmbeddingsSpymaster,
)
from codenames_rl.eval import compare_agents
from codenames_rl.utils.config import WORDLIST_PATH, VOCABULARY_PATH, NUM_GAMES, AGENT_SEED, START_SEED


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline embedding agents vs improved agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick comparison (20 games)
  python scripts/compare_improved.py

  # Full evaluation (100 games)
  python scripts/compare_improved.py --num-games 100

  # Save results to file
  python scripts/compare_improved.py --num-games 100 --output results/comparison.json

  # Use French wordlists
  python scripts/compare_improved.py \\
      --wordlist configs/wordlist_fr.txt \\
      --vocabulary configs/vocabulary_fr.txt
        """
    )
    
    parser.add_argument(
        "--wordlist",
        type=str,
        default=WORDLIST_PATH,
        help=f"Path to wordlist file (default: {WORDLIST_PATH})"
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        default=VOCABULARY_PATH,
        help=f"Path to vocabulary file (default: {VOCABULARY_PATH})"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help=f"Number of games per configuration (default: 50)"
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
    
    # Define agent configurations
    configs = {
        "Baseline (Embeddings/Embeddings)": {
            "spymaster": EmbeddingsSpymaster(
                vocabulary_path=args.vocabulary,
                seed=args.seed,
                top_k=100
            ),
            "guesser": EmbeddingsGuesser(seed=args.seed)
        },
        "Improved (Cluster/Contextual)": {
            "spymaster": ClusterSpymaster(
                vocabulary_path=args.vocabulary,
                seed=args.seed,
                top_k=100
            ),
            "guesser": ContextualGuesser(seed=args.seed)
        },
        "Improved (Cluster/Adaptive)": {
            "spymaster": ClusterSpymaster(
                vocabulary_path=args.vocabulary,
                seed=args.seed,
                top_k=100
            ),
            "guesser": AdaptiveGuesser(seed=args.seed)
        },
    }
    
    print("="*70)
    print("COMPARING BASELINE VS IMPROVED AGENTS")
    print("="*70)
    print(f"Wordlist: {args.wordlist}")
    print(f"Vocabulary: {args.vocabulary}")
    print(f"Games per config: {args.num_games}")
    print(f"Starting seed: {args.start_seed}")
    print()
    
    # Generate seeds for all games
    seeds = list(range(args.start_seed, args.start_seed + args.num_games))
    
    results = compare_agents(
        agent_configs=configs,
        wordlist_path=args.wordlist,
        num_games=args.num_games,
        seeds=seeds,
        verbose=args.verbose
    )
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Configuration':<35} {'Win Rate':<12} {'Avg Score':<12} {'Assassin%':<12} {'Cards/Turn':<12}")
    print("-"*70)
    
    for name, metrics in results.items():
        print(f"{name:<35} {metrics.win_rate:>10.1%}  {metrics.avg_score:>10.2f}  "
              f"{metrics.assassin_rate:>10.1%}  {metrics.avg_cards_per_turn:>10.2f}")
    
    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(metrics)
    
    # Save results if output specified
    if args.output:
        output_data = {
            "config": {
                "wordlist": args.wordlist,
                "vocabulary": args.vocabulary,
                "num_games": args.num_games,
                "start_seed": args.start_seed,
                "agent_seed": args.seed
            },
            "results": {
                name: metrics.to_dict()
                for name, metrics in results.items()
            }
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

