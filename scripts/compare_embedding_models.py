#!/usr/bin/env python3
"""Compare different embedding models for Codenames agents."""

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
        description="Compare different embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all-mpnet-base-v2 vs paraphrase-multilingual-mpnet-base-v2
  python scripts/compare_embedding_models.py

  # Full evaluation (100 games)
  python scripts/compare_embedding_models.py --num-games 100

  # Save results to file
  python scripts/compare_embedding_models.py --num-games 100 --output results/embedding_comparison.json
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
    
    # Define the two models to compare
    model1 = "all-mpnet-base-v2"  # Will become sentence-transformers/all-mpnet-base-v2
    model2 = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Define agent configurations for each model
    configs = {}
    
    # Model 1: all-mpnet-base-v2
    configs[f"all-mpnet-base-v2 (Baseline)"] = {
        "spymaster": EmbeddingsSpymaster(
            vocabulary_path=args.vocabulary,
            model_name=f"sentence-transformers/{model1}",
            seed=args.seed,
            top_k=100
        ),
        "guesser": EmbeddingsGuesser(
            model_name=f"sentence-transformers/{model1}",
            seed=args.seed
        )
    }
    
    configs[f"all-mpnet-base-v2 (Improved)"] = {
        "spymaster": ClusterSpymaster(
            vocabulary_path=args.vocabulary,
            model_name=f"sentence-transformers/{model1}",
            seed=args.seed,
            top_k=100
        ),
        "guesser": ContextualGuesser(
            model_name=f"sentence-transformers/{model1}",
            seed=args.seed
        )
    }
    
    # Model 2: paraphrase-multilingual-mpnet-base-v2
    configs[f"paraphrase-multilingual-mpnet-base-v2 (Baseline)"] = {
        "spymaster": EmbeddingsSpymaster(
            vocabulary_path=args.vocabulary,
            model_name=model2,
            seed=args.seed,
            top_k=100
        ),
        "guesser": EmbeddingsGuesser(
            model_name=model2,
            seed=args.seed
        )
    }
    
    configs[f"paraphrase-multilingual-mpnet-base-v2 (Improved)"] = {
        "spymaster": ClusterSpymaster(
            vocabulary_path=args.vocabulary,
            model_name=model2,
            seed=args.seed,
            top_k=100
        ),
        "guesser": ContextualGuesser(
            model_name=model2,
            seed=args.seed
        )
    }
    
    print("="*70)
    print("COMPARING EMBEDDING MODELS")
    print("="*70)
    print(f"Model 1: sentence-transformers/{model1}")
    print(f"Model 2: {model2}")
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
    print(f"{'Configuration':<50} {'Win Rate':<12} {'Avg Score':<12} {'Assassin%':<12} {'Cards/Turn':<12}")
    print("-"*70)
    
    for name, metrics in results.items():
        print(f"{name:<50} {metrics.win_rate:>10.1%}  {metrics.avg_score:>10.2f}  "
              f"{metrics.assassin_rate:>10.1%}  {metrics.avg_cards_per_turn:>10.2f}")
    
    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(metrics)
    
    # Print summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    model1_baseline = results.get(f"all-mpnet-base-v2 (Baseline)")
    model1_improved = results.get(f"all-mpnet-base-v2 (Improved)")
    model2_baseline = results.get(f"paraphrase-multilingual-mpnet-base-v2 (Baseline)")
    model2_improved = results.get(f"paraphrase-multilingual-mpnet-base-v2 (Improved)")
    
    if model1_baseline and model2_baseline:
        print(f"\nBaseline Comparison:")
        print(f"  all-mpnet-base-v2:           {model1_baseline.win_rate:.1%} win rate, {model1_baseline.avg_score:.2f} avg score")
        print(f"  paraphrase-multilingual:     {model2_baseline.win_rate:.1%} win rate, {model2_baseline.avg_score:.2f} avg score")
        win_diff = model2_baseline.win_rate - model1_baseline.win_rate
        score_diff = model2_baseline.avg_score - model1_baseline.avg_score
        print(f"  Difference:                  {win_diff:+.1%} win rate, {score_diff:+.2f} avg score")
    
    if model1_improved and model2_improved:
        print(f"\nImproved Comparison:")
        print(f"  all-mpnet-base-v2:           {model1_improved.win_rate:.1%} win rate, {model1_improved.avg_score:.2f} avg score")
        print(f"  paraphrase-multilingual:     {model2_improved.win_rate:.1%} win rate, {model2_improved.avg_score:.2f} avg score")
        win_diff = model2_improved.win_rate - model1_improved.win_rate
        score_diff = model2_improved.avg_score - model1_improved.avg_score
        print(f"  Difference:                  {win_diff:+.1%} win rate, {score_diff:+.2f} avg score")
    
    # Save results if output specified
    if args.output:
        output_data = {
            "config": {
                "model1": f"sentence-transformers/{model1}",
                "model2": model2,
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

