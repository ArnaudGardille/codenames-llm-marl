#!/usr/bin/env python3
"""CLI script for running agent evaluations."""

import argparse
import json
from pathlib import Path

from codenames_rl.agents import (
    AGENT_KINDS,
    EmbeddingsGuesser,
    LLMGuesser,
    QwenEmbeddingGuesser,
    create_agent,
)
from codenames_rl.eval import EvaluationHarness
from codenames_rl.utils.config import (
    MAX_GUESSES,
    MAX_TURNS,
    NUM_GAMES,
    START_SEED,
    AGENT_SEED,
    WORDLIST_PATH,
    VOCABULARY_PATH,
    get_language_paths,
)

# Pairings used by --compare. Each entry is (spymaster_kind, guesser_kind).
COMPARE_PAIRS = [
    ("random", "random"),
    ("random", "embeddings"),
    ("embeddings", "random"),
    ("embeddings", "embeddings"),
    ("qwen_embedding", "qwen_embedding"),
    ("qwen_embedding", "embeddings"),
    ("embeddings", "qwen_embedding"),
    ("llm", "llm"),
    ("llm", "embeddings"),
    ("embeddings", "llm"),
    ("llm", "random"),
    ("random", "llm"),
]


def _build_pair(spy_kind: str, guesser_kind: str, vocabulary_path: str, seed: int):
    """Build a (spymaster, guesser) pair, sharing the model when both are LLM/Qwen."""
    spymaster = create_agent(f"{spy_kind}_spymaster", vocabulary_path=vocabulary_path, seed=seed)
    if spy_kind == guesser_kind == "llm":
        guesser = LLMGuesser(model=spymaster.model, tokenizer=spymaster.tokenizer,
                             device=spymaster.device, seed=seed)
    elif spy_kind == guesser_kind == "qwen_embedding":
        guesser = QwenEmbeddingGuesser(model=spymaster.model, tokenizer=spymaster.tokenizer,
                                       device=spymaster.device, seed=seed)
    else:
        guesser = create_agent(f"{guesser_kind}_guesser", seed=seed)
    return spymaster, guesser


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Codenames agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate embeddings baseline (EN)
  python scripts/run_eval.py \\
      --lang en \\
      --spymaster embeddings \\
      --guesser embeddings \\
      --num-games 100 \\
      --output results/baseline_embeddings_en.json

  # Evaluate random baseline (FR)
  python scripts/run_eval.py \\
      --lang fr \\
      --spymaster random \\
      --guesser random \\
      --num-games 50 \\
      --verbose

  # Compare configurations
  python scripts/run_eval.py \\
      --lang en \\
      --compare
        """
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "fr"],
        default=None,
        help="Language code (en/fr). Sets both wordlist and vocabulary paths."
    )
    parser.add_argument(
        "--wordlist",
        type=str,
        default=None,
        help=f"Path to wordlist file for board generation (default: {WORDLIST_PATH}, or from --lang)"
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        default=None,
        help=f"Path to vocabulary file for clue generation (default: {VOCABULARY_PATH}, or from --lang)"
    )
    parser.add_argument(
        "--spymaster",
        type=str,
        choices=AGENT_KINDS,
        default="embeddings",
        help="Spymaster agent type",
    )
    parser.add_argument(
        "--guesser",
        type=str,
        choices=AGENT_KINDS,
        default="embeddings",
        help="Guesser agent type",
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
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"Maximum turns before truncating game (default: {MAX_TURNS})"
    )
    parser.add_argument(
        "--max-guesses",
        type=int,
        default=MAX_GUESSES,
        help=f"Maximum guesses per turn (default: {MAX_GUESSES})"
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all baseline combinations"
    )
    
    args = parser.parse_args()
    
    # Resolve wordlist and vocabulary paths
    # Priority: explicit paths > --lang > defaults
    if args.lang:
        lang_wordlist, lang_vocabulary = get_language_paths(args.lang)
        wordlist_path = args.wordlist if args.wordlist else lang_wordlist
        vocabulary_path = args.vocabulary if args.vocabulary else lang_vocabulary
    else:
        wordlist_path = args.wordlist if args.wordlist else WORDLIST_PATH
        vocabulary_path = args.vocabulary if args.vocabulary else VOCABULARY_PATH
    
    # Validate vocabulary path for non-random, non-llm spymasters
    if args.spymaster not in ["random", "llm"] and not vocabulary_path:
        parser.error("--vocabulary or --lang required for non-random, non-llm spymaster")
    
    if args.compare:
        print("Comparing baseline agent combinations...")
        print(f"Wordlist: {wordlist_path}")
        print(f"Vocabulary: {vocabulary_path}")
        print(f"Games per config: {args.num_games}")
        print(f"Starting seed: {args.start_seed}\n")

        configs = {}
        for spy_kind, guesser_kind in COMPARE_PAIRS:
            spymaster, guesser = _build_pair(spy_kind, guesser_kind, vocabulary_path, args.seed)
            label = f"{spy_kind}/{guesser_kind}"
            configs[label] = {"spymaster": spymaster, "guesser": guesser}

        from codenames_rl.eval import compare_agents

        results = compare_agents(
            agent_configs=configs,
            wordlist_path=wordlist_path,
            num_games=args.num_games,
            seeds=list(range(args.start_seed, args.start_seed + args.num_games)),
            verbose=args.verbose
        )
        
        # Print comparison table
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"{'Configuration':<25} {'Win Rate':<12} {'Avg Score':<12} {'Assassin%':<12}")
        print("-"*70)
        
        for name, metrics in results.items():
            print(f"{name:<25} {metrics.win_rate:>10.1%}  {metrics.avg_score:>10.2f}  {metrics.assassin_rate:>10.1%}")
        
        # Save all results if output specified
        if args.output:
            output_data = {
                name: metrics.to_dict()
                for name, metrics in results.items()
            }
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    else:
        print(f"Evaluating: {args.spymaster.title()} Spymaster + {args.guesser.title()} Guesser")
        print(f"Wordlist: {wordlist_path}")
        if vocabulary_path:
            print(f"Vocabulary: {vocabulary_path}")
        print(f"Games: {args.num_games}")
        print(f"Starting seed: {args.start_seed}\n")

        spymaster, guesser = _build_pair(
            args.spymaster, args.guesser, vocabulary_path, args.seed
        )

        harness = EvaluationHarness(
            wordlist_path=wordlist_path,
            spymaster=spymaster,
            guesser=guesser,
            max_turns=args.max_turns,
            max_guesses=args.max_guesses,
            verbose=args.verbose
        )
        
        metrics, results = harness.evaluate_with_details(
            num_games=args.num_games,
            start_seed=args.start_seed
        )
        
        # Print results
        print("\n" + "="*70)
        print(metrics)
        print("="*70)
        
        # Save results if output specified
        if args.output:
            output_data = {
                "config": {
                    "spymaster": args.spymaster,
                    "guesser": args.guesser,
                    "wordlist": wordlist_path,
                    "vocabulary": vocabulary_path,
                    "num_games": args.num_games,
                    "start_seed": args.start_seed,
                    "agent_seed": args.seed
                },
                "metrics": metrics.to_dict(),
                "games": [r.to_dict() for r in results]
            }
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

