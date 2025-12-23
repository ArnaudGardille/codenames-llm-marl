#!/usr/bin/env python3
"""CLI script for running agent evaluations."""

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
        choices=["random", "embeddings", "qwen_embedding", "llm"],
        default="embeddings",
        help="Spymaster agent type"
    )
    parser.add_argument(
        "--guesser",
        type=str,
        choices=["random", "embeddings", "qwen_embedding", "llm"],
        default="embeddings",
        help="Guesser agent type"
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
        # Compare all combinations
        print("Comparing baseline agent combinations...")
        print(f"Wordlist: {wordlist_path}")
        print(f"Vocabulary: {vocabulary_path}")
        print(f"Games per config: {args.num_games}")
        print(f"Starting seed: {args.start_seed}\n")
        
        configs = {}
        
        # Random-Random
        configs["Random/Random"] = {
            "spymaster": RandomSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": RandomGuesser(seed=args.seed)
        }
        
        # Random-Embeddings
        configs["Random/Embeddings"] = {
            "spymaster": RandomSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": EmbeddingsGuesser(seed=args.seed)
        }
        
        # Embeddings-Random
        configs["Embeddings/Random"] = {
            "spymaster": EmbeddingsSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": RandomGuesser(seed=args.seed)
        }
        
        # Embeddings-Embeddings
        configs["Embeddings/Embeddings"] = {
            "spymaster": EmbeddingsSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": EmbeddingsGuesser(seed=args.seed)
        }
        
        # QwenEmbedding-QwenEmbedding (share model to save memory)
        qwen_emb_spymaster = QwenEmbeddingSpymaster(vocabulary_path=vocabulary_path, seed=args.seed)
        configs["QwenEmbedding/QwenEmbedding"] = {
            "spymaster": qwen_emb_spymaster,
            "guesser": QwenEmbeddingGuesser(
                model=qwen_emb_spymaster.model,
                tokenizer=qwen_emb_spymaster.tokenizer,
                device=qwen_emb_spymaster.device,
                seed=args.seed
            )
        }
        
        # QwenEmbedding-Embeddings
        configs["QwenEmbedding/Embeddings"] = {
            "spymaster": QwenEmbeddingSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": EmbeddingsGuesser(seed=args.seed)
        }
        
        # Embeddings-QwenEmbedding
        configs["Embeddings/QwenEmbedding"] = {
            "spymaster": EmbeddingsSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": QwenEmbeddingGuesser(seed=args.seed)
        }
        
        # LLM-LLM (share model to save memory)
        llm_spymaster = LLMSpymaster(seed=args.seed)
        configs["LLM/LLM"] = {
            "spymaster": llm_spymaster,
            "guesser": LLMGuesser(
                model=llm_spymaster.model,
                tokenizer=llm_spymaster.tokenizer,
                device=llm_spymaster.device,
                seed=args.seed
            )
        }
        
        # LLM-Embeddings
        configs["LLM/Embeddings"] = {
            "spymaster": LLMSpymaster(seed=args.seed),
            "guesser": EmbeddingsGuesser(seed=args.seed)
        }
        
        # Embeddings-LLM
        configs["Embeddings/LLM"] = {
            "spymaster": EmbeddingsSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": LLMGuesser(seed=args.seed)
        }
        
        # LLM-Random
        configs["LLM/Random"] = {
            "spymaster": LLMSpymaster(seed=args.seed),
            "guesser": RandomGuesser(seed=args.seed)
        }
        
        # Random-LLM
        configs["Random/LLM"] = {
            "spymaster": RandomSpymaster(vocabulary_path=vocabulary_path, seed=args.seed),
            "guesser": LLMGuesser(seed=args.seed)
        }
        
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
        # Single configuration
        print(f"Evaluating: {args.spymaster.title()} Spymaster + {args.guesser.title()} Guesser")
        print(f"Wordlist: {wordlist_path}")
        if vocabulary_path:
            print(f"Vocabulary: {vocabulary_path}")
        print(f"Games: {args.num_games}")
        print(f"Starting seed: {args.start_seed}\n")
        
        # Create agents
        spymaster = create_agent(
            f"{args.spymaster}_spymaster",
            vocabulary_path=vocabulary_path,
            seed=args.seed
        )
        
        # Share model if both agents are LLM or qwen_embedding to save memory
        if args.spymaster == "llm" and args.guesser == "llm":
            guesser = LLMGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device=spymaster.device,
                seed=args.seed
            )
        elif args.spymaster == "qwen_embedding" and args.guesser == "qwen_embedding":
            guesser = QwenEmbeddingGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device=spymaster.device,
                seed=args.seed
            )
        else:
            guesser = create_agent(
                f"{args.guesser}_guesser",
                seed=args.seed
            )
        
        # Run evaluation
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

