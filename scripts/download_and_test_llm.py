#!/usr/bin/env python3
"""
Download and test the full Qwen2.5-7B-Instruct model.

This script will:
1. Download the full 7B model from HuggingFace (requires ~14GB disk space)
2. Test it on a sample game observation
3. Verify it works correctly

Requirements:
- GPU with at least 14GB VRAM for FP16 (or 28GB for FP32)
- Or use CPU (very slow, not recommended for production)

Usage:
    # With GPU (recommended)
    python scripts/download_and_test_llm.py
    
    # Force CPU (slow)
    python scripts/download_and_test_llm.py --cpu
"""

import argparse
import torch
from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser
from codenames_rl.env.spaces import CardColor, GamePhase, Observation


def check_system():
    """Check system capabilities."""
    print("\n" + "="*70)
    print("System Check")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check for Apple Silicon MPS
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon) available: {has_mps}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total_memory:.1f} GB")
        
        if total_memory < 14:
            print("\n⚠️  WARNING: GPU has less than 14GB VRAM.")
            print("   The 7B model in FP16 requires ~14GB.")
            print("   Consider using a smaller model or quantization.")
            return False
    elif has_mps:
        print("\n✓ Apple Silicon GPU detected!")
        print("  Using MPS (Metal Performance Shaders) backend")
        print("  Recommended for M1/M2/M3/M4 chips")
        print("  7B model should work with 16GB+ unified memory")
        return True
    else:
        print("\n⚠️  WARNING: No GPU detected. Model will run on CPU (VERY slow).")
        print("   For production use, a GPU is highly recommended.")
    
    return True


def create_test_observation():
    """Create a sample game observation."""
    return Observation(
        board_words=[
            "apple", "dog", "river", "mountain", "book",
            "car", "tree", "phone", "house", "cloud",
            "star", "ocean", "guitar", "lamp", "chair",
            "sun", "moon", "rain", "snow", "wind",
            "fire", "water", "earth", "light", "dark"
        ],
        revealed_mask=[False] * 25,
        board_colors=[
            # 9 team words (nature theme)
            CardColor.TEAM, CardColor.TEAM, CardColor.TEAM,  # apple, dog, river
            CardColor.TEAM, CardColor.TEAM, CardColor.TEAM,  # mountain, book, car
            CardColor.TEAM, CardColor.TEAM, CardColor.TEAM,  # tree, phone, house
            # 8 opponent words
            CardColor.OPPONENT, CardColor.OPPONENT, CardColor.OPPONENT,  # cloud, star, ocean
            CardColor.OPPONENT, CardColor.OPPONENT, CardColor.OPPONENT,  # guitar, lamp, chair
            CardColor.OPPONENT, CardColor.OPPONENT,  # sun, moon
            # 7 neutral words
            CardColor.NEUTRAL, CardColor.NEUTRAL, CardColor.NEUTRAL,  # rain, snow, wind
            CardColor.NEUTRAL, CardColor.NEUTRAL, CardColor.NEUTRAL,  # fire, water, earth
            CardColor.NEUTRAL,  # light
            # 1 assassin
            CardColor.ASSASSIN  # dark
        ],
        current_clue=None,
        current_count=None,
        remaining_guesses=0,
        phase=GamePhase.SPYMASTER_TURN,
        team_remaining=9,
        opponent_remaining=8
    )


def test_spymaster(model_name, device):
    """Test the Spymaster agent."""
    print("\n" + "="*70)
    print("Testing Spymaster")
    print("="*70)
    
    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")
    print("\n⏳ This will download ~14GB on first run (cached afterwards)...")
    print("   Download location: ~/.cache/huggingface/hub/")
    
    try:
        spymaster = LLMSpymaster(
            model_name=model_name,
            device=device,
            temperature=0.7,
            seed=42
        )
        print("\n✓ Model loaded successfully!")
        
        # Create test observation
        obs = create_test_observation()
        team_words = [obs.board_words[i] for i, c in enumerate(obs.board_colors) if c == CardColor.TEAM]
        opponent_words = [obs.board_words[i] for i, c in enumerate(obs.board_colors) if c == CardColor.OPPONENT]
        assassin_word = [obs.board_words[i] for i, c in enumerate(obs.board_colors) if c == CardColor.ASSASSIN][0]
        
        print("\n" + "-"*70)
        print("Game Board:")
        print("-"*70)
        print(f"Team words (9): {', '.join(team_words)}")
        print(f"Opponent words (8): {', '.join(opponent_words[:5])}...")
        print(f"Assassin: {assassin_word}")
        
        print("\n⏳ Generating clue (this may take 10-30 seconds on CPU)...")
        action = spymaster.get_clue(obs)
        
        print("\n" + "-"*70)
        print("Generated Clue:")
        print("-"*70)
        print(f"  Clue: '{action.clue}'")
        print(f"  Count: {action.count}")
        
        # Validate
        board_lower = [w.lower() for w in obs.board_words]
        is_valid = action.clue.lower() not in board_lower
        
        if is_valid:
            print("\n✓ Clue is VALID (not on board)")
        else:
            print("\n✗ Clue is INVALID (word is on board)")
        
        return spymaster
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_guesser(spymaster, device):
    """Test the Guesser agent."""
    print("\n" + "="*70)
    print("Testing Guesser")
    print("="*70)
    
    try:
        # Reuse spymaster's model
        if spymaster:
            print("\nReusing Spymaster's model (saves memory)...")
            guesser = LLMGuesser(
                model=spymaster.model,
                tokenizer=spymaster.tokenizer,
                device=device,
                temperature=0.7,
                seed=42
            )
        else:
            print("\nLoading separate model for Guesser...")
            guesser = LLMGuesser(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                device=device,
                temperature=0.7,
                seed=42
            )
        
        print("✓ Guesser initialized")
        
        # Create observation with active clue
        obs = create_test_observation()
        obs.current_clue = "nature"
        obs.current_count = 3
        obs.remaining_guesses = 4
        obs.phase = GamePhase.GUESSER_TURN
        
        unrevealed = [w for w, r in zip(obs.board_words, obs.revealed_mask) if not r]
        
        print("\n" + "-"*70)
        print("Game State:")
        print("-"*70)
        print(f"Clue given: '{obs.current_clue}' (count: {obs.current_count})")
        print(f"Unrevealed words: {', '.join(unrevealed[:10])}...")
        
        print("\n⏳ Generating guess...")
        action = guesser.get_guess(obs)
        
        print("\n" + "-"*70)
        print("Generated Guess:")
        print("-"*70)
        if action.word_index is not None:
            guessed_word = obs.board_words[action.word_index]
            word_color = obs.board_colors[action.word_index]
            print(f"  Guess: '{guessed_word}'")
            print(f"  Actual color: {word_color.value}")
            if word_color == CardColor.TEAM:
                print("  ✓ Correct! (team word)")
            elif word_color == CardColor.ASSASSIN:
                print("  ✗ Assassin! (game over)")
            else:
                print(f"  ✗ Wrong ({word_color.value})")
        else:
            print("  Decision: STOP")
        
        print("\n✓ Guesser test complete")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Download and test Qwen2.5-7B-Instruct")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (slow)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Model name (default: Qwen/Qwen2.5-7B-Instruct)")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Qwen2.5-7B-Instruct Download and Test")
    print("="*70)
    
    # Check system
    system_ok = check_system()
    
    # Determine device
    if args.cpu:
        device = "cpu"
        print("\n⚠️  Using CPU (forced by --cpu flag)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("\n✓ Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("\n✓ Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("\n⚠️  Using CPU (no GPU available)")
    
    if device == "cpu" and not args.cpu:
        response = input("\nContinue with CPU? (very slow) [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Test agents
    spymaster = test_spymaster(args.model, device)
    
    if spymaster:
        test_guesser(spymaster, device)
    
    print("\n" + "="*70)
    print("✓ Test Complete!")
    print("="*70)
    print("\nThe model is now cached in: ~/.cache/huggingface/hub/")
    print("Next runs will be much faster (no download needed).")
    print("\nTo use in your code:")
    print("    from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser")
    print("    spymaster = LLMSpymaster(model_name='Qwen/Qwen2.5-7B-Instruct')")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

