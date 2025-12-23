"""Demo script showing TRL self-play training with adversarial Codenames.

This script demonstrates how to use the Gymnasium adversarial wrapper
for TRL training with frozen opponent policies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codenames_rl.env.adversarial_gym import CodenamesAdversarialGym
from codenames_rl.agents.baselines import (
    RandomSpymaster,
    RandomGuesser,
    EmbeddingsSpymaster,
    EmbeddingsGuesser,
)


def demo_self_play_episode():
    """Demonstrate a single episode with self-play environment."""
    print("="*70)
    print("Adversarial Self-Play Demo")
    print("="*70)
    print("\nSetup:")
    print("- Team A: You control (demo uses random actions)")
    print("- Team B: Opponent (baseline policies)")
    print()
    
    # Configuration
    wordlist_path = "configs/wordlist_en.txt"
    vocabulary_path = "configs/vocabulary_en.txt"
    
    # Create opponent policies (Team B)
    print("Loading opponent policies...")
    opponent_spy = EmbeddingsSpymaster(vocabulary_path, seed=42)
    opponent_guess = EmbeddingsGuesser(seed=42)
    print("✓ Opponent policies loaded (Embeddings baseline)")
    
    # Create environment
    print("\nCreating adversarial environment...")
    env = CodenamesAdversarialGym(
        wordlist_path=wordlist_path,
        opponent_spymaster_policy=opponent_spy.get_clue,
        opponent_guesser_policy=opponent_guess.get_guess,
        render_mode="human"
    )
    print("✓ Environment created")
    
    # Run episode
    print("\n" + "="*70)
    print("Starting Episode")
    print("="*70)
    
    obs, info = env.reset(seed=123)
    done = False
    step = 0
    max_steps = 50
    
    # Simple Team A policy (for demo - replace with trained model)
    team_a_spy = RandomSpymaster(vocabulary_path, seed=42)
    team_a_guess = RandomGuesser(seed=42)
    
    while not done and step < max_steps:
        print(f"\n--- Step {step + 1} ---")
        
        # Determine current role for Team A
        if env.core.current_phase == 'spymaster':
            print("Team A Spymaster's turn...")
            action = team_a_spy.get_clue(obs)
            print(f"Clue: {action.clue} ({action.count})")
        else:
            print("Team A Guesser's turn...")
            action = team_a_guess.get_guess(obs)
            if action.word_index is not None:
                print(f"Guessing: {obs.board_words[action.word_index]}")
            else:
                print("Passing")
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.2f}")
        if done:
            print(f"\n{'='*70}")
            print(f"Game Over!")
            print(f"Result: {info.get('result', 'unknown')}")
            print(f"Total Episode Reward: {info.get('episode_reward', 0):.2f}")
            print(f"{'='*70}")
        
        # Render current state
        if step % 5 == 0:  # Render every 5 steps
            env.render()
        
        step += 1
    
    env.close()
    print("\n✓ Demo completed")


def explain_trl_integration():
    """Explain how to integrate with TRL for actual training."""
    print("\n" + "="*70)
    print("TRL Integration Guide")
    print("="*70)
    print("""
To use this environment for actual TRL training:

1. SETUP ENVIRONMENT:
   ```python
   from codenames_rl.env.adversarial_gym import CodenamesAdversarialGym
   
   # Freeze opponent policies at current training stage
   opponent_spy = load_checkpoint("opponent_spymaster_v1.pth")
   opponent_guess = load_checkpoint("opponent_guesser_v1.pth")
   
   env = CodenamesAdversarialGym(
       wordlist_path="configs/wordlist_en.txt",
       opponent_spymaster_policy=opponent_spy,
       opponent_guesser_policy=opponent_guess
   )
   ```

2. TRL TRAINING (GRPO):
   ```python
   from trl import GRPOTrainer, GRPOConfig
   from transformers import AutoModelForCausalLM
   
   # Load your model
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
   
   # Configure trainer
   config = GRPOConfig(
       output_dir="./outputs",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       learning_rate=1e-5,
   )
   
   trainer = GRPOTrainer(
       model=model,
       config=config,
       train_dataset=...,  # Your dataset
       # Environment will be wrapped automatically
   )
   
   trainer.train()
   ```

3. SELF-PLAY CURRICULUM:
   
   Phase 1: Train against weak baselines (Random)
   Phase 2: Train against medium baselines (Embeddings)
   Phase 3: Self-play (frozen copy of your own model)
   Phase 4: Iterative improvement (periodically update frozen opponent)

4. EVALUATION:
   
   Test against multiple opponent strengths:
   - Random opponents (sanity check)
   - Embeddings opponents (baseline)
   - Previous versions (measure improvement)
   - Current version (robustness)

Note: The environment handles all opponent actions automatically,
      so TRL sees it as a standard single-agent Gymnasium environment.
""")


if __name__ == "__main__":
    print("Codenames Adversarial Self-Play Training Demo\n")
    
    try:
        demo_self_play_episode()
        explain_trl_integration()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure wordlist and vocabulary files exist:")
        print("  - configs/wordlist_en.txt")
        print("  - configs/vocabulary_en.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

