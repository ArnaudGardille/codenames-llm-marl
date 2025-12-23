#!/usr/bin/env python3
"""Debug script to see the actual prompt sent to the LLM."""

from codenames_rl.agents.baselines import LLMSpymaster
from codenames_rl.env.spaces import Observation, CardColor, GamePhase

# Create a sample observation
board_words = [
    "SHAKESPEARE", "SWITCH", "AGENT", "LEAD", "LEPRECHAUN",
    "BELL", "AMBULANCE", "TOOTH", "DRAGON", "PAPER",
    "EGYPT", "BACK", "MINT", "MASS", "CHECK",
    "PLANE", "TURKEY", "OCTOPUS", "DIAMOND", "HAWK",
    "PYRAMID", "PALM", "SINK", "NINJA", "SMUGGLER"
]

board_colors = [
    CardColor.TEAM, CardColor.OPPONENT, CardColor.NEUTRAL, CardColor.OPPONENT, CardColor.NEUTRAL,
    CardColor.TEAM, CardColor.TEAM, CardColor.OPPONENT, CardColor.OPPONENT, CardColor.OPPONENT,
    CardColor.NEUTRAL, CardColor.ASSASSIN, CardColor.NEUTRAL, CardColor.TEAM, CardColor.NEUTRAL,
    CardColor.TEAM, CardColor.TEAM, CardColor.OPPONENT, CardColor.TEAM, CardColor.TEAM,
    CardColor.NEUTRAL, CardColor.OPPONENT, CardColor.NEUTRAL, CardColor.TEAM, CardColor.OPPONENT
]

obs = Observation(
    board_words=board_words,
    revealed_mask=[False] * 25,
    board_colors=board_colors,
    current_clue=None,
    current_count=None,
    remaining_guesses=0,
    phase=GamePhase.SPYMASTER_TURN,
    team_remaining=9,
    opponent_remaining=8
)

print("="*70)
print("DEBUG: Actual Prompt Sent to LLM")
print("="*70)

spymaster = LLMSpymaster(device="cpu", seed=42)

# Manually build the prompt as the agent does
team_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
              if c == CardColor.TEAM and not r]
opponent_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                  if c == CardColor.OPPONENT and not r]
neutral_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                 if c == CardColor.NEUTRAL and not r]
assassin_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                  if c == CardColor.ASSASSIN and not r]
revealed_words = []

system_message = (
    "You are a Spymaster in Codenames. Your team MUST win!\n\n"
    "GAME OBJECTIVE:\n"
    "Help your team identify ALL your team's words before the opponent finds theirs. "
    "If your team reveals the ASSASSIN word, you IMMEDIATELY LOSE.\n\n"
    "YOUR ROLE:\n"
    "Give a ONE-WORD clue that connects to multiple team words while avoiding:\n"
    "- ASSASSIN (instant loss)\n"
    "- OPPONENT words (helps them win)\n"
    "- NEUTRAL words (wastes guesses)\n\n"
    "CLUE RULES:\n"
    "(1) Must be ONE word only\n"
    "(2) CANNOT be ANY word currently on the board\n"
    "(3) Must contain only letters and numbers (no hyphens, no compound words)\n"
    "(4) Count = how many team words your clue relates to\n\n"
    "OUTPUT FORMAT:\n"
    "Respond ONLY with valid JSON: {\"clue\": \"<your_clue_here>\", \"count\": <number>}\n\n"
    "Think strategically - connect multiple team words while staying far from danger!"
)

example1_user = """TEAM_WORDS: cat, dog, mouse
OPPONENT_WORDS: car, tree, house
NEUTRAL_WORDS: sky, ocean
ASSASSIN: fire
ALL_BOARD_WORDS (DO NOT USE): cat, dog, mouse, car, tree, house, sky, ocean, fire

Give me a clue as JSON: {"clue": "<your_clue_here>", "count": <number>}"""

example1_assistant = '{"clue": "pets", "count": 3}'

example2_user = """TEAM_WORDS: apple, banana, orange
OPPONENT_WORDS: desk, chair
NEUTRAL_WORDS: clock, lamp
ASSASSIN: knife
ALL_BOARD_WORDS (DO NOT USE): apple, banana, orange, desk, chair, clock, lamp, knife

Give me a clue as JSON: {"clue": "<your_clue_here>", "count": <number>}"""

example2_assistant = '{"clue": "fruit", "count": 3}'

user_message = f"""TEAM_WORDS: {', '.join(team_words)}
OPPONENT_WORDS: {', '.join(opponent_words)}
NEUTRAL_WORDS: {', '.join(neutral_words)}
ASSASSIN: {', '.join(assassin_words)}
REVEALED: {', '.join(revealed_words) if revealed_words else '(none)'}

ALL_BOARD_WORDS (DO NOT USE): {', '.join(obs.board_words)}

Give me a clue as JSON: {{"clue": "<your_clue_here>", "count": <number>}}"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": example1_user},
    {"role": "assistant", "content": example1_assistant},
    {"role": "user", "content": example2_user},
    {"role": "assistant", "content": example2_assistant},
    {"role": "user", "content": user_message}
]

text = spymaster.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n" + "="*70)
print("FULL PROMPT (as sent to model):")
print("="*70)
print(text)
print("\n" + "="*70)
print("Searching for 'word' in prompt...")
print("="*70)

# Count occurrences of "word" (case-insensitive)
import re
word_matches = list(re.finditer(r'\bword\b', text, re.IGNORECASE))
if word_matches:
    print(f"⚠️  Found {len(word_matches)} occurrence(s) of 'word' in prompt:")
    for i, match in enumerate(word_matches, 1):
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end]
        print(f"\n  {i}. Position {match.start()}-{match.end()}:")
        print(f"     ...{context}...")
else:
    print("✅ No occurrences of 'word' found in prompt")

print("\n" + "="*70)
print("Testing actual generation...")
print("="*70)

# Test actual generation
inputs = spymaster.tokenizer(text, return_tensors="pt").to(spymaster.device)

import torch
with torch.no_grad():
    outputs = spymaster.model.generate(
        **inputs,
        max_new_tokens=spymaster.max_new_tokens,
        temperature=spymaster.temperature,
        do_sample=True,
        pad_token_id=spymaster.tokenizer.eos_token_id
    )

generated = spymaster.tokenizer.decode(
    outputs[0][inputs['input_ids'].shape[1]:],
    skip_special_tokens=True
)

print(f"Generated output: {generated}")

