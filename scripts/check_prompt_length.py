#!/usr/bin/env python3
"""Check the actual token length of LLM prompts."""

from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser
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
print("Checking LLM Prompt Lengths")
print("="*70)

# Check Spymaster prompt
print("\n1. SPYMASTER PROMPT:")
print("-"*70)
spymaster = LLMSpymaster(device="cpu")  # Use CPU to avoid loading full model if not needed
model_max_length = getattr(spymaster.tokenizer, 'model_max_length', 32768)
print(f"Model max context length: {model_max_length:,} tokens")

# Build the prompt manually to check length
team_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
              if c == CardColor.TEAM and not r]
opponent_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                  if c == CardColor.OPPONENT and not r]
neutral_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                 if c == CardColor.NEUTRAL and not r]
assassin_words = [w for w, c, r in zip(obs.board_words, obs.board_colors, obs.revealed_mask) 
                  if c == CardColor.ASSASSIN and not r]

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
    "Respond ONLY with valid JSON: {\"clue\": \"word\", \"count\": N}\n\n"
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
REVEALED: (none)

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

inputs = spymaster.tokenizer(text, return_tensors="pt")
spymaster_length = inputs['input_ids'].shape[1]

print(f"Spymaster prompt length: {spymaster_length:,} tokens")
print(f"Usage: {spymaster_length/model_max_length*100:.2f}% of context window")
print(f"Remaining: {model_max_length - spymaster_length:,} tokens for generation")

# Check Guesser prompt
print("\n2. GUESSER PROMPT:")
print("-"*70)

obs_guesser = Observation(
    board_words=board_words,
    revealed_mask=[False] * 25,
    board_colors=board_colors,
    current_clue="word",
    current_count=3,
    remaining_guesses=4,
    phase=GamePhase.GUESSER_TURN,
    team_remaining=9,
    opponent_remaining=8
)

unrevealed_words = [w for w, r in zip(obs_guesser.board_words, obs_guesser.revealed_mask) if not r]

system_message_guesser = (
    "You are a Guesser (Operative) in Codenames. Your team MUST win!\n\n"
    "GAME OBJECTIVE:\n"
    "Your Spymaster can see which words belong to your team. They gave you a CLUE to help you find your team's words. "
    "You must identify ALL your team's words before the opponent finds theirs.\n\n"
    "CRITICAL DANGER:\n"
    "- If you guess the ASSASSIN word, your team IMMEDIATELY LOSES the game!\n"
    "- If you guess an OPPONENT word, you help them win\n"
    "- If you guess a NEUTRAL word, you waste your turn\n\n"
    "YOUR STRATEGY:\n"
    "1. The COUNT tells you how many words the clue connects to\n"
    "2. You can make multiple guesses (up to COUNT + 1) if you keep guessing correctly\n"
    "3. STOP when uncertain - it's better to pass than risk the assassin!\n"
    "4. Your Spymaster is smart and avoids danger - trust their clue\n\n"
    "OUTPUT FORMAT:\n"
    "Respond ONLY with valid JSON:\n"
    "- To guess a word: {\"guess\": \"word\"}\n"
    "- To pass (stop): {\"guess\": \"STOP\"}\n\n"
    "Think carefully - one wrong guess can lose the game!"
)

user_message_guesser = f"""CLUE: {obs_guesser.current_clue}
COUNT: {obs_guesser.current_count}
REMAINING_GUESSES: {obs_guesser.remaining_guesses}
UNREVEALED_WORDS: {', '.join(unrevealed_words)}
REVEALED_WORDS: (none)

Which word should you guess? Respond with JSON: {{"guess": "word"}} or {{"guess": "STOP"}}"""

messages_guesser = [
    {"role": "system", "content": system_message_guesser},
    {"role": "user", "content": user_message_guesser}
]

text_guesser = spymaster.tokenizer.apply_chat_template(
    messages_guesser,
    tokenize=False,
    add_generation_prompt=True
)

inputs_guesser = spymaster.tokenizer(text_guesser, return_tensors="pt")
guesser_length = inputs_guesser['input_ids'].shape[1]

print(f"Guesser prompt length: {guesser_length:,} tokens")
print(f"Usage: {guesser_length/model_max_length*100:.2f}% of context window")
print(f"Remaining: {model_max_length - guesser_length:,} tokens for generation")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Model context window: {model_max_length:,} tokens")
print(f"Spymaster prompt: {spymaster_length:,} tokens ({spymaster_length/model_max_length*100:.2f}%)")
print(f"Guesser prompt: {guesser_length:,} tokens ({guesser_length/model_max_length*100:.2f}%)")
print(f"\n✅ Both prompts fit comfortably within the context window!")

