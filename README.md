# Codenames RL (LLM Self-Play)
Evaluating LLM Agents for Codenames and Fine-Tuning Them 

This repository implements **two distinct LLM agents** for Codenames — **Spymaster** and **Guesser** — trained and evaluated in a reproducible way.  
The goal is to build a **testable**, **measurable**, **iterative** system where the environment (the rules) is the source of truth, and learning follows a SFT → DPO → RL (GRPO) loop using **TRL**.

> **Architecture choice:** Each role has its own LoRA adapter (more stable than a single multi-role model). Training prioritizes the Spymaster first (richer decision space), then the Guesser.

---

## Goals

- A **deterministic Codenames environment** (rules in code, reproducible seeds).
- **Strong baselines** (heuristics + embeddings) to establish a performance floor.
- **LLM training** via:
  1. **SFT** (imitation),
  2. **DPO** (preferences),
  3. **GRPO** (online RL, optional / advanced).
- **Rigorous evaluation** on a frozen benchmark set (versioned seeds + wordlists).
- **Full test coverage**: unit, invariants, integration, E2E, plus LLM-output checks (pytest + DeepEval).

---

## Key Concepts

### Two Agent Types (trained separately)

| Agent | Observation | Action | Goal |
|-------|-------------|--------|------|
| **Spymaster** | 25 words + hidden labels (team/opponent/neutral/assassin) + revealed words + history | `(clue_word, clue_number)` | Give clues that maximize team guesses while avoiding danger |
| **Guesser** | 25 words + revealed words + current clue + guesses remaining | `guess_word` or `STOP` | Guess team words based on clue, stop before mistakes |

### MDP Formalization

The game is modeled as a Markov Decision Process:

**State (Spymaster)**
- `words`: list of 25 board words
- `labels`: hidden mapping (TEAM / OPPONENT / NEUTRAL / ASSASSIN)
- `revealed`: words already uncovered (with their type)
- `history`: previous clues given

**State (Guesser)**
- `words`: list of 25 board words
- `revealed`: words already uncovered (with their type)
- `clue`: current `(word, number)` from spymaster
- `guesses_left`: remaining guesses this turn

**Rewards**
| Event | Reward |
|-------|--------|
| Team word guessed | +1 |
| Opponent word guessed | -1 (ends turn) |
| Neutral word guessed | -0.2 (ends turn) |
| Assassin guessed | -5 (game lost) |
| Invalid clue | -2 (ends turn, official rules penalize this) |

> **Reward shaping is critical.** Without it, RL is notoriously unstable on Codenames due to sparse rewards and combinatorial action space.

### Engineering Principles
- Rules are **never "left to the prompt"**: clue validity and transitions are **implemented in code**.
- The agent is constrained to a **strict action schema** (JSON / dataclasses) validated in code.
- Exploration is controlled: prefer **"propose K candidates then choose"** over "invent any clue".
- **Hybrid approach recommended**: LLM generates candidates, embeddings + heuristics score/filter, code enforces rules.

-

## Installation



```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

---

## Quickstart

1) Run a game with a baseline:
```bash
python -m codenames_rl.eval.harness --config configs/base.yaml --agent baseline_embeddings
```

2) Run with LLM baseline (requires GPU):
```python
from codenames_rl.agents.baselines import LLMSpymaster, LLMGuesser

# Downloads model automatically on first run
spymaster = LLMSpymaster(seed=42, temperature=0.7)
guesser = LLMGuesser(seed=42, temperature=0.7)

# Share model between agents to save memory
guesser_shared = LLMGuesser(
    model=spymaster.model,
    tokenizer=spymaster.tokenizer,
    seed=42
)
```

**Getting the full 7B model:** See [`doc/GET_LLM.md`](doc/GET_LLM.md) or run:
```bash
python scripts/download_and_test_llm.py
```

**Apple Silicon (M1/M2/M3/M4):** See [`doc/APPLE_SILICON.md`](doc/APPLE_SILICON.md) for MPS-optimized setup.

3) Run the test suite:
```bash
pytest -q
pytest -q -m "not slow"  # Skip LLM tests
```

4) Evaluate on the frozen benchmark:
```bash
python scripts/run_eval.py --config configs/base.yaml --split eval
```

5) Run the interactive Streamlit app:
```bash
streamlit run src/codenames_rl/app/app.py
```

The app will open in your browser at `http://localhost:8501` and provides:
- Interactive 5×5 word board
- Spymaster view toggle (reveals card colors)
- Language selection (English/French)
- Seed control for reproducible games
- Turn-based gameplay (Spymaster gives clues, Guesser makes guesses)
- Game history sidebar

---

## Game Modes

This project supports two distinct modes for different training and evaluation scenarios:

### 1. Cooperative Mode (2-Player)

**Environment**: `CodenamesEnv` (Gymnasium)

You control ONE team (Spymaster + Guesser) working together:
- **Your agents**: Spymaster gives clues, Guesser makes guesses
- **Opponent cards**: Passive obstacles (not controlled by another team)
- **Goal**: Find all your team's words before hitting the assassin
- **Use case**: Simpler training scenario, faster iteration

```python
from codenames_rl.env import CodenamesEnv

env = CodenamesEnv(wordlist_path="configs/wordlist_en.txt")
obs, info = env.reset(seed=42)

# Your spymaster gives a clue
spymaster_action = spymaster.get_clue(obs)
obs, reward, done, truncated, info = env.step(spymaster_action)

# Your guesser makes guesses
guesser_action = guesser.get_guess(obs)
obs, reward, done, truncated, info = env.step(guesser_action)
```

### 2. Adversarial Mode (4-Player Competitive)

**Environments**: 
- `CodenamesAdversarialPZ` (PettingZoo) - for multi-agent research
- `CodenamesAdversarialGym` (Gymnasium + self-play) - for TRL training

True 4-player competition with two teams:
- **Team A**: Your agents (Spymaster + Guesser)
- **Team B**: Opponent agents (Spymaster + Guesser)
- **Teams alternate**: Team A plays full turn, then Team B plays
- **Goal**: Be the first team to find all your words
- **Use case**: Self-play training, competitive evaluation

#### PettingZoo Interface (Multi-Agent Control)
```python
from codenames_rl.env.adversarial_pz import env as make_env

env = make_env(wordlist_path="configs/wordlist_en.txt")
env.reset(seed=42)

# Control all 4 agents
agents = {
    'team_a_spymaster': LLMSpymaster(seed=42),
    'team_a_guesser': LLMGuesser(seed=42),
    'team_b_spymaster': EmbeddingsSpymaster(...),
    'team_b_guesser': EmbeddingsGuesser(...),
}

for agent in env.agent_iter():
    obs = env.observe(agent)
    action = agents[agent].get_action(obs)
    env.step(action)
```

#### Gymnasium Self-Play Interface (TRL Compatible)
```python
from codenames_rl.env.adversarial_gym import CodenamesAdversarialGym

# Opponent uses frozen policies
opponent_spy = EmbeddingsSpymaster(...)
opponent_guess = EmbeddingsGuesser(...)

env = CodenamesAdversarialGym(
    wordlist_path="configs/wordlist_en.txt",
    opponent_spymaster_policy=opponent_spy.get_clue,
    opponent_guesser_policy=opponent_guess.get_guess
)

# TRL sees this as single-agent environment
obs, info = env.reset(seed=42)

# Your turn (Team A)
action = your_agent.get_action(obs)
obs, reward, done, truncated, info = env.step(action)
# Opponent's turn executes automatically inside step()
```

**Training Progression**:
1. Start with **cooperative mode** for initial training (simpler, faster)
2. Move to **adversarial mode** for advanced training (self-play, competition)
3. Use **self-play curriculum**: weak opponents → strong opponents → self



## Evaluation

### Frozen Benchmark
- `splits/` (or `configs/`) contains **versioned** seeds and wordlists.
- Strict separation:
  - **train seeds**
  - **eval seeds**
  - **holdout seeds** (rare, for final comparison only)

### Metrics Tracked
- Win rate
- Average score / game
- Illegal clue rate
- Assassin hit rate
- Average game length
- Performance vs baselines (A/B)

### Running Evaluations

**Cooperative Mode (2-player)**:
```bash
# Single configuration
python scripts/run_eval.py \
    --wordlist configs/wordlist_en.txt \
    --vocabulary configs/vocabulary_en.txt \
    --spymaster llm \
    --guesser llm \
    --num-games 100

# Compare all baselines
python scripts/run_eval.py \
    --wordlist configs/wordlist_en.txt \
    --vocabulary configs/vocabulary_en.txt \
    --compare \
    --num-games 100
```

**Adversarial Mode (4-player competitive)**:
```bash
# LLM team vs Embeddings team
python scripts/run_eval_adversarial.py \
    --wordlist configs/wordlist_en.txt \
    --vocabulary configs/vocabulary_en.txt \
    --red-spymaster llm \
    --red-guesser llm \
    --blue-spymaster embeddings \
    --blue-guesser embeddings \
    --num-games 50 \
    --output results/llm_vs_embeddings.json
```

---

## Testing (pytest + DeepEval)

### Test Types
- `tests/unit/`: rules and transitions (fast, deterministic)
- `tests/property/`: invariants (Hypothesis recommended)
- `tests/integration/`: LLM wrappers (parsing, schemas, error handling)
- `tests/e2e/`: smoke tests over a few games (marked `slow`)
- `tests/llm_quality/` (optional): DeepEval checks (format, constraints, rubrics)

### Commands
```bash
pytest -q
pytest -q -m "not slow"
pytest -q -m slow
```

---

## Configuration

Runs are driven by `configs/*.yaml` (model, quantization, hyperparams, seeds, reward shaping).  
Each run logs:
- git commit
- full config
- wordlist versions
- evaluation metrics

---

## Reproducibility

- A run = (seed, config, wordlist_version, code_commit)
- The environment is deterministic.
- Evaluations always run on the same frozen seeds.

---

## Prompt Formats

### Spymaster Input
```
WORDS: apple, river, bank, knight, ...
TEAM_WORDS: apple, knight, castle
OPPONENT_WORDS: river, bank
NEUTRAL_WORDS: cloud, paper, ...
ASSASSIN: bomb
REVEALED: [] 
```

### Spymaster Output
```
CLUE: fruit
NUMBER: 2
```

### Guesser Input
```
WORDS: apple, river, bank, knight, ...
REVEALED: [river (OPPONENT)]
CLUE: fruit
NUMBER: 2
GUESSES_LEFT: 3
```

### Guesser Output
```
GUESS: apple
```
or
```
STOP
```

> The model outputs a structured response; **validation happens in code** (single word, not on board, allowed vocabulary, etc.). Invalid outputs incur penalties.





---

## License

To be defined based on your needs. Note that the model license (and any datasets you use) may introduce additional constraints.
