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
- Rules enforced in code (not prompts)
- Strict action validation (JSON schemas)
- Hybrid approach: LLM + embeddings + code validation

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

**Model Download & Hardware:**
- First run downloads ~14GB (Qwen2.5-7B-Instruct, cached in `~/.cache/huggingface/`)
- GPU: RTX 3090/4090 (14GB+ VRAM) or Apple Silicon M1/M2/M3/M4 with 16GB+ unified memory (auto-detects MPS)
- Smaller alternatives: Qwen2.5-1.5B (~3GB) or Qwen2.5-3B (~6GB)
- Quick test: `python scripts/download_and_test_llm.py`

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

**Cooperative Mode** (`CodenamesEnv` - Gymnasium): Single team vs passive opponents. Simpler, faster training.

**Adversarial Mode** (PettingZoo / Gymnasium): Full 4-player competitive (Team A vs Team B). For self-play and competitive evaluation.

See environment documentation for usage details.



## Evaluation

**Metrics**: Win rate, avg score, assassin rate, illegal clues, game length, performance vs baselines.

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

## Testing

```bash
pytest -q              # All tests
pytest -q -m "not slow"  # Skip LLM tests
```


---

## License

To be defined based on your needs. Note that the model license (and any datasets you use) may introduce additional constraints.
