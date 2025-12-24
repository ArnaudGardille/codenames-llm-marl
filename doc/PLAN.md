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

---

## Recommended Tech Stack

- **Python 3.10+**
- **Gymnasium**: standard `reset/step` API, wrappers, reproducibility.
- **Transformers + Accelerate**
- **PEFT + bitsandbytes**: LoRA / QLoRA (fits well on 24 GB GPUs)
- **TRL**: SFT, DPO, GRPO (online RL)
- **pytest**: tests
- **DeepEval**: LLM output tests (format, constraints, rubrics)
- Dev tools: `ruff`, `mypy`, `pre-commit`

> GPU note: an RTX 3090 / 24 GB is a good fit for **7B/8B** models with **QLoRA** for SFT/DPO and a reasonable GRPO setup (small batches, short sequences).

---

## Project Steps (Pipeline)

### Step 0 — Environment (foundation)
- Implement `CodenamesEnv` (Gymnasium):
  - `reset(seed)`: generates a reproducible board
  - `step(action)`: applies an action and returns `obs, reward, terminated, truncated, info`
- Enforce **all** rules in code:
  - clue is **one word**
  - clue is **not on the board**
  - correct transitions (turn, terminal states, assassin)

**Deliverable:** deterministic env + unit tests for rules.

---

### Step 1 — Baselines
- **Random**: uniform sampling (lower bound)
- **Embeddings-based Spymaster**: `score(clue) = sim(clue, team_words) - α·sim(clue, enemies+assassin) - β·sim(clue, neutrals)`. Generate K candidates from allowed vocabulary, pick best.
- **Embeddings-based Guesser**: rank unrevealed words by similarity to clue, pick top-N
- **LLM-based (Qwen2.5-7B-Instruct)**: zero-shot prompting with chat format and JSON output parsing, strict validation
- **Prudent**: hard filters rejecting any clue too close to assassin/opponent

> Start here even if you want to finetune. Baselines provide (1) an immediate benchmark and (2) a data generator for SFT.

**Deliverable:** evaluation scripts + reference curves.

---

### Step 2 — Data Generation (traces)
- Let (baseline spymaster / baseline guesser) play and log:
  - observation → action → outcome
- Recommended storage: **JSONL** (simple) or **Parquet** (fast).
- Filtering:
  - discard illegal clues
  - optionally remove outliers

**Deliverable:** versioned synthetic dataset + stats (size, error rate, distributions).

---

### Step 3 — SFT (Supervised Fine-Tuning)
- Goal: learn the action format and a plausible behavior.
- Method: **QLoRA** on a 7B/8B instruct model.
- **Spymaster SFT**: `(board + labels) → (clue, number)`
- **Guesser SFT**: `(board + clue) → guess/STOP`
- One LoRA adapter per role (modular, easier to iterate).

**Deliverable:** LoRA checkpoints (spymaster + guesser) + metrics (loss, format compliance).

---

### Step 4 — DPO (Direct Preference Optimization)
- Build `(chosen, rejected)` pairs:
  - For each state, generate 2–8 candidate actions
  - Evaluate via simulation (or proxy: reward shaping + risk assessment)
  - Pair winning vs losing candidates
- Goal: learn useful preferences (lower risk, higher expected score).
- More natural than supervised learning for Codenames: "clue A is better than B" is easier to define than "the one correct clue".

**Deliverable:** DPO LoRA checkpoints + improvement on frozen benchmark.

---

### Step 5 — RL (GRPO, optional / advanced)
- Goal: squeeze extra performance via online optimization.
- **Warning:** RL on Codenames is notoriously hard (sparse rewards, combinatorial actions, easy to learn "cheats"). Only attempt after SFT+DPO are stable.
- Recommended strategy:
  - **reduce action space**: instead of free-form generation, have the model propose N candidates then classify/rank (much easier to optimize)
  - **self-play with adversarial mode**: train one team while opponent uses frozen policies
  - **reward shaping**: strong penalties for assassin proximity, invalid clues

**Self-Play Training Example**:
```python
from codenames_rl.env.adversarial_gym import CodenamesAdversarialGym

# Phase 1: Train against weak baseline
opponent_spy = RandomSpymaster(...)
opponent_guess = RandomGuesser(...)

env = CodenamesAdversarialGym(
    wordlist_path="configs/wordlist_en.txt",
    opponent_spymaster_policy=opponent_spy.get_clue,
    opponent_guesser_policy=opponent_guess.get_guess
)

# Train with TRL
trainer = GRPOTrainer(model, env, ...)
trainer.train()

# Phase 2: Freeze trained model, use as opponent
# Train next generation against previous version
opponent_spy_v2 = load_trained_model("checkpoint_v1")
opponent_guess_v2 = load_trained_model("checkpoint_v1")

# Continue training...
```

**Deliverable:** RL checkpoint + stable evaluation (controlled variance).

---

## Suggested Roadmap

**Foundation**
- [x] Complete env + strict clue validator
- [x] Baselines: random + embeddings + LLM (zero-shot) for both roles
- [ ] Evaluation harness + local leaderboard

**Data**
- [ ] Synthetic dataset generation (baseline self-play → JSONL/Parquet)
- [ ] Filtering pipeline (reject invalid clues, outliers)

**Spymaster (priority)**
- [ ] QLoRA SFT (spymaster)
- [ ] DPO (spymaster)

**Guesser**
- [ ] QLoRA SFT (guesser)
- [ ] DPO (guesser)

**Advanced (optional)**
- [ ] Alternating self-play (GRPO) + ablations
- [ ] Final report: baselines vs SFT vs DPO vs RL

---

## Model & Training Approach

This project uses **Qwen2.5-7B-Instruct** as the base model (MIT license, multilingual EN/FR support, excellent instruction-following).

**Training approach:**
- **Baseline (zero-shot)**: FP16 inference with chat-based prompting and JSON output validation (already implemented)
- **Fine-tuning**: QLoRA (LoRA adapters) on 24 GB GPUs for SFT → DPO → GRPO pipeline

**Recommended hybrid approach:**
- LLM proposes candidates (creative, contextual)
- Embeddings score/rank candidates (safety, relevance)
- Code enforces rules (no invalid clues reach the game)

This gives the best of both worlds: LLM fluency + controllable safety. Pure end-to-end LLM is possible but harder to debug and constrain.

---

## License

To be defined based on your needs. Note that the model license (and any datasets you use) may introduce additional constraints.
