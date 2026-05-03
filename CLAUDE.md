# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Codenames RL (LLM Self-Play): a research codebase for evaluating and (eventually) training LLM agents to play Codenames in two distinct roles — **Spymaster** (gives a `(clue_word, count)`) and **Guesser** (picks a board word or `STOP`). Training plan: SFT → DPO → GRPO via TRL, with separate LoRA adapters per role.

## ⚠️ Critical repository state

**The `codenames_rl.env` package is missing from `src/codenames_rl/`** even though virtually every other file imports from it. The rest of the codebase expects these modules:

- `codenames_rl.env.spaces` — `CardColor`, `GamePhase`, `Observation`, `SpymasterAction`, `GuesserAction`
- `codenames_rl.env.core` — `CodenamesEnv` (Gymnasium, cooperative mode)
- `codenames_rl.env.validation` — `is_valid_clue`, `load_wordlist`
- `codenames_rl.env.adversarial_core` — `CodenamesAdversarialCore` (4-player game logic)
- `codenames_rl.env.adversarial_pz` — `CodenamesAdversarialPZ` + `env()` factory (PettingZoo AEC)
- `codenames_rl.env.adversarial_gym` — `CodenamesAdversarialGym` (Gym wrapper with frozen-opponent policies, used for TRL self-play)

`tests/conftest.py`, `tests/unit/test_env.py`, `tests/unit/test_adversarial.py`, the eval harness, agents, the Streamlit app, and the training scripts all import from these modules. **Without them nothing imports.** When working in this repo, treat the env module as a known gap; if the user asks you to run tests, evaluation, or the app, verify whether `src/codenames_rl/env/` has been (re)introduced before assuming commands will succeed. Also note `README.md` documents `python -m codenames_rl.eval.harness` as an entry point, but `harness.py` has no `__main__` — use `scripts/run_eval.py` instead.

## Common commands

Install (editable, with dev tools):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Tests (pytest config in `pyproject.toml`, testpaths=`tests`):
```bash
pytest -q                         # all tests
pytest -q -m "not slow"           # skip LLM-heavy integration tests
pytest tests/unit/test_eval.py    # single file
pytest tests/unit/test_eval.py::TestEvaluationMetrics::test_compute_metrics  # single test
```
The `slow` marker is the convention for tests that download/load LLM weights (see `tests/integration/`). Mark new heavyweight tests with `@pytest.mark.slow`.

Lint / type-check (configured but not wired into CI):
```bash
ruff check .
mypy src
```

Cooperative evaluation (Spymaster + Guesser vs passive opponent):
```bash
python scripts/run_eval.py --lang en --spymaster embeddings --guesser embeddings --num-games 100
python scripts/run_eval.py --lang en --compare              # run all baseline pairings
python scripts/run_eval.py --lang en --spymaster llm --guesser llm --verbose --output results/llm.json
```

Adversarial 4-player evaluation (Red vs Blue):
```bash
python scripts/run_eval_adversarial.py \
    --wordlist configs/wordlist_en.txt --vocabulary configs/vocabulary_en.txt \
    --red-spymaster llm --red-guesser llm \
    --blue-spymaster embeddings --blue-guesser embeddings \
    --num-games 50 --output results/llm_vs_embeddings.json
```

Streamlit demo:
```bash
streamlit run src/codenames_rl/app/app.py
```

Smoke-test LLM download / generation (~14 GB for default Qwen2.5-7B-Instruct):
```bash
python scripts/download_and_test_llm.py
```

## Architecture

### Layered package layout (`src/codenames_rl/`)

```
env/      (missing, see above)         game rules, MDP spaces, two env flavors (Gym + PettingZoo)
agents/   baselines.py, improved.py    Spymaster/Guesser implementations
eval/     harness.py, metrics.py       per-episode driver + aggregated metrics
app/      app.py                       Streamlit human-play UI
utils/    config.py                    .env loader, language path resolver, model defaults
```

### Agent taxonomy

All agents subclass `BaseSpymaster` / `BaseGuesser` from `agents/baselines.py`. `agents/__init__.py` is the public surface — `scripts/run_eval.py` selects agents by string keys mapped here. Implementations in order of cost/quality:

- **Random** — uniform over vocabulary / unrevealed words.
- **Embeddings** (sentence-transformers, default `all-MiniLM-L6-v2`). Spymaster scores candidate clues with `mean_sim(team) − α·max_sim(assassin) − β·max_sim(opponent) − γ·mean_sim(neutral)` (α=3.0, β=1.5, γ=0.3 are the tuned weights). Guesser ranks unrevealed words by similarity to the clue and `STOP`s below `confidence_threshold`.
- **QwenEmbedding** — same scoring scheme but with `Qwen/Qwen3-Embedding-8B`; supports model+tokenizer sharing between Spymaster and Guesser to halve VRAM.
- **LLM** — `Qwen/Qwen2.5-7B-Instruct` with chat templates, JSON-only output, retry-on-invalid loop, prompt-length guard, `bad_words_ids` token suppression for board words. Defaults come from `utils.config` (`LLM_MODEL_NAME`, `LLM_TEMPERATURE`, `LLM_MAX_NEW_TOKENS`, `LLM_QUANTIZATION`, `DEVICE`).
- **Improved** (`agents/improved.py`) — `ClusterSpymaster` (uses MIN intra-cluster sim, not mean), `ContextualGuesser`, `AdaptiveGuesser`, and a hybrid Retrieve-and-Rerank `CrossEncoderSpymaster`/`CrossEncoderGuesser`.

When both Spymaster and Guesser are LLM/QwenEmbedding, **always share `model` + `tokenizer`** (passed as kwargs) to avoid loading twice — `run_eval.py` and the `--compare` path do this; mirror that pattern for any new agent code.

`_load_tokenizer_with_fallback` handles models like Ministral whose tokenizer backend isn't recognized by `AutoTokenizer` — it falls back to `PreTrainedTokenizerFast` from the raw `tokenizer.json`.

### LLM Spymaster prompt contract

`LLMSpymaster.get_clue` enforces validity through layered defenses, in this order, and you should preserve all of them when editing:
1. System message states win condition + clue rules (one word, not on board, letters/numbers only).
2. Two few-shot exemplars in the chat template.
3. `ALL_BOARD_WORDS (DO NOT USE)` line in the user message.
4. `bad_words_ids` derived from every casing/spacing variant of every board word.
5. Up to 3 retries with seed jitter and temperature scaling (`temperature * (1 + 0.1·attempt)`).
6. `is_valid_clue` post-validation; failures raise `ValueError` after retries exhausted.

Output is parsed by regex (`{ ... "clue" ... "count" ... }`) then `json.loads`. The Guesser uses an analogous JSON-with-`STOP` contract.

### Evaluation harness flow

`EvaluationHarness.run_episode(seed)` (`eval/harness.py`):
1. Constructs `CodenamesEnv(wordlist_path, max_guesses, render_mode)`, calls `reset(seed=seed)`, resets agents.
2. Loops on `obs.phase`: SPYMASTER → call `spymaster.get_clue(obs)` → `env.step`; GUESSER → inner loop calling `guesser.get_guess(obs)` until phase changes or `terminated`.
3. Tracks per-turn cards, illegal clues (`info["invalid_clue"]`), outcome (`info["result"]` ∈ `win`/`loss_assassin`/`loss_opponent_won`/`truncated`), clue history.
4. Returns `GameResult`. `evaluate(num_games)` aggregates with `compute_metrics` → `EvaluationMetrics`.

Outcome strings and the reward shaping (+1 team, −1 opponent, −0.2 neutral, −5 assassin, −2 invalid clue) are the source of truth for metrics — keep them aligned with whatever `env/core.py` ends up emitting.

### Configuration system (`utils/config.py`)

All runtime knobs are environment variables (loaded from project-root `.env` via `python-dotenv`). See `.env.example` for the canonical list. Notable behavior:
- `EMBEDDING_MODEL`: if it contains `/` it's used verbatim, otherwise prefixed with `sentence-transformers/`.
- `LLM_QUANTIZATION` ∈ {`none`, `4bit`, `8bit`} — `4bit`/`8bit` require CUDA + bitsandbytes; using them on MPS raises a clear error directing users to FP16 / smaller models / MLX.
- `DEVICE` empty triggers CUDA → MPS → CPU auto-detection inside agents (the same fallback ladder is duplicated in each LLM/embedding agent constructor).
- `get_language_paths("en"|"fr")` returns absolute `(wordlist, vocabulary)` paths under `configs/`. Use this rather than hardcoding paths when adding multi-language code.

### Wordlists vs vocabularies

Two distinct files per language in `configs/`:
- `wordlist_{en,fr}.txt` — the **board pool** (~400 words) sampled to populate the 25-card grid.
- `vocabulary_{en,fr}.txt` — the **clue candidate pool** (~2.7k EN / 2.3k FR) the Spymaster searches over.

Don't conflate them. Spymasters take `vocabulary_path`; the env (and thus the harness) takes `wordlist_path`. The Random/Embeddings/QwenEmbeddings spymasters explicitly filter out any board word from candidates.

### Cooperative vs adversarial modes

- **Cooperative** (`CodenamesEnv`, Gymnasium): single team plays alone; opponent words just sit on the board. Used by `EvaluationHarness` and the Streamlit app.
- **Adversarial** (`CodenamesAdversarialPZ`, PettingZoo AEC + `CodenamesAdversarialGym` wrapper): two teams alternate Spymaster/Guesser turns. The Gym wrapper is designed for **TRL self-play**: the user trains one team while the other team's policies are passed in as frozen callables (`opponent_spymaster_policy`, `opponent_guesser_policy`) — see `scripts/train_adversarial_selfplay.py` for the integration pattern.

## Conventions to follow

- Action types are `SpymasterAction(clue, count)` and `GuesserAction(word_index)` where `word_index=None` means STOP/pass. Never represent a guess as a string at the env boundary.
- Card colors are an enum (`CardColor.TEAM/OPPONENT/NEUTRAL/ASSASSIN`); compare with the enum, not strings.
- Game-rule decisions belong in `env/` (when it exists). Agents do scoring + I/O only; never reimplement clue legality outside `is_valid_clue`.
- Reproducibility: pass `seed` everywhere. The harness uses `range(start_seed, start_seed+num_games)` by default; preserve that contract so result files are diffable across runs.
- LLM agents store the last raw decode in `self.last_raw_output` for debug printing in the harness — keep this hook when adding new generative agents.
