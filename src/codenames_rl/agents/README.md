# Codenames Baseline Agents

This module contains baseline agents for Codenames. These agents establish a **performance floor** that more sophisticated LLM-based agents should beat.

---

## Why Baselines?

Baselines serve three critical purposes:

1. **Benchmark**: Establish a reference point to measure progress
2. **Data generation**: Create synthetic training data for SFT/DPO
3. **Debugging**: Simple, deterministic agents help validate the environment

Good baselines should be:
- Simple to implement and understand
- Fast to run (for generating large datasets)
- Weak enough to have room for improvement
- Strong enough to occasionally succeed

---

## Agent Types

### Spymaster Agents

The Spymaster sees the hidden board labels and must give clues to help the Guesser identify team words while avoiding danger.

#### 1. **RandomSpymaster**

**Strategy**: Pick a random valid word from the vocabulary.

**How it works**:
- Loads a vocabulary of candidate words
- Filters out words that are on the current board
- Randomly selects a clue word
- Randomly picks a count (1-3)

**Strengths**:
- Fast (~1ms per clue)
- Deterministic with seed
- Never gives invalid clues (enforced by vocabulary filtering)

**Weaknesses**:
- No semantic connection to team words
- Doesn't avoid danger words (assassin, opponents)
- Very high assassin rate (~60%)

**Expected performance**: Win rate ~0%, avg score 4-5 cards

---

#### 2. **EmbeddingsSpymaster**

**Strategy**: Use semantic similarity to score candidate clues, prefer those close to team words and far from danger.

**How it works**:
1. Encode all board words and candidate clues using sentence-transformers
2. For each candidate clue, compute a score:
   ```
   score = mean_similarity(clue, team_words)
         - α × max_similarity(clue, assassin)
         - β × max_similarity(clue, opponent_words)
         - γ × mean_similarity(clue, neutral_words)
   ```
   Default weights: α=3.0, β=1.5, γ=0.3
3. Pick the highest-scoring valid clue
4. Count = number of team words above similarity threshold (default 0.3)

**Strengths**:
- Semantically meaningful clues
- Actively avoids assassin (0% assassin rate in testing)
- Generally safe play
- Tunable via α, β, γ parameters

**Weaknesses**:
- Limited by embedding quality (surface-level semantics only)
- No multi-hop reasoning ("fruit" → "apple" but not "fruit" → "New York" via "apple")
- No game strategy (doesn't consider board state, opponent progress)
- Slow (~1s per clue due to encoding)

**Expected performance**: Win rate ~10-20%, avg score 5-7 cards

**Note on Cross-Encoders**: Cross-Encoders (rerankers) provide superior accuracy by computing similarity scores directly from text pairs, but are slower than Bi-Encoders. For the Spymaster use case (scoring ~100 candidates × ~25 board words = ~2,500 pairs), pure Cross-Encoder would be prohibitively slow (~10-30s per clue). A hybrid **Retrieve & Re-Rank** approach (Bi-Encoder for retrieval, Cross-Encoder for re-ranking top-k) could improve accuracy while maintaining reasonable speed. See `CrossEncoderSpymaster` and `CrossEncoderGuesser` for implementations.

---

#### 3. **ClusterSpymaster**

**Strategy**: Find tight word clusters using minimum similarity within groups, rather than average similarity across all words.

**Key improvement over EmbeddingsSpymaster**:
- Uses **MIN similarity** within clusters to ensure tight connections between words
- EmbeddingsSpymaster uses MEAN similarity, which can connect loosely to many words
- ClusterSpymaster finds groups where ALL words are strongly related to the clue

**How it works**:
1. Encode all board words and candidate clues using sentence-transformers
2. For each candidate clue, identify team words above a similarity threshold (default 0.35)
3. Score the cluster using the **minimum** similarity within the cluster (ensures tight connection)
4. Apply the same penalty system as EmbeddingsSpymaster (assassin, opponent, neutral penalties)
5. Count = size of the cluster (number of team words above threshold)

**Strengths**:
- Finds tighter, more coherent word groups
- Better at identifying multi-word connections
- Same safety mechanisms as EmbeddingsSpymaster
- More accurate count predictions

**Weaknesses**:
- Still limited by embedding quality
- No game strategy beyond clustering
- Similar speed to EmbeddingsSpymaster (~1s per clue)

---

#### 4. **CrossEncoderSpymaster**

**Strategy**: Use hybrid **Retrieve & Re-Rank** approach: Bi-Encoder for fast retrieval, Cross-Encoder for accurate re-ranking.

**Key improvement over EmbeddingsSpymaster**:
- Cross-Encoders compute similarity scores directly from text pairs (more accurate)
- Better at understanding nuanced relationships between clues and words
- Hybrid approach balances speed and accuracy

**How it works**:
1. **Step 1 (Retrieval)**: Use Bi-Encoder to score ~100 candidate clues against board words (fast)
2. **Step 2 (Re-ranking)**: Use Cross-Encoder to re-rank top-20 candidates against all board words (accurate)
3. Apply same penalty system as EmbeddingsSpymaster (assassin, opponent, neutral penalties)
4. Count = number of team words above relative threshold in Cross-Encoder scores

**Strengths**:
- More accurate semantic matching than pure Bi-Encoder
- Better at understanding context-dependent relationships
- Maintains reasonable speed (~2-5s per clue) via hybrid approach
- Same safety mechanisms as EmbeddingsSpymaster

**Weaknesses**:
- Slower than pure Bi-Encoder (~2-5s vs ~1s per clue)
- Requires loading two models (Bi-Encoder + Cross-Encoder)
- Still limited compared to full LLM reasoning

**Expected performance**: Win rate ~15-25%, avg score 6-8 cards, assassin rate <5%

---

#### 5. **LLMSpymaster**

**Strategy**: Use a large language model (Qwen2.5-7B-Instruct) with zero-shot prompting to generate contextually aware clues.

**Prompt Structure**:

The prompt uses a multi-turn chat format with:

1. **System message**: Role definition and JSON output format specification
2. **Few-shot examples**: Two demonstration exchanges showing valid clue generation
3. **Current game state**: Board words categorized by color + explicit "DO NOT USE" list

**Complete prompt template**:
```
[SYSTEM]
You are a Spymaster in Codenames. Your team MUST win!

GAME OBJECTIVE:
Help your team identify ALL your team's words before the opponent finds theirs.
If your team reveals the ASSASSIN word, you IMMEDIATELY LOSE.

YOUR ROLE:
Give a ONE-WORD clue that connects to multiple team words while avoiding:
- ASSASSIN (instant loss)
- OPPONENT words (helps them win)
- NEUTRAL words (wastes guesses)

CLUE RULES:
(1) Must be ONE word only
(2) CANNOT be ANY word currently on the board
(3) Must contain only letters and numbers (no hyphens, no compound words)
(4) Count = how many team words your clue relates to

OUTPUT FORMAT:
Respond ONLY with valid JSON: {"clue": "word", "count": N}

Think strategically - connect multiple team words while staying far from danger!

[USER - Example 1]
TEAM_WORDS: cat, dog, mouse
OPPONENT_WORDS: car, tree, house
NEUTRAL_WORDS: sky, ocean
ASSASSIN: fire
ALL_BOARD_WORDS (DO NOT USE): cat, dog, mouse, car, tree, house, sky, ocean, fire

Give me a clue as JSON: {"clue": "word", "count": N}

[ASSISTANT - Example 1]
{"clue": "pets", "count": 3}

[USER - Example 2]
TEAM_WORDS: apple, banana, orange
OPPONENT_WORDS: desk, chair
NEUTRAL_WORDS: clock, lamp
ASSASSIN: knife
ALL_BOARD_WORDS (DO NOT USE): apple, banana, orange, desk, chair, clock, lamp, knife

Give me a clue as JSON: {"clue": "word", "count": N}

[ASSISTANT - Example 2]
{"clue": "fruit", "count": 3}

[USER - Actual Game]
TEAM_WORDS: <team words>
OPPONENT_WORDS: <opponent words>
NEUTRAL_WORDS: <neutral words>
ASSASSIN: <assassin word>
REVEALED: <revealed words> or (none)

ALL_BOARD_WORDS (DO NOT USE): <all board words>

Give me a clue as JSON: {"clue": "word", "count": N}
```

**Defense mechanisms** (to prevent invalid clues):
1. **Game rules in system message**: Explains objective, roles, and consequences (assassin = instant loss)
2. **Winning mindset**: "Your team MUST win!" creates motivation
3. **Explicit listing**: `ALL_BOARD_WORDS (DO NOT USE)` prominently displays forbidden words
4. **Few-shot examples**: Shows correct behavior (clues NOT on board)
5. **Token suppression**: `bad_words_ids` parameter blocks board words during generation
6. **Retry logic**: Up to 3 attempts with temperature sampling for different outputs
7. **Post-validation**: Checks clue validity, prints errors, retries on failure

**Strengths**:
- Context-aware reasoning (considers board state, word relationships)
- Multi-hop semantic connections (beyond simple embeddings)
- Natural language understanding
- Strategic thinking potential

**Weaknesses**:
- Slow (~10-30s per clue on CPU, ~2-5s on GPU)
- Requires ~14GB model download (first run only)
- Non-deterministic despite seed (due to floating-point ops on GPU)
- Can still generate invalid clues occasionally (hence retry logic)

**Expected performance**: Win rate ~30-50%, avg score 6-8 cards, assassin rate <10%

---

### Guesser Agents

The Guesser sees only the board words (not labels) and must identify team words based on the Spymaster's clue.

#### 3. **RandomGuesser**

**Strategy**: Randomly pick an unrevealed word or pass.

**How it works**:
- List all unrevealed words
- 60% chance: pick a random word
- 40% chance: pass (STOP)

**Strengths**:
- Fast
- Simple baseline

**Weaknesses**:
- Ignores the clue completely
- No reasoning about word relationships
- Often guesses opponent/neutral/assassin words

**Expected performance**: Helps team find ~4-5 cards (by luck)

---

#### 4. **EmbeddingsGuesser**

**Strategy**: Rank unrevealed words by semantic similarity to the clue, guess the most similar if above confidence threshold.

**How it works**:
1. Encode the clue and all unrevealed words
2. Compute cosine similarity between clue and each word
3. Find the best match
4. If similarity ≥ threshold (default 0.25): guess that word
5. Otherwise: pass (STOP)

**Strengths**:
- Actually uses the clue
- Conservative (stops when uncertain)
- Fast once embeddings are cached

**Weaknesses**:
- Only considers one guess at a time (doesn't look ahead)
- No memory of previous clues
- Limited by embedding quality
- Doesn't consider revealed word patterns

**Expected performance**: Helps team find ~5-7 cards

**Note on Cross-Encoders**: For the Guesser use case (1 clue × ~20 words = ~20 pairs), Cross-Encoders are more feasible (~1-2s per guess) and could provide better accuracy. A hybrid approach (Bi-Encoder retrieval + Cross-Encoder re-ranking) is implemented in `CrossEncoderGuesser`.

---

#### 5. **ContextualGuesser**

**Strategy**: Remember previous clues and game history to make better connections across multiple turns.

**Key improvement over EmbeddingsGuesser**:
- Maintains a history of all previous clues given in the game
- Tracks which guesses were correct/incorrect
- Uses history to boost words that relate to previous clues
- Penalizes words similar to known incorrect guesses

**How it works**:
1. Encode the current clue and all unrevealed words
2. Compute similarity to current clue (same as EmbeddingsGuesser)
3. **Add history boost**: Words similar to previous clues get a boost (weighted by history_weight, default 0.3)
4. **Apply wrong-guess penalty**: Words similar to previously incorrect guesses get penalized
5. Combine current similarity + history boost - wrong-guess penalty
6. Guess if total score ≥ threshold (default 0.25), otherwise pass

**Strengths**:
- Can connect clues across multiple turns
- Learns from mistakes (avoids words similar to wrong guesses)
- Better at following multi-turn strategies
- More context-aware than single-clue matching

**Weaknesses**:
- Requires maintaining state across turns
- History weighting may need tuning
- Still limited by embedding quality

---

#### 6. **AdaptiveGuesser**

**Strategy**: Dynamically adjust confidence threshold based on game state (ahead/behind, remaining guesses).

**Key improvement over EmbeddingsGuesser**:
- Adapts confidence threshold based on team's position in the game
- More aggressive when ahead, more conservative when behind
- Adjusts threshold for last guess in a turn

**How it works**:
1. Compute base similarity scores (same as EmbeddingsGuesser)
2. **Adaptive threshold calculation**:
   - If team is ahead (team_ratio > 0.6): lower threshold (more aggressive)
   - If team is behind (team_ratio < 0.4): higher threshold (more conservative)
   - If last guess remaining: increase threshold (be more confident)
3. Guess if similarity ≥ adaptive threshold, otherwise pass

**Strengths**:
- Responds to game state dynamically
- Can be more aggressive when safe to do so
- More conservative when behind to avoid mistakes

**Weaknesses**:
- May be too conservative in some situations
- Threshold adaptation logic may need tuning
- Doesn't use history like ContextualGuesser

---

#### 7. **LLMGuesser**

**Strategy**: Use a large language model (Qwen2.5-7B-Instruct) to interpret clues and select the most relevant word.

**Prompt Structure**:

Includes game rules, winning objective, and strategic guidance:

```
[SYSTEM]
You are a Guesser (Operative) in Codenames. Your team MUST win!

GAME OBJECTIVE:
Your Spymaster can see which words belong to your team. They gave you a CLUE to help you find your team's words.
You must identify ALL your team's words before the opponent finds theirs.

CRITICAL DANGER:
- If you guess the ASSASSIN word, your team IMMEDIATELY LOSES the game!
- If you guess an OPPONENT word, you help them win
- If you guess a NEUTRAL word, you waste your turn

YOUR STRATEGY:
1. The COUNT tells you how many words the clue connects to
2. You can make multiple guesses (up to COUNT + 1) if you keep guessing correctly
3. STOP when uncertain - it's better to pass than risk the assassin!
4. Your Spymaster is smart and avoids danger - trust their clue

OUTPUT FORMAT:
Respond ONLY with valid JSON:
- To guess a word: {"guess": "word"}
- To pass (stop): {"guess": "STOP"}

Think carefully - one wrong guess can lose the game!

[USER]
CLUE: <clue word>
COUNT: <expected number of words>
REMAINING_GUESSES: <how many guesses left this turn>
UNREVEALED_WORDS: <list of unrevealed words>
REVEALED_WORDS: <list of revealed words> or (none)

Which word should you guess? Respond with JSON: {"guess": "word"} or {"guess": "STOP"}
```

**Strengths**:
- Better semantic understanding than embeddings
- Can handle abstract/creative clues
- Considers revealed words as context
- Uses count hint to calibrate confidence

**Weaknesses**:
- Slow (~10-30s per guess on CPU, ~2-5s on GPU)
- No explicit memory of previous clues (though sees revealed words)
- Can guess invalid words (not on board, already revealed) - caught by validation

**Expected performance**: Helps team find ~6-8 cards

---

#### 8. **CrossEncoderGuesser**

**Strategy**: Use hybrid **Retrieve & Re-Rank** approach: Bi-Encoder for fast retrieval, Cross-Encoder for accurate re-ranking.

**Key improvement over EmbeddingsGuesser**:
- Cross-Encoders provide superior accuracy for semantic matching
- Better at understanding nuanced relationships between clues and words
- Hybrid approach maintains reasonable speed (~1-2s per guess)

**How it works**:
1. **Step 1 (Retrieval)**: Use Bi-Encoder to score clue against all unrevealed words (fast)
2. **Step 2 (Re-ranking)**: Use Cross-Encoder to re-rank top-10 candidates (accurate)
3. Guess if best Cross-Encoder score exceeds relative threshold, otherwise pass

**Strengths**:
- More accurate than pure Bi-Encoder
- Better semantic understanding of clue-word relationships
- Reasonable speed (~1-2s per guess)
- Conservative threshold prevents risky guesses

**Weaknesses**:
- Slower than pure Bi-Encoder (~1-2s vs ~0.1s per guess)
- Requires loading two models (Bi-Encoder + Cross-Encoder)
- Still single-clue matching (no history like ContextualGuesser)

**Expected performance**: Helps team find ~6-8 cards

---

## Performance Summary

Based on preliminary testing (10 games per configuration):

| Spymaster | Guesser | Win Rate | Avg Score | Assassin Rate | Notes |
|-----------|---------|----------|-----------|---------------|-------|
| Random | Random | 0% | 4.5/9 | 60% | Pure chaos, high danger |
| Random | Embeddings | 20% | 5.3/9 | 60% | Good guessing can't save bad clues |
| Embeddings | Random | 0% | 4.5/9 | 60% | Good clues wasted on random guesses |
| **Embeddings** | **Embeddings** | **10%** | **5.7/9** | **0%** | **Best baseline, safe play** |

**Key insights**:
- Random Spymaster is extremely dangerous (60% assassin rate)
- Embeddings Spymaster is much safer (0% assassin rate)
- Good guesser helps more than good spymaster (Random/Embeddings > Embeddings/Random)
- Overall win rates are low (~7.5% average), leaving plenty of room for LLM improvement

---

## Evaluation

Run full evaluations with:

```bash
# Single configuration
python scripts/run_eval.py \
    --wordlist configs/wordlist_en.txt \
    --vocabulary configs/vocabulary_en.txt \
    --spymaster embeddings \
    --guesser embeddings \
    --num-games 100

# Compare all baselines
python scripts/run_eval.py \
    --wordlist configs/wordlist_en.txt \
    --vocabulary configs/vocabulary_en.txt \
    --compare \
    --num-games 100
```

---

## Next Steps

These baselines will be used to:

1. **Generate training data**: Run thousands of games to create SFT datasets
2. **Establish benchmarks**: Measure LLM improvements against these results
3. **Create preference pairs**: Use simulated outcomes to build DPO datasets

Target for trained LLMs: **>30% win rate**, **<10% assassin rate**, **avg score >7/9**

