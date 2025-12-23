# Codenames Rules (Clean Summary)

This is a **plain-language summary** of the tabletop rules for **Codenames** (original team-vs-team word version).  
It is **not a verbatim copy** of the rulebook. For official wording, consult the publisher’s rulebook.

---

## 1) Objective

Two teams (Red and Blue) race to identify **all of their own agents** on a 5×5 grid of word cards.  
If your team reveals the **assassin**, you immediately lose.

---

## 2) Roles

- **Spymaster (one per team):** knows which words belong to which team (via a secret key card).
- **Operatives (the rest):** see only the 25 words; they guess their team’s agents based on the spymaster’s clue.

---

## 3) Setup

1. Lay out **25 word cards** in a **5×5** grid (all visible to everyone).
2. Put the **key card** in the stand so only spymasters can see it.
3. The key card indicates which words are:
   - Red agents
   - Blue agents
   - Innocent bystanders (neutral)
   - The assassin
4. The key card also indicates **which team goes first**. The starting team has **one extra agent** to find.

---

## 4) Turn Structure (High Level)

Teams alternate turns. On your team’s turn:

1. **Spymaster gives one clue:** **one word + one number**.
2. **Operatives make guesses** (at least one), one card at a time.
3. The turn ends when operatives stop voluntarily or make a wrong guess.

---

## 5) Giving a Clue

A clue has two parts:

- **Clue word (one word):** a single word intended to connect to one or more of your team’s unrevealed agent words.
- **Number:** how many words the clue is meant to connect.

### Special numbers (common variants in the official rules)
- **“Unlimited” clue:** instead of a number, the spymaster may say “unlimited”. Operatives may keep guessing as long as they keep being correct.
- **Zero clue:** a clue with **0** also allows unlimited guessing, but it generally signals “avoid words related to this clue.”

---

## 6) Guessing

Operatives discuss, then choose a word by indicating it (e.g., touching/pointing). The spymaster reveals its identity by covering it with the appropriate tile.

### If the guess is…
- **Your team’s agent:** you may guess again (no new clue).
- **Innocent bystander:** turn ends.
- **Opponent’s agent:** turn ends, and you have helped the other team.
- **Assassin:** immediate loss.

### “Plus one” rule
Operatives may guess up to **(number + 1)** cards, as long as they keep guessing correctly.

### Ending your turn
Operatives may **stop early** if further guesses feel too risky, but must make **at least one guess** each turn.

---

## 7) Winning

You win if:
- Your team reveals **all** of its agents (even if the last one gets revealed during the other team’s turn), **or**
- The other team reveals the **assassin**.

---

## 8) Clue Validity (Spirit of the Game)

Codenames expects clues to be about **meaning**, not about exploiting letter patterns or board position. If unsure, spymasters should resolve disputes quietly together.

### Common restrictions (paraphrased)
A clue is generally **not valid** if it:
- Refers to **letters**, **spelling patterns**, or **where a word sits** on the grid (rather than meaning).
- Is any **form** of a visible word (e.g., word + suffix/prefix), or a **part** of a visible compound word, while that word is still on the table.
- Uses accents/melodies or other “non-meaning” cues.

### Soundalikes
Traditional play typically allows soundalikes only when they connect by **meaning** (not merely by phonetic similarity). Many digital adaptations are more permissive, so decide what your group considers valid.

### Language and spelling
- Clues are normally given in the group’s play language; foreign words may be acceptable if commonly used by the group.
- You may be asked to **spell** your clue; identical spellings are treated as the same word.

### Practical guidance
Groups are encouraged not to be absurdly strict when the clue is clearly in the spirit of the game and both spymasters agree.

---

## 9) Practical Notes (Useful for Software / AI Implementations)

If you’re implementing Codenames or training agents, define your rule “strictness” explicitly (especially for soundalikes, morphology, and compound words), then keep it consistent.

Recommended logs/metrics:
- illegal clue rate
- assassin rate
- opponent/bystander hits
- average guesses per clue
- win rate vs fixed baselines

---

## Sources (for authoritative wording)

- Publisher rulebook PDF (Czech Games Edition)
- Publisher game page (Czech Games Edition)
