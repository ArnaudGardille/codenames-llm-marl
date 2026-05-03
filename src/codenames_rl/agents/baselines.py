"""Baseline agents for Codenames using embeddings and heuristics."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from ..env.spaces import GuesserAction, Observation, SpymasterAction
from ..env.validation import is_valid_clue
from ..utils.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    LLM_QUANTIZATION,
    DEVICE,
    EMBEDDING_MODEL,
)

# Conditional imports for LLM agents
if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed,
    )
    HAS_TORCH = True
    # Try to import bitsandbytes for quantization
    try:
        from transformers import BitsAndBytesConfig
        HAS_BITSANDBYTES = True
    except ImportError:
        HAS_BITSANDBYTES = False
except ImportError:
    HAS_TORCH = False
    HAS_BITSANDBYTES = False


def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a vector and a matrix of row-vectors."""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return np.dot(matrix_norm, vec_norm)


def _resolve_device(device: Optional[str], model: Optional[object] = None) -> str:
    """Resolve target device.

    Priority: explicit ``device`` arg > device of a shared ``model`` >
    config ``DEVICE`` > auto-detect (cuda → mps → cpu).
    """
    if device is not None:
        return device
    if model is not None:
        model_device = getattr(model, "device", None)
        if model_device is not None:
            return str(model_device)
        try:
            first_param = next(model.parameters(), None)
            if first_param is not None:
                return str(first_param.device)
        except Exception:
            pass
        return "cpu"
    if DEVICE and DEVICE != "cpu":
        return DEVICE
    if HAS_TORCH:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _build_quantization_config(quantization: str, device: str):
    """Return a ``BitsAndBytesConfig`` for 4bit/8bit, or ``None`` for no quantization.

    Raises ``ValueError`` if the requested mode is unsupported on the target device.
    """
    q = (quantization or "none").lower().replace("-", "")
    if q in ("none", ""):
        return None
    if q not in ("4bit", "8bit"):
        raise ValueError(f"Unknown quantization mode: {quantization!r}")
    if not HAS_BITSANDBYTES:
        raise ValueError(
            f"{q} quantization requires bitsandbytes (pip install bitsandbytes)"
        )
    if device != "cuda":
        raise ValueError(
            f"{q} quantization requires CUDA, got device={device!r}. "
            "On Apple Silicon (MPS), use float16 or a smaller model."
        )
    if q == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _load_causal_lm(model_name: str, device: str, quantization_config) -> "AutoModelForCausalLM":
    """Load an HF causal LM with the right dtype/device for the target.

    When ``quantization_config`` is given, it controls dtype and device-map.
    Otherwise we use float16 on cuda/mps and float32 on cpu.
    """
    kwargs = {"trust_remote_code": True}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = device
        kwargs["dtype"] = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


class BaseSpymaster(ABC):
    """Abstract base class for Spymaster agents."""

    @abstractmethod
    def get_clue(self, obs: Observation) -> SpymasterAction:
        """Generate a clue given the current observation.
        
        Args:
            obs: Current game observation
            
        Returns:
            SpymasterAction with clue word and count
        """
        pass

    def reset(self) -> None:
        """Reset agent state (optional, for stateful agents)."""
        pass


class BaseGuesser(ABC):
    """Abstract base class for Guesser agents."""

    @abstractmethod
    def get_guess(self, obs: Observation) -> GuesserAction:
        """Make a guess given the current observation.
        
        Args:
            obs: Current game observation
            
        Returns:
            GuesserAction with word_index or None (pass)
        """
        pass

    def reset(self) -> None:
        """Reset agent state (optional, for stateful agents)."""
        pass


class RandomSpymaster(BaseSpymaster):
    """Random baseline spymaster that gives random valid clues."""

    def __init__(
        self,
        vocabulary_path: str,
        seed: Optional[int] = None
    ):
        """Initialize random spymaster.
        
        Args:
            vocabulary_path: Path to vocabulary file for clue candidates
            seed: Random seed for reproducibility
        """
        self.vocabulary_path = vocabulary_path
        self.vocabulary = self._load_vocabulary()
        self.rng = np.random.default_rng(seed)

    def _load_vocabulary(self) -> List[str]:
        """Load vocabulary from file."""
        path = Path(self.vocabulary_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocabulary_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        return words

    def get_clue(self, obs: Observation) -> SpymasterAction:
        """Generate a random valid clue."""
        # Filter vocabulary to exclude board words
        board_lower = [w.lower() for w in obs.board_words]
        candidates = [w for w in self.vocabulary if w not in board_lower]
        
        if not candidates:
            # Fallback: use a generic word not on board
            candidates = ["thing", "stuff", "item", "object", "concept"]
            candidates = [w for w in candidates if w not in board_lower]
        
        # Pick random clue
        clue = str(self.rng.choice(candidates))
        count = int(self.rng.integers(1, min(obs.team_remaining, 3) + 1))
        
        return SpymasterAction(clue=clue, count=count)


class RandomGuesser(BaseGuesser):
    """Random baseline guesser that picks random unrevealed words."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize random guesser.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def get_guess(self, obs: Observation) -> GuesserAction:
        """Make a random guess from unrevealed words."""
        # Find unrevealed indices
        unrevealed = [i for i, revealed in enumerate(obs.revealed_mask) if not revealed]
        
        if not unrevealed:
            return GuesserAction(word_index=None)  # Pass
        
        # Randomly decide to pass or guess (60% guess, 40% pass)
        if self.rng.random() < 0.4:
            return GuesserAction(word_index=None)
        
        # Pick random unrevealed word
        word_idx = int(self.rng.choice(unrevealed))
        return GuesserAction(word_index=word_idx)


class EmbeddingsSpymaster(BaseSpymaster):
    """Spymaster using embeddings to score candidate clues.
    
    Scoring formula:
        score(clue) = mean_sim(clue, team_words) 
                      - alpha * max_sim(clue, assassin)
                      - beta * max_sim(clue, opponent_words)
                      - gamma * mean_sim(clue, neutral_words)
    """

    def __init__(
        self,
        vocabulary_path: str,
        model_name: str = EMBEDDING_MODEL,
        alpha: float = 3.0,  # Assassin penalty
        beta: float = 1.5,   # Opponent penalty
        gamma: float = 0.3,  # Neutral penalty
        similarity_threshold: float = 0.3,  # Min similarity to count toward clue number
        top_k: int = 100,    # Number of top candidates to consider
        seed: Optional[int] = None
    ):
        """Initialize embeddings-based spymaster.
        
        Args:
            vocabulary_path: Path to vocabulary file
            model_name: SentenceTransformer model name
            alpha: Penalty weight for assassin similarity
            beta: Penalty weight for opponent similarity
            gamma: Penalty weight for neutral similarity
            similarity_threshold: Minimum similarity to include in count
            top_k: Number of candidates to evaluate
            seed: Random seed for tie-breaking
        """
        self.vocabulary_path = vocabulary_path
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

        self._init_encoder(model_name)
        self.vocabulary = self._load_vocabulary()

    def _init_encoder(self, model_name: str) -> None:
        """Load the encoder. Subclasses override to use a different model class."""
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into a numpy array of embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)

    def _load_vocabulary(self) -> List[str]:
        """Load vocabulary from file."""
        path = Path(self.vocabulary_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocabulary_path}")

        with open(path, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]

        return words

    def get_clue(self, obs: Observation) -> SpymasterAction:
        """Generate clue using embedding-based scoring."""
        from ..env.spaces import CardColor

        team_words, opponent_words, neutral_words, assassin_words = [], [], [], []
        for word, color, revealed in zip(obs.board_words, obs.board_colors, obs.revealed_mask):
            if revealed:
                continue
            bucket = {
                CardColor.TEAM: team_words,
                CardColor.OPPONENT: opponent_words,
                CardColor.NEUTRAL: neutral_words,
                CardColor.ASSASSIN: assassin_words,
            }.get(color)
            if bucket is not None:
                bucket.append(word.lower())

        if not team_words:
            return SpymasterAction(clue="pass", count=0)

        board_lower = [w.lower() for w in obs.board_words]
        candidates = [w for w in self.vocabulary if w not in board_lower]

        if len(candidates) > self.top_k:
            candidates = self.rng.choice(candidates, size=self.top_k, replace=False).tolist()

        candidate_embs = self._encode(candidates)
        team_embs = self._encode(team_words)
        opponent_embs = self._encode(opponent_words) if opponent_words else None
        neutral_embs = self._encode(neutral_words) if neutral_words else None
        assassin_embs = self._encode(assassin_words) if assassin_words else None

        best_score = float('-inf')
        best_clue = None
        best_count = 1

        for candidate, cand_emb in zip(candidates, candidate_embs):
            is_valid, _ = is_valid_clue(candidate, obs.board_words)
            if not is_valid:
                continue

            team_sims = _cosine_similarity(cand_emb, team_embs)
            score = np.mean(team_sims)
            if assassin_embs is not None:
                score -= self.alpha * np.max(_cosine_similarity(cand_emb, assassin_embs))
            if opponent_embs is not None:
                score -= self.beta * np.max(_cosine_similarity(cand_emb, opponent_embs))
            if neutral_embs is not None:
                score -= self.gamma * np.mean(_cosine_similarity(cand_emb, neutral_embs))

            if score > best_score:
                best_score = score
                best_clue = candidate
                best_count = max(1, int(np.sum(team_sims >= self.similarity_threshold)))

        if best_clue is None:
            for word in candidates[:20]:
                if is_valid_clue(word, obs.board_words)[0]:
                    best_clue = word
                    break
            if best_clue is None:
                best_clue = "thing"
            best_count = 1

        return SpymasterAction(clue=best_clue, count=best_count)


class EmbeddingsGuesser(BaseGuesser):
    """Guesser using embeddings to rank words by similarity to clue."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        confidence_threshold: float = 0.25,  # Min similarity to guess
        seed: Optional[int] = None
    ):
        """Initialize embeddings-based guesser.
        
        Args:
            model_name: SentenceTransformer model name
            confidence_threshold: Minimum similarity to make a guess (else STOP)
            seed: Random seed for tie-breaking
        """
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)
        self._init_encoder(model_name)

    def _init_encoder(self, model_name: str) -> None:
        """Load the encoder. Subclasses override to use a different model class."""
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into a numpy array of embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)

    def get_guess(self, obs: Observation) -> GuesserAction:
        """Pick the unrevealed word most similar to the clue, or STOP below threshold."""
        if obs.current_clue is None:
            return GuesserAction(word_index=None)

        unrevealed_indices = [i for i, r in enumerate(obs.revealed_mask) if not r]
        if not unrevealed_indices:
            return GuesserAction(word_index=None)

        unrevealed_words = [obs.board_words[i].lower() for i in unrevealed_indices]
        clue_emb = self._encode([obs.current_clue.lower()])[0]
        word_embs = self._encode(unrevealed_words)
        similarities = _cosine_similarity(clue_emb, word_embs)

        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= self.confidence_threshold:
            return GuesserAction(word_index=unrevealed_indices[best_idx])
        return GuesserAction(word_index=None)


class _QwenEncoderMixin:
    """Encoder hook used by both Qwen embedding agents.

    Subclasses must set ``self.device`` before ``_init_encoder`` runs.
    """

    _QWEN_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def _init_encoder(self, model_name: str) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device
        )
        self.model.eval()

    def _encode(self, texts: List[str]) -> np.ndarray:
        formatted = [f"{self._QWEN_INSTRUCTION}{t}" for t in texts]
        inputs = self.tokenizer(
            formatted, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb.cpu().numpy()
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


class QwenEmbeddingSpymaster(_QwenEncoderMixin, EmbeddingsSpymaster):
    """Spymaster scored with Qwen3-Embedding-8B, same formula as EmbeddingsSpymaster."""

    def __init__(
        self,
        vocabulary_path: str,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        alpha: float = 3.0,
        beta: float = 1.5,
        gamma: float = 0.3,
        similarity_threshold: float = 0.3,
        top_k: int = 100,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if not HAS_TORCH:
            raise ImportError(
                "Qwen embedding agents require torch and transformers."
            )
        self.device = _resolve_device(device)
        super().__init__(
            vocabulary_path=vocabulary_path,
            model_name=model_name,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            seed=seed,
        )


class QwenEmbeddingGuesser(_QwenEncoderMixin, EmbeddingsGuesser):
    """Guesser scored with Qwen3-Embedding-8B; can share model with the spymaster."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if not HAS_TORCH:
            raise ImportError(
                "Qwen embedding agents require torch and transformers."
            )
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = _resolve_device(device, model)
        else:
            self.device = _resolve_device(device)
            self._init_encoder(model_name)


class LLMSpymaster(BaseSpymaster):
    """Spymaster using a language model for zero-shot clue generation.
    
    Uses chat-based prompting with strict JSON output parsing and validation.
    Supports quantization (4-bit/8-bit) for memory-efficient model loading.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        quantization: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize LLM-based spymaster.
        
        Args:
            model_name: HuggingFace model identifier (defaults to LLM_MODEL_NAME from config)
            device: Device to load model on (defaults to DEVICE from config, auto-detects if None)
            temperature: Sampling temperature for generation (defaults to LLM_TEMPERATURE from config)
            max_new_tokens: Maximum tokens to generate (defaults to LLM_MAX_NEW_TOKENS from config)
            quantization: Quantization mode - "none", "4bit", or "8bit" (defaults to LLM_QUANTIZATION from config)
            seed: Random seed for reproducibility
            
        Raises:
            ImportError: If torch or transformers are not installed
            ValueError: If quantization is requested but bitsandbytes is not available
        """
        if not HAS_TORCH:
            raise ImportError("LLM agents require torch and transformers.")

        self.model_name = model_name if model_name is not None else LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else LLM_MAX_NEW_TOKENS
        quantization = quantization if quantization is not None else LLM_QUANTIZATION
        self.seed = seed
        self.device = _resolve_device(device)

        if seed is not None:
            set_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        quantization_config = _build_quantization_config(quantization, self.device)
        self.model = _load_causal_lm(self.model_name, self.device, quantization_config)
        self.last_raw_output = None

    def get_clue(self, obs: Observation, max_retries: int = 3) -> SpymasterAction:
        """Generate clue using LLM with chat-based prompting and retry logic.
        
        Multiple defense mechanisms prevent invalid clues (words on the board):
        1. Explicit game rules and winning objective in system message
        2. "ALL_BOARD_WORDS (DO NOT USE)" line in prompt
        3. Few-shot examples showing valid clue generation
        4. Token-level suppression via bad_words_ids during generation
        5. Retry logic with temperature sampling for different outputs
        6. Post-generation validation and error feedback
        
        Args:
            obs: Game observation
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Valid SpymasterAction
            
        Raises:
            ValueError: If unable to generate valid clue after max_retries
        """
        from ..env.spaces import CardColor
        
        # Separate board words by color
        team_words = []
        opponent_words = []
        neutral_words = []
        assassin_words = []
        revealed_words = []
        
        for i, (word, color, revealed) in enumerate(
            zip(obs.board_words, obs.board_colors, obs.revealed_mask)
        ):
            if revealed:
                revealed_words.append(f"{word} ({color.value})")
            else:
                if color == CardColor.TEAM:
                    team_words.append(word)
                elif color == CardColor.OPPONENT:
                    opponent_words.append(word)
                elif color == CardColor.NEUTRAL:
                    neutral_words.append(word)
                elif color == CardColor.ASSASSIN:
                    assassin_words.append(word)
        
        # Attempt generation with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                # Build chat messages with game rules and winning objective
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
                
                # Few-shot examples showing correct behavior
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
                
                user_message = f"""TEAM_WORDS (find these): {', '.join(team_words) if team_words else '(none - all found!)'}
OPPONENT_WORDS (avoid these): {', '.join(opponent_words)}
NEUTRAL_WORDS (avoid these): {', '.join(neutral_words)}
ASSASSIN (DO NOT connect to this!): {', '.join(assassin_words)}
REVEALED (already found): {', '.join(revealed_words) if revealed_words else '(none)'}

⚠️  IMPORTANT: The REVEALED words are already found by your team. Give a NEW clue for the remaining TEAM_WORDS above.
Do NOT repeat clues for words that are already REVEALED!

ALL_BOARD_WORDS (DO NOT USE as clue): {', '.join(obs.board_words)}

Give me a clue as JSON: {{"clue": "<your_clue_here>", "count": <number>}}
The count should match how many of the CURRENT TEAM_WORDS (not revealed ones) your clue connects to."""
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": example1_user},
                    {"role": "assistant", "content": example1_assistant},
                    {"role": "user", "content": example2_user},
                    {"role": "assistant", "content": example2_assistant},
                    {"role": "user", "content": user_message}
                ]
                
                # Generate
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Check prompt length vs model context window
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_length = inputs['input_ids'].shape[1]
                model_max_length = getattr(self.tokenizer, 'model_max_length', 32768)  # Qwen2.5 default is 32K
                
                if input_length > model_max_length * 0.9:  # Warn if using >90% of context
                    print(f"⚠️  Warning: Prompt is {input_length}/{model_max_length} tokens ({input_length/model_max_length*100:.1f}% of context window)")
                elif input_length > model_max_length:
                    raise ValueError(
                        f"Prompt too long: {input_length} tokens exceeds model max length {model_max_length}. "
                        f"Consider reducing few-shot examples or board word lists."
                    )
                
                # Create logit processor to suppress board words
                board_token_ids = self._get_board_token_ids(obs.board_words)
                
                with torch.no_grad():
                    # Add randomness to seed for each generation to ensure variation
                    import random
                    import time
                    # Use combination of seed, attempt number, and time for variation
                    generation_seed = (hash(str(self.seed) + str(attempt) + str(time.time())) % (2**31)) if self.seed is not None else None
                    if generation_seed is not None:
                        set_seed(generation_seed)
                    
                    # Slightly increase temperature on retries for more variation
                    current_temp = self.temperature * (1.0 + attempt * 0.1)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=current_temp,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        bad_words_ids=board_token_ids if board_token_ids else None
                    )
                
                # Decode output
                generated = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                self.last_raw_output = generated
                
                # Parse JSON
                clue, count = self._parse_spymaster_output(generated, obs.board_words)
                
                # Validate
                is_valid, error_msg = is_valid_clue(clue, obs.board_words)
                if not is_valid:
                    last_error = f"{error_msg}. Output: {generated}"
                    print(f"✗ Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                    continue
                
                # Success!
                if attempt > 0:
                    print(f"✓ Succeeded on attempt {attempt + 1}")
                return SpymasterAction(clue=clue, count=count)
                
            except Exception as e:
                last_error = str(e)
                print(f"✗ Attempt {attempt + 1}/{max_retries} error: {e}")
                continue
        
        # All retries exhausted
        raise ValueError(f"LLM failed to generate valid clue after {max_retries} attempts. Last error: {last_error}")

    def _get_board_token_ids(self, board_words: List[str]) -> List[List[int]]:
        """Get token IDs for board words to suppress during generation.
        
        Args:
            board_words: List of words on the board
            
        Returns:
            List of token ID sequences to suppress (for bad_words_ids parameter)
        """
        bad_token_ids = []
        for word in board_words:
            # Try different variations: lowercase, uppercase, title case
            for variant in [word.lower(), word.upper(), word.title(), word]:
                # Tokenize with and without leading space
                for text in [variant, f" {variant}"]:
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if token_ids:
                        bad_token_ids.append(token_ids)
        return bad_token_ids if bad_token_ids else None

    def _parse_spymaster_output(self, output: str, board_words: List[str]) -> tuple[str, int]:
        """Parse LLM output to extract clue and count.
        
        Args:
            output: Raw LLM output
            board_words: Board words for validation
            
        Returns:
            Tuple of (clue, count)
            
        Raises:
            ValueError: If parsing fails or output is invalid
        """
        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"clue"[^}]*"count"[^}]*\}', output, re.IGNORECASE)
        if not json_match:
            raise ValueError(f"Could not find JSON in output: {output}")
        
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in output: {output}. Error: {e}")
        
        # Extract fields
        if "clue" not in data or "count" not in data:
            raise ValueError(f"Missing 'clue' or 'count' in JSON: {data}")
        
        clue = str(data["clue"]).strip()
        count = int(data["count"])
        
        if not clue:
            raise ValueError("Empty clue in output")
        
        if count < 0:
            raise ValueError(f"Invalid count: {count}")
        
        return clue, count


class LLMGuesser(BaseGuesser):
    """Guesser using a language model for zero-shot guessing.
    
    Uses chat-based prompting with strict JSON output parsing and validation.
    Supports quantization (4-bit/8-bit) for memory-efficient model loading.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        quantization: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize LLM-based guesser.
        
        Args:
            model_name: HuggingFace model identifier (defaults to LLM_MODEL_NAME from config)
            model: Pre-loaded model (optional, for sharing with spymaster)
            tokenizer: Pre-loaded tokenizer (optional)
            device: Device to load model on (defaults to DEVICE from config, auto-detects if None)
            temperature: Sampling temperature for generation (defaults to LLM_TEMPERATURE from config)
            max_new_tokens: Maximum tokens to generate (defaults to LLM_MAX_NEW_TOKENS from config)
            quantization: Quantization mode - "none", "4bit", or "8bit" (defaults to LLM_QUANTIZATION from config)
            seed: Random seed for reproducibility
            
        Raises:
            ImportError: If torch or transformers are not installed
            ValueError: If quantization is requested but bitsandbytes is not available
        """
        if not HAS_TORCH:
            raise ImportError("LLM agents require torch and transformers.")

        self.model_name = model_name if model_name is not None else LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else LLM_MAX_NEW_TOKENS
        quantization = quantization if quantization is not None else LLM_QUANTIZATION
        self.seed = seed
        self.device = _resolve_device(device, model)

        if seed is not None:
            set_seed(seed)

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            quantization_config = _build_quantization_config(quantization, self.device)
            self.model = _load_causal_lm(self.model_name, self.device, quantization_config)

        self.last_raw_output = None

    def get_guess(self, obs: Observation, max_retries: int = 3) -> GuesserAction:
        """Make a guess using LLM with chat-based prompting and retry logic.
        
        Args:
            obs: Game observation
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Valid GuesserAction
            
        Raises:
            ValueError: If unable to generate valid guess after max_retries
        """
        if obs.current_clue is None:
            return GuesserAction(word_index=None)
        
        # Find unrevealed words
        unrevealed_words = [
            word for i, (word, revealed) in enumerate(
                zip(obs.board_words, obs.revealed_mask)
            ) if not revealed
        ]
        
        if not unrevealed_words:
            return GuesserAction(word_index=None)
        
        # Build revealed list with types
        revealed_list = []
        for i, (word, revealed) in enumerate(zip(obs.board_words, obs.revealed_mask)):
            if revealed:
                # Note: Guesser doesn't see colors, but sees which were revealed
                revealed_list.append(word)
        
        # Attempt generation with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                # Build chat messages with game rules and winning objective
                system_message = (
                    "You are a Guesser (Operative) in Codenames. Your team MUST win!\n\n"
                    "GAME OBJECTIVE:\n"
                    "Your Spymaster can see which words belong to your team. They gave you a CLUE to help you find your team's words. "
                    "You must identify ALL your team's words before the opponent finds theirs.\n\n"
                    "CRITICAL DANGER - READ CAREFULLY:\n"
                    "- ⚠️  If you guess the ASSASSIN word, your team IMMEDIATELY LOSES THE ENTIRE GAME!\n"
                    "- The ASSASSIN word is hidden among the unrevealed words - you don't know which one it is!\n"
                    "- If you guess an OPPONENT word, you help them win\n"
                    "- If you guess a NEUTRAL word, you waste your turn\n\n"
                    "YOUR STRATEGY:\n"
                    "1. The COUNT tells you how many words the clue connects to\n"
                    "2. You can make multiple guesses (up to COUNT + 1) if you keep guessing correctly\n"
                    "3. ⚠️  BE CONSERVATIVE: STOP when uncertain - it's MUCH better to pass than risk the assassin!\n"
                    "4. Your Spymaster is smart and avoids danger - trust their clue, but don't be reckless\n"
                    "5. NEVER guess words that are already REVEALED - only guess from UNREVEALED_WORDS!\n"
                    "6. If you're not confident about a word matching the clue, STOP instead of guessing!\n\n"
                    "OUTPUT FORMAT:\n"
                    "Respond ONLY with valid JSON:\n"
                    "- To guess a word: {\"guess\": \"<word_from_board>\"}\n"
                    "- To pass (stop): {\"guess\": \"STOP\"}\n\n"
                    "Think carefully - one wrong guess can lose the game!"
                )
                
                user_message = f"""CLUE: {obs.current_clue}
COUNT: {obs.current_count}
REMAINING_GUESSES: {obs.remaining_guesses}

⚠️  WARNING: One of the UNREVEALED_WORDS below is the ASSASSIN. If you guess it, you LOSE immediately!
Be conservative - if you're not confident, choose STOP instead of guessing.

UNREVEALED_WORDS (one of these is the ASSASSIN - be careful!): {', '.join(unrevealed_words)}
REVEALED_WORDS (DO NOT guess these - already revealed): {', '.join(revealed_list) if revealed_list else '(none)'}

Which word should you guess? Only guess if you're confident it matches the clue. Otherwise, choose STOP.
Respond with JSON: {{"guess": "<word_from_unrevealed_only>"}} or {{"guess": "STOP"}}"""
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                # Generate
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Check prompt length vs model context window
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_length = inputs['input_ids'].shape[1]
                model_max_length = getattr(self.tokenizer, 'model_max_length', 32768)  # Qwen2.5 default is 32K
                
                if input_length > model_max_length * 0.9:  # Warn if using >90% of context
                    print(f"⚠️  Warning: Prompt is {input_length}/{model_max_length} tokens ({input_length/model_max_length*100:.1f}% of context window)")
                elif input_length > model_max_length:
                    raise ValueError(
                        f"Prompt too long: {input_length} tokens exceeds model max length {model_max_length}. "
                        f"Consider reducing few-shot examples or board word lists."
                    )
                
                with torch.no_grad():
                    # Add randomness to seed for each generation to ensure variation
                    import random
                    import time
                    # Use combination of seed, attempt number, and time for variation
                    generation_seed = (hash(str(self.seed) + str(attempt) + str(time.time())) % (2**31)) if self.seed is not None else None
                    if generation_seed is not None:
                        set_seed(generation_seed)
                    
                    # Slightly increase temperature on retries for more variation
                    current_temp = self.temperature * (1.0 + attempt * 0.1)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=current_temp,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode output
                generated = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                self.last_raw_output = generated
                
                # Parse JSON
                guess_word = self._parse_guesser_output(generated)
                
                # Handle STOP
                if guess_word.upper() == "STOP":
                    return GuesserAction(word_index=None)
                
                # Find word index
                try:
                    word_index = obs.board_words.index(guess_word)
                except ValueError:
                    # Try case-insensitive match
                    board_lower = [w.lower() for w in obs.board_words]
                    try:
                        word_index = board_lower.index(guess_word.lower())
                    except ValueError:
                        error_msg = f"LLM guessed word '{guess_word}' not on board. Output: {generated}"
                        last_error = error_msg
                        print(f"✗ Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                        continue
                
                # Verify not revealed - if already revealed, return STOP instead of crashing
                if obs.revealed_mask[word_index]:
                    # Word already revealed, guesser should pass
                    # This can happen if the LLM doesn't see the updated observation
                    return GuesserAction(word_index=None)
                
                # Success!
                if attempt > 0:
                    print(f"✓ Succeeded on attempt {attempt + 1}")
                return GuesserAction(word_index=word_index)
                
            except Exception as e:
                last_error = str(e)
                print(f"✗ Attempt {attempt + 1}/{max_retries} error: {e}")
                continue
        
        # All retries exhausted
        raise ValueError(f"LLM failed to generate valid guess after {max_retries} attempts. Last error: {last_error}")

    def _parse_guesser_output(self, output: str) -> str:
        """Parse LLM output to extract guess.
        
        Args:
            output: Raw LLM output
            
        Returns:
            Guessed word or "STOP"
            
        Raises:
            ValueError: If parsing fails
        """
        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"guess"[^}]*\}', output, re.IGNORECASE)
        if not json_match:
            raise ValueError(f"Could not find JSON in output: {output}")
        
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in output: {output}. Error: {e}")
        
        # Extract guess
        if "guess" not in data:
            raise ValueError(f"Missing 'guess' in JSON: {data}")
        
        guess = str(data["guess"]).strip()
        
        if not guess:
            raise ValueError("Empty guess in output")
        
        return guess
