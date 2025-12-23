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
        PreTrainedTokenizerFast,
        set_seed
    )
    from huggingface_hub import hf_hub_download
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


def _load_tokenizer_with_fallback(model_name: str, trust_remote_code: bool = True):
    """Load tokenizer with fallback for models with custom tokenizer backends.
    
    Some models (like Ministral) use custom tokenizer backends (e.g., TokenizersBackend)
    that aren't recognized by transformers' AutoTokenizer. This function first tries
    the standard AutoTokenizer, and if that fails, falls back to loading the tokenizer
    directly using PreTrainedTokenizerFast.
    
    Args:
        model_name: The model name or path
        trust_remote_code: Whether to trust remote code (for custom tokenizers)
        
    Returns:
        A tokenizer instance
        
    Raises:
        ValueError: If the tokenizer cannot be loaded with either method
    """
    try:
        # Try standard AutoTokenizer first
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except ValueError as e:
        if "does not exist or is not currently imported" in str(e):
            # Fallback for custom tokenizer backends
            try:
                tokenizer_file = hf_hub_download(model_name, 'tokenizer.json')
                tokenizer_config_file = hf_hub_download(model_name, 'tokenizer_config.json')
                
                # Load config to get special tokens
                with open(tokenizer_config_file, 'r') as f:
                    config = json.load(f)
                
                # Create tokenizer from the json file
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
                
                # Set special tokens from config
                tokenizer.pad_token = config.get('pad_token')
                tokenizer.eos_token = config.get('eos_token', '</s>')
                tokenizer.bos_token = config.get('bos_token', '<s>')
                tokenizer.unk_token = config.get('unk_token', '<unk>')
                
                # Set model max length
                if 'model_max_length' in config:
                    tokenizer.model_max_length = config['model_max_length']
                
                return tokenizer
            except Exception as fallback_error:
                raise ValueError(
                    f"Failed to load tokenizer for {model_name}. "
                    f"AutoTokenizer error: {e}. "
                    f"Fallback error: {fallback_error}"
                )
        else:
            raise


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
        
        # Load model
        self.model = SentenceTransformer(model_name)
        self.vocabulary = self._load_vocabulary()

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
        
        # Separate board words by color
        team_words = []
        opponent_words = []
        neutral_words = []
        assassin_words = []
        
        for i, (word, color, revealed) in enumerate(
            zip(obs.board_words, obs.board_colors, obs.revealed_mask)
        ):
            if revealed:
                continue
            if color == CardColor.TEAM:
                team_words.append(word.lower())
            elif color == CardColor.OPPONENT:
                opponent_words.append(word.lower())
            elif color == CardColor.NEUTRAL:
                neutral_words.append(word.lower())
            elif color == CardColor.ASSASSIN:
                assassin_words.append(word.lower())
        
        if not team_words:
            # No team words left (shouldn't happen, but safety)
            return SpymasterAction(clue="pass", count=0)
        
        # Filter vocabulary to exclude board words
        board_lower = [w.lower() for w in obs.board_words]
        candidates = [w for w in self.vocabulary if w not in board_lower]
        
        # Sample candidates if too many
        if len(candidates) > self.top_k:
            candidates = self.rng.choice(candidates, size=self.top_k, replace=False).tolist()
        
        # Encode everything
        candidate_embs = self.model.encode(candidates, convert_to_numpy=True)
        team_embs = self.model.encode(team_words, convert_to_numpy=True)
        
        opponent_embs = (
            self.model.encode(opponent_words, convert_to_numpy=True)
            if opponent_words else None
        )
        neutral_embs = (
            self.model.encode(neutral_words, convert_to_numpy=True)
            if neutral_words else None
        )
        assassin_embs = (
            self.model.encode(assassin_words, convert_to_numpy=True)
            if assassin_words else None
        )
        
        # Score each candidate
        best_score = float('-inf')
        best_clue = None
        best_count = 1
        
        for i, (candidate, cand_emb) in enumerate(zip(candidates, candidate_embs)):
            # Validate clue
            is_valid, _ = is_valid_clue(candidate, obs.board_words)
            if not is_valid:
                continue
            
            # Compute similarities
            team_sims = self._cosine_similarity(cand_emb, team_embs)
            team_mean = np.mean(team_sims)
            
            # Penalties
            assassin_penalty = 0.0
            if assassin_embs is not None:
                assassin_sims = self._cosine_similarity(cand_emb, assassin_embs)
                assassin_penalty = self.alpha * np.max(assassin_sims)
            
            opponent_penalty = 0.0
            if opponent_embs is not None:
                opponent_sims = self._cosine_similarity(cand_emb, opponent_embs)
                opponent_penalty = self.beta * np.max(opponent_sims)
            
            neutral_penalty = 0.0
            if neutral_embs is not None:
                neutral_sims = self._cosine_similarity(cand_emb, neutral_embs)
                neutral_penalty = self.gamma * np.mean(neutral_sims)
            
            # Final score
            score = team_mean - assassin_penalty - opponent_penalty - neutral_penalty
            
            if score > best_score:
                best_score = score
                best_clue = candidate
                # Count: how many team words have similarity above threshold
                best_count = max(1, int(np.sum(team_sims >= self.similarity_threshold)))
        
        if best_clue is None:
            # Fallback: pick first valid word
            for word in candidates[:20]:
                is_valid, _ = is_valid_clue(word, obs.board_words)
                if is_valid:
                    best_clue = word
                    best_count = 1
                    break
            
            if best_clue is None:
                best_clue = "thing"
                best_count = 1
        
        return SpymasterAction(clue=best_clue, count=best_count)

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return np.dot(matrix_norm, vec_norm)


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
        
        # Load model
        self.model = SentenceTransformer(model_name)

    def get_guess(self, obs: Observation) -> GuesserAction:
        """Make a guess by ranking unrevealed words by clue similarity."""
        if obs.current_clue is None:
            # No active clue (shouldn't happen during guesser turn)
            return GuesserAction(word_index=None)
        
        # Find unrevealed words
        unrevealed_indices = [
            i for i, revealed in enumerate(obs.revealed_mask) if not revealed
        ]
        
        if not unrevealed_indices:
            return GuesserAction(word_index=None)
        
        unrevealed_words = [obs.board_words[i].lower() for i in unrevealed_indices]
        
        # Encode clue and words
        clue_emb = self.model.encode([obs.current_clue.lower()], convert_to_numpy=True)[0]
        word_embs = self.model.encode(unrevealed_words, convert_to_numpy=True)
        
        # Compute similarities
        similarities = self._cosine_similarity(clue_emb, word_embs)
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        # Decide: guess or pass
        if best_sim >= self.confidence_threshold:
            return GuesserAction(word_index=unrevealed_indices[best_idx])
        else:
            return GuesserAction(word_index=None)  # STOP

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return np.dot(matrix_norm, vec_norm)


class QwenEmbeddingSpymaster(BaseSpymaster):
    """Spymaster using Qwen3-Embedding-8B for better semantic understanding.
    
    Uses a large embedding model (8B parameters) that provides better semantic
    understanding than small sentence-transformers models, while being faster
    and more efficient than full LLMs.
    
    Scoring formula:
        score(clue) = mean_sim(clue, team_words) 
                      - alpha * max_sim(clue, assassin)
                      - beta * max_sim(clue, opponent_words)
                      - gamma * mean_sim(clue, neutral_words)
    """

    def __init__(
        self,
        vocabulary_path: str,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        alpha: float = 3.0,  # Assassin penalty
        beta: float = 1.5,   # Opponent penalty
        gamma: float = 0.3,  # Neutral penalty
        similarity_threshold: float = 0.3,  # Min similarity to count toward clue number
        top_k: int = 100,    # Number of top candidates to consider
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize Qwen embedding-based spymaster.
        
        Args:
            vocabulary_path: Path to vocabulary file
            model_name: HuggingFace model identifier for Qwen embedding model
            alpha: Penalty weight for assassin similarity
            beta: Penalty weight for opponent similarity
            gamma: Penalty weight for neutral similarity
            similarity_threshold: Minimum similarity to include in count
            top_k: Number of candidates to evaluate
            device: Device to load model on (auto-detects if None)
            seed: Random seed for tie-breaking
            
        Raises:
            ImportError: If torch or transformers are not installed
        """
        if not HAS_TORCH:
            raise ImportError(
                "Qwen embedding agents require torch and transformers. "
                "Install them with: pip install torch transformers"
            )
        
        self.vocabulary_path = vocabulary_path
        self.model_name = model_name
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)
        
        # Auto-detect device
        if device is None:
            if DEVICE != "cpu":
                self.device = DEVICE
            elif torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine dtype based on device
        if self.device == "cuda":
            dtype = torch.float16
        elif self.device == "mps":
            dtype = torch.float16  # Use FP16 to reduce memory usage
        else:
            dtype = torch.float32
        
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device
        )
        self.model.eval()
        
        self.vocabulary = self._load_vocabulary()

    def _load_vocabulary(self) -> List[str]:
        """Load vocabulary from file."""
        path = Path(self.vocabulary_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocabulary_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        return words

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3-Embedding model.
        
        Qwen3-Embedding models use instruction-aware inputs. We format
        the input with an instruction prefix for better embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        # Qwen3-Embedding uses instruction-aware format
        # Format: "Represent this sentence for searching relevant passages: {text}"
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_texts = [f"{instruction}{text}" for text in texts]
        
        # Tokenize and encode
        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Extract embeddings from last hidden state
            # For Qwen3-Embedding, we typically use mean pooling or the [CLS] token
            # Check if model has pooler_output, otherwise use mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to numpy and normalize
        embeddings = embeddings.cpu().numpy()
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings

    def get_clue(self, obs: Observation) -> SpymasterAction:
        """Generate clue using Qwen embedding-based scoring."""
        from ..env.spaces import CardColor
        
        # Separate board words by color
        team_words = []
        opponent_words = []
        neutral_words = []
        assassin_words = []
        
        for i, (word, color, revealed) in enumerate(
            zip(obs.board_words, obs.board_colors, obs.revealed_mask)
        ):
            if revealed:
                continue
            if color == CardColor.TEAM:
                team_words.append(word.lower())
            elif color == CardColor.OPPONENT:
                opponent_words.append(word.lower())
            elif color == CardColor.NEUTRAL:
                neutral_words.append(word.lower())
            elif color == CardColor.ASSASSIN:
                assassin_words.append(word.lower())
        
        if not team_words:
            # No team words left (shouldn't happen, but safety)
            return SpymasterAction(clue="pass", count=0)
        
        # Filter vocabulary to exclude board words
        board_lower = [w.lower() for w in obs.board_words]
        candidates = [w for w in self.vocabulary if w not in board_lower]
        
        # Sample candidates if too many
        if len(candidates) > self.top_k:
            candidates = self.rng.choice(candidates, size=self.top_k, replace=False).tolist()
        
        # Encode everything
        candidate_embs = self._encode(candidates)
        team_embs = self._encode(team_words)
        
        opponent_embs = (
            self._encode(opponent_words)
            if opponent_words else None
        )
        neutral_embs = (
            self._encode(neutral_words)
            if neutral_words else None
        )
        assassin_embs = (
            self._encode(assassin_words)
            if assassin_words else None
        )
        
        # Score each candidate
        best_score = float('-inf')
        best_clue = None
        best_count = 1
        
        for i, (candidate, cand_emb) in enumerate(zip(candidates, candidate_embs)):
            # Validate clue
            is_valid, _ = is_valid_clue(candidate, obs.board_words)
            if not is_valid:
                continue
            
            # Compute similarities
            team_sims = self._cosine_similarity(cand_emb, team_embs)
            team_mean = np.mean(team_sims)
            
            # Penalties
            assassin_penalty = 0.0
            if assassin_embs is not None:
                assassin_sims = self._cosine_similarity(cand_emb, assassin_embs)
                assassin_penalty = self.alpha * np.max(assassin_sims)
            
            opponent_penalty = 0.0
            if opponent_embs is not None:
                opponent_sims = self._cosine_similarity(cand_emb, opponent_embs)
                opponent_penalty = self.beta * np.max(opponent_sims)
            
            neutral_penalty = 0.0
            if neutral_embs is not None:
                neutral_sims = self._cosine_similarity(cand_emb, neutral_embs)
                neutral_penalty = self.gamma * np.mean(neutral_sims)
            
            # Final score
            score = team_mean - assassin_penalty - opponent_penalty - neutral_penalty
            
            if score > best_score:
                best_score = score
                best_clue = candidate
                # Count: how many team words have similarity above threshold
                best_count = max(1, int(np.sum(team_sims >= self.similarity_threshold)))
        
        if best_clue is None:
            # Fallback: pick first valid word
            for word in candidates[:20]:
                is_valid, _ = is_valid_clue(word, obs.board_words)
                if is_valid:
                    best_clue = word
                    best_count = 1
                    break
            
            if best_clue is None:
                best_clue = "thing"
                best_count = 1
        
        return SpymasterAction(clue=best_clue, count=best_count)

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return np.dot(matrix_norm, vec_norm)


class QwenEmbeddingGuesser(BaseGuesser):
    """Guesser using Qwen3-Embedding-8B to rank words by similarity to clue."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        confidence_threshold: float = 0.25,  # Min similarity to guess
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize Qwen embedding-based guesser.
        
        Args:
            model_name: HuggingFace model identifier (defaults to Qwen/Qwen3-Embedding-8B)
            model: Pre-loaded model (optional, for sharing with spymaster)
            tokenizer: Pre-loaded tokenizer (optional)
            confidence_threshold: Minimum similarity to make a guess (else STOP)
            device: Device to load model on (auto-detects if None, or uses model's device)
            seed: Random seed for tie-breaking
            
        Raises:
            ImportError: If torch or transformers are not installed
        """
        if not HAS_TORCH:
            raise ImportError(
                "Qwen embedding agents require torch and transformers. "
                "Install them with: pip install torch transformers"
            )
        
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)
        
        # Use provided model/tokenizer or load new ones
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            # Use device from parameter, or try to get from model, or default to cpu
            if device is not None:
                self.device = device
            else:
                # Try to get device from model
                model_device = getattr(model, "device", None)
                if model_device is not None:
                    if isinstance(model_device, torch.device):
                        self.device = str(model_device)
                    else:
                        self.device = model_device
                else:
                    # Try to get device from first parameter
                    try:
                        first_param = next(model.parameters(), None)
                        if first_param is not None:
                            self.device = str(first_param.device)
                        else:
                            self.device = "cpu"
                    except:
                        self.device = "cpu"
        else:
            self.model_name = model_name
            
            # Auto-detect device
            if device is None:
                if DEVICE != "cpu":
                    self.device = DEVICE
                elif torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"  # Apple Silicon GPU
                else:
                    self.device = "cpu"
            else:
                self.device = device
            
            # Load model and tokenizer
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Determine dtype based on device
            if self.device == "cuda":
                dtype = torch.float16
            elif self.device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.device
            )
            self.model.eval()

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3-Embedding model.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        # Qwen3-Embedding uses instruction-aware format
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_texts = [f"{instruction}{text}" for text in texts]
        
        # Tokenize and encode
        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Extract embeddings
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to numpy and normalize
        embeddings = embeddings.cpu().numpy()
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings

    def get_guess(self, obs: Observation) -> GuesserAction:
        """Make a guess by ranking unrevealed words by clue similarity."""
        if obs.current_clue is None:
            # No active clue (shouldn't happen during guesser turn)
            return GuesserAction(word_index=None)
        
        # Find unrevealed words
        unrevealed_indices = [
            i for i, revealed in enumerate(obs.revealed_mask) if not revealed
        ]
        
        if not unrevealed_indices:
            return GuesserAction(word_index=None)
        
        unrevealed_words = [obs.board_words[i].lower() for i in unrevealed_indices]
        
        # Encode clue and words
        clue_emb = self._encode([obs.current_clue.lower()])[0]
        word_embs = self._encode(unrevealed_words)
        
        # Compute similarities
        similarities = self._cosine_similarity(clue_emb, word_embs)
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        # Decide: guess or pass
        if best_sim >= self.confidence_threshold:
            return GuesserAction(word_index=unrevealed_indices[best_idx])
        else:
            return GuesserAction(word_index=None)  # STOP

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return np.dot(matrix_norm, vec_norm)


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
            raise ImportError(
                "LLM agents require torch and transformers. "
                "Install them with: pip install torch transformers"
            )
        
        # Use config defaults if not provided
        self.model_name = model_name if model_name is not None else LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else LLM_MAX_NEW_TOKENS
        quantization = quantization if quantization is not None else LLM_QUANTIZATION
        self.seed = seed
        
        # Auto-detect device
        if device is None:
            if DEVICE != "cpu":
                self.device = DEVICE
            elif torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
        
        # Load model and tokenizer
        self.tokenizer = _load_tokenizer_with_fallback(self.model_name)
        
        # Setup quantization if requested
        quantization_config = None
        if quantization.lower() in ("4bit", "4-bit"):
            if not HAS_BITSANDBYTES:
                raise ValueError(
                    "4-bit quantization requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )
            if self.device != "cuda":
                # bitsandbytes only supports CUDA, not MPS
                if self.device == "mps":
                    raise ValueError(
                        "4-bit quantization with bitsandbytes is not supported on Apple Silicon (MPS).\n"
                        "Alternatives for Apple Silicon:\n"
                        "  1. Use float16 (default on MPS) - already memory efficient\n"
                        "  2. Use a smaller model (e.g., Qwen2.5-3B-Instruct instead of 7B)\n"
                        "  3. Use MLX framework with 4-bit quantization (requires different code path)\n"
                        "  4. Use GGML/GGUF format with llama.cpp\n"
                        "\n"
                        "For now, set LLM_QUANTIZATION=none and use float16 (automatic on MPS)"
                    )
                else:
                    raise ValueError(
                        f"4-bit quantization only works on CUDA devices, not {self.device}"
                    )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization.lower() in ("8bit", "8-bit"):
            if not HAS_BITSANDBYTES:
                raise ValueError(
                    "8-bit quantization requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )
            if self.device != "cuda":
                if self.device == "mps":
                    raise ValueError(
                        "8-bit quantization with bitsandbytes is not supported on Apple Silicon (MPS).\n"
                        "Alternatives for Apple Silicon:\n"
                        "  1. Use float16 (default on MPS) - already memory efficient\n"
                        "  2. Use a smaller model (e.g., Qwen2.5-3B-Instruct instead of 7B)\n"
                        "\n"
                        "For now, set LLM_QUANTIZATION=none and use float16 (automatic on MPS)"
                    )
                else:
                    raise ValueError(
                        f"8-bit quantization only works on CUDA devices, not {self.device}"
                    )
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Determine dtype based on device (only if not using quantization)
        if quantization_config is None:
            if self.device == "cuda":
                dtype = torch.float16
            elif self.device == "mps":
                dtype = torch.float16  # Use FP16 to reduce memory usage (~14GB vs ~28GB in FP32)
            else:
                dtype = torch.float32
        else:
            dtype = None  # Quantization config handles dtype
        
        # Load model with appropriate configuration
        model_kwargs = {
            "device_map": self.device if quantization_config is None else "auto",
            "trust_remote_code": True,  # Required for some newer models like Ministral
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()
        
        # Store last raw LLM output for debugging
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
            raise ImportError(
                "LLM agents require torch and transformers. "
                "Install them with: pip install torch transformers"
            )
        
        # Use config defaults if not provided
        self.model_name = model_name if model_name is not None else LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else LLM_MAX_NEW_TOKENS
        quantization = quantization if quantization is not None else LLM_QUANTIZATION
        self.seed = seed
        
        # Auto-detect device
        if device is None:
            if model is not None:
                # Use device from shared model
                self.device = getattr(model, "device", None) or "cpu"
            elif DEVICE != "cpu":
                self.device = DEVICE
            elif torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
        
        # Use provided model/tokenizer or load new ones
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = _load_tokenizer_with_fallback(self.model_name)
            
            # Setup quantization if requested
            quantization_config = None
            if quantization.lower() in ("4bit", "4-bit"):
                if not HAS_BITSANDBYTES:
                    raise ValueError(
                        "4-bit quantization requires bitsandbytes. "
                        "Install with: pip install bitsandbytes"
                    )
                if self.device != "cuda":
                    if self.device == "mps":
                        raise ValueError(
                            "4-bit quantization with bitsandbytes is not supported on Apple Silicon (MPS).\n"
                            "Alternatives for Apple Silicon:\n"
                            "  1. Use float16 (default on MPS) - already memory efficient\n"
                            "  2. Use a smaller model (e.g., Qwen2.5-3B-Instruct instead of 7B)\n"
                            "  3. Use MLX framework with 4-bit quantization (requires different code path)\n"
                            "  4. Use GGML/GGUF format with llama.cpp\n"
                            "\n"
                            "For now, set LLM_QUANTIZATION=none and use float16 (automatic on MPS)"
                        )
                    else:
                        raise ValueError(
                            f"4-bit quantization only works on CUDA devices, not {self.device}"
                        )
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization.lower() in ("8bit", "8-bit"):
                if not HAS_BITSANDBYTES:
                    raise ValueError(
                        "8-bit quantization requires bitsandbytes. "
                        "Install with: pip install bitsandbytes"
                    )
                if self.device != "cuda":
                    if self.device == "mps":
                        raise ValueError(
                            "8-bit quantization with bitsandbytes is not supported on Apple Silicon (MPS).\n"
                            "Alternatives for Apple Silicon:\n"
                            "  1. Use float16 (default on MPS) - already memory efficient\n"
                            "  2. Use a smaller model (e.g., Qwen2.5-3B-Instruct instead of 7B)\n"
                            "\n"
                            "For now, set LLM_QUANTIZATION=none and use float16 (automatic on MPS)"
                        )
                    else:
                        raise ValueError(
                            f"8-bit quantization only works on CUDA devices, not {self.device}"
                        )
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Determine dtype based on device (only if not using quantization)
            if quantization_config is None:
                if self.device == "cuda":
                    dtype = torch.float16
                elif self.device == "mps":
                    dtype = torch.float16 
                else:
                    dtype = torch.float32
            else:
                dtype = None  # Quantization config handles dtype
            
            # Load model with appropriate configuration
            model_kwargs = {
                "device_map": self.device if quantization_config is None else "auto",
                "trust_remote_code": True,  # Required for some newer models like Ministral
            }
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["dtype"] = dtype
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.eval()
        
        # Store last raw LLM output for debugging
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
