"""Improved baseline agents with better heuristics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from ..env.spaces import CardColor, GuesserAction, Observation, SpymasterAction
from ..env.validation import is_valid_clue
from ..utils.config import EMBEDDING_MODEL
from .baselines import BaseGuesser, BaseSpymaster


class ClusterSpymaster(BaseSpymaster):
    """Spymaster that finds word clusters and gives clues for tight groups.
    
    Key improvement: Uses MIN similarity to cluster (tight connection)
    rather than MEAN similarity (loose connection to all).
    """

    def __init__(
        self,
        vocabulary_path: str,
        model_name: str = EMBEDDING_MODEL,
        cluster_threshold: float = 0.35,  # Min similarity within cluster
        alpha: float = 3.0,
        beta: float = 1.5,
        gamma: float = 0.3,
        top_k: int = 100,
        seed: Optional[int] = None
    ):
        from pathlib import Path
        
        self.vocabulary_path = vocabulary_path
        self.cluster_threshold = cluster_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)
        
        self.model = SentenceTransformer(model_name)
        
        # Load vocabulary
        path = Path(vocabulary_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.vocabulary = [line.strip().lower() for line in f if line.strip()]

    def get_clue(self, obs: Observation) -> SpymasterAction:
        """Generate clue using cluster-based approach."""
        # Separate unrevealed board words by color
        team_words = []
        opponent_words = []
        neutral_words = []
        assassin_words = []
        
        for word, color, revealed in zip(obs.board_words, obs.board_colors, obs.revealed_mask):
            if revealed:
                continue
            word_lower = word.lower()
            if color == CardColor.TEAM:
                team_words.append(word_lower)
            elif color == CardColor.OPPONENT:
                opponent_words.append(word_lower)
            elif color == CardColor.NEUTRAL:
                neutral_words.append(word_lower)
            elif color == CardColor.ASSASSIN:
                assassin_words.append(word_lower)
        
        if not team_words:
            return SpymasterAction(clue="pass", count=0)
        
        # Filter vocabulary
        board_lower = [w.lower() for w in obs.board_words]
        candidates = [w for w in self.vocabulary if w not in board_lower]
        
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
        
        # Find best clue
        best_score = float('-inf')
        best_clue = None
        best_count = 1
        
        for candidate, cand_emb in zip(candidates, candidate_embs):
            is_valid, _ = is_valid_clue(candidate, obs.board_words)
            if not is_valid:
                continue
            
            # Compute similarities to all team words
            team_sims = self._cosine_similarity(cand_emb, team_embs)
            
            # Find best cluster: words above threshold
            cluster_mask = team_sims >= self.cluster_threshold
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size == 0:
                # No cluster, use single best word
                cluster_size = 1
                cluster_score = np.max(team_sims)
            else:
                # Use MIN similarity in cluster (tightest connection)
                cluster_sims = team_sims[cluster_mask]
                cluster_score = np.min(cluster_sims) + 0.1 * cluster_size
            
            # Penalties for danger
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
            score = cluster_score - assassin_penalty - opponent_penalty - neutral_penalty
            
            if score > best_score:
                best_score = score
                best_clue = candidate
                best_count = int(cluster_size)
        
        if best_clue is None:
            # Fallback
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


class ContextualGuesser(BaseGuesser):
    """Guesser that remembers previous clues and uses game history."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        confidence_threshold: float = 0.25,
        history_weight: float = 0.3,  # Weight for previous clues
        seed: Optional[int] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.history_weight = history_weight
        self.rng = np.random.default_rng(seed)
        
        self.model = SentenceTransformer(model_name)
        
        # State tracking
        self.clue_history: List[str] = []
        self.guess_history: List[Tuple[str, bool]] = []  # (word, was_correct)

    def reset(self) -> None:
        """Reset history for new game."""
        self.clue_history = []
        self.guess_history = []

    def get_guess(self, obs: Observation) -> GuesserAction:
        """Make guess using current clue + history."""
        if obs.current_clue is None:
            return GuesserAction(word_index=None)
        
        # Track clue history
        if not self.clue_history or self.clue_history[-1] != obs.current_clue:
            self.clue_history.append(obs.current_clue.lower())
        
        # Find unrevealed words
        unrevealed_indices = [
            i for i, revealed in enumerate(obs.revealed_mask) if not revealed
        ]
        
        if not unrevealed_indices:
            return GuesserAction(word_index=None)
        
        unrevealed_words = [obs.board_words[i].lower() for i in unrevealed_indices]
        
        # Encode current clue and words
        clue_emb = self.model.encode([obs.current_clue.lower()], convert_to_numpy=True)[0]
        word_embs = self.model.encode(unrevealed_words, convert_to_numpy=True)
        
        # Compute similarities to current clue
        current_sims = self._cosine_similarity(clue_emb, word_embs)
        
        # Add history boost: words similar to previous clues are more likely team words
        history_boost = np.zeros_like(current_sims)
        if len(self.clue_history) > 1:
            for prev_clue in self.clue_history[:-1]:  # Exclude current clue
                prev_emb = self.model.encode([prev_clue], convert_to_numpy=True)[0]
                prev_sims = self._cosine_similarity(prev_emb, word_embs)
                history_boost += prev_sims * self.history_weight
        
        # Combine current + history
        total_scores = current_sims + history_boost
        
        # Eliminate words similar to incorrect guesses
        for wrong_word, was_correct in self.guess_history:
            if not was_correct:
                wrong_emb = self.model.encode([wrong_word], convert_to_numpy=True)[0]
                wrong_sims = self._cosine_similarity(wrong_emb, word_embs)
                # Penalize words similar to known wrong guesses
                total_scores -= 0.2 * wrong_sims
        
        # Get best match
        best_idx = np.argmax(total_scores)
        best_sim = total_scores[best_idx]
        
        # Decide: guess or pass
        if best_sim >= self.confidence_threshold:
            word_idx = unrevealed_indices[best_idx]
            guessed_word = obs.board_words[word_idx].lower()
            return GuesserAction(word_index=word_idx)
        else:
            return GuesserAction(word_index=None)

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return np.dot(matrix_norm, vec_norm)


class AdaptiveGuesser(BaseGuesser):
    """Guesser that adapts confidence threshold based on game state."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        base_threshold: float = 0.25,
        seed: Optional[int] = None
    ):
        self.base_threshold = base_threshold
        self.rng = np.random.default_rng(seed)
        
        self.model = SentenceTransformer(model_name)

    def get_guess(self, obs: Observation) -> GuesserAction:
        """Make guess with adaptive confidence threshold."""
        if obs.current_clue is None:
            return GuesserAction(word_index=None)
        
        unrevealed_indices = [
            i for i, revealed in enumerate(obs.revealed_mask) if not revealed
        ]
        
        if not unrevealed_indices:
            return GuesserAction(word_index=None)
        
        # Adaptive threshold based on game state
        # Be aggressive when ahead, conservative when behind
        team_ratio = obs.team_remaining / max(obs.team_remaining + obs.opponent_remaining, 1)
        
        if team_ratio > 0.6:
            # Ahead: be more aggressive (lower threshold)
            threshold = self.base_threshold * 0.8
        elif team_ratio < 0.4:
            # Behind: be conservative (higher threshold)
            threshold = self.base_threshold * 1.2
        else:
            threshold = self.base_threshold
        
        # Also consider guesses remaining: last guess should be more confident
        if obs.remaining_guesses == 1:
            threshold *= 1.15
        
        unrevealed_words = [obs.board_words[i].lower() for i in unrevealed_indices]
        
        # Encode and score
        clue_emb = self.model.encode([obs.current_clue.lower()], convert_to_numpy=True)[0]
        word_embs = self.model.encode(unrevealed_words, convert_to_numpy=True)
        
        similarities = self._cosine_similarity(clue_emb, word_embs)
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= threshold:
            return GuesserAction(word_index=unrevealed_indices[best_idx])
        else:
            return GuesserAction(word_index=None)

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return np.dot(matrix_norm, vec_norm)


class CrossEncoderSpymaster(BaseSpymaster):
    """Spymaster using hybrid Retrieve & Re-Rank with Cross-Encoder.
    
    Strategy: Use Bi-Encoder (fast) to retrieve top-k candidates, then
    Cross-Encoder (accurate) to re-rank them. This provides better accuracy
    than pure Bi-Encoder while maintaining reasonable speed.
    
    Key improvement over EmbeddingsSpymaster:
    - Cross-Encoders compute similarity scores directly from text pairs
    - Better at understanding nuanced relationships between clues and words
    - Hybrid approach balances speed and accuracy
    """

    def __init__(
        self,
        vocabulary_path: str,
        bi_encoder_model: str = EMBEDDING_MODEL,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha: float = 3.0,  # Assassin penalty
        beta: float = 1.5,   # Opponent penalty
        gamma: float = 0.3,  # Neutral penalty
        similarity_threshold: float = 0.3,
        top_k_retrieve: int = 20,  # Top candidates from Bi-Encoder
        top_k_candidates: int = 100,  # Initial candidate pool
        seed: Optional[int] = None
    ):
        """Initialize Cross-Encoder-based spymaster.
        
        Args:
            vocabulary_path: Path to vocabulary file
            bi_encoder_model: SentenceTransformer model for fast retrieval
            cross_encoder_model: CrossEncoder model for accurate re-ranking
            alpha: Penalty weight for assassin similarity
            beta: Penalty weight for opponent similarity
            gamma: Penalty weight for neutral similarity
            similarity_threshold: Minimum similarity to include in count
            top_k_retrieve: Number of top candidates to re-rank with Cross-Encoder
            top_k_candidates: Number of candidates to evaluate with Bi-Encoder
            seed: Random seed for tie-breaking
        """
        from pathlib import Path
        
        self.vocabulary_path = vocabulary_path
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.similarity_threshold = similarity_threshold
        self.top_k_retrieve = top_k_retrieve
        self.top_k_candidates = top_k_candidates
        self.rng = np.random.default_rng(seed)
        
        # Load models
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Load vocabulary
        path = Path(vocabulary_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.vocabulary = [line.strip().lower() for line in f if line.strip()]

    def get_clue(self, obs: Observation) -> SpymasterAction:
        """Generate clue using hybrid Retrieve & Re-Rank approach."""
        # Separate unrevealed board words by color
        team_words = []
        opponent_words = []
        neutral_words = []
        assassin_words = []
        
        for word, color, revealed in zip(obs.board_words, obs.board_colors, obs.revealed_mask):
            if revealed:
                continue
            word_lower = word.lower()
            if color == CardColor.TEAM:
                team_words.append(word_lower)
            elif color == CardColor.OPPONENT:
                opponent_words.append(word_lower)
            elif color == CardColor.NEUTRAL:
                neutral_words.append(word_lower)
            elif color == CardColor.ASSASSIN:
                assassin_words.append(word_lower)
        
        if not team_words:
            return SpymasterAction(clue="pass", count=0)
        
        # Filter vocabulary
        board_lower = [w.lower() for w in obs.board_words]
        candidates = [w for w in self.vocabulary if w not in board_lower]
        
        if len(candidates) > self.top_k_candidates:
            candidates = self.rng.choice(candidates, size=self.top_k_candidates, replace=False).tolist()
        
        # STEP 1: Fast retrieval with Bi-Encoder
        candidate_embs = self.bi_encoder.encode(candidates, convert_to_numpy=True)
        team_embs = self.bi_encoder.encode(team_words, convert_to_numpy=True)
        
        opponent_embs = (
            self.bi_encoder.encode(opponent_words, convert_to_numpy=True)
            if opponent_words else None
        )
        neutral_embs = (
            self.bi_encoder.encode(neutral_words, convert_to_numpy=True)
            if neutral_words else None
        )
        assassin_embs = (
            self.bi_encoder.encode(assassin_words, convert_to_numpy=True)
            if assassin_words else None
        )
        
        # Score candidates with Bi-Encoder (fast)
        candidate_scores = []
        for cand_emb in candidate_embs:
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
            
            score = team_mean - assassin_penalty - opponent_penalty - neutral_penalty
            candidate_scores.append(score)
        
        # Get top-k candidates for re-ranking
        top_k_indices = np.argsort(candidate_scores)[-self.top_k_retrieve:][::-1]
        top_k_candidates = [candidates[i] for i in top_k_indices]
        
        # STEP 2: Accurate re-ranking with Cross-Encoder
        best_score = float('-inf')
        best_clue = None
        best_count = 1
        
        for candidate in top_k_candidates:
            is_valid, _ = is_valid_clue(candidate, obs.board_words)
            if not is_valid:
                continue
            
            # Create pairs: (candidate, word) for all board words
            pairs = []
            word_types = []
            
            for word in team_words:
                pairs.append([candidate, word])
                word_types.append('team')
            
            if opponent_words:
                for word in opponent_words:
                    pairs.append([candidate, word])
                    word_types.append('opponent')
            
            if neutral_words:
                for word in neutral_words:
                    pairs.append([candidate, word])
                    word_types.append('neutral')
            
            if assassin_words:
                for word in assassin_words:
                    pairs.append([candidate, word])
                    word_types.append('assassin')
            
            # Score pairs with Cross-Encoder
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Compute final score with penalties
            team_scores = [cross_scores[i] for i, wt in enumerate(word_types) if wt == 'team']
            team_mean = np.mean(team_scores) if team_scores else 0.0
            
            assassin_penalty = 0.0
            assassin_scores = [cross_scores[i] for i, wt in enumerate(word_types) if wt == 'assassin']
            if assassin_scores:
                assassin_penalty = self.alpha * np.max(assassin_scores)
            
            opponent_penalty = 0.0
            opponent_scores = [cross_scores[i] for i, wt in enumerate(word_types) if wt == 'opponent']
            if opponent_scores:
                opponent_penalty = self.beta * np.max(opponent_scores)
            
            neutral_penalty = 0.0
            neutral_scores = [cross_scores[i] for i, wt in enumerate(word_types) if wt == 'neutral']
            if neutral_scores:
                neutral_penalty = self.gamma * np.mean(neutral_scores)
            
            # Count: how many team words have score above threshold
            # Note: Cross-Encoder scores are logits, not normalized similarities
            # We use a relative threshold based on the score distribution
            team_threshold = np.mean(team_scores) - 0.5 * np.std(team_scores) if team_scores else 0.0
            count = max(1, int(np.sum(np.array(team_scores) >= team_threshold)))
            
            score = team_mean - assassin_penalty - opponent_penalty - neutral_penalty
            
            if score > best_score:
                best_score = score
                best_clue = candidate
                best_count = count
        
        if best_clue is None:
            # Fallback
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


class CrossEncoderGuesser(BaseGuesser):
    """Guesser using hybrid Retrieve & Re-Rank with Cross-Encoder.
    
    Strategy: Use Bi-Encoder to retrieve top-k candidates, then Cross-Encoder
    to re-rank them. This provides better accuracy than pure Bi-Encoder.
    
    Key improvement over EmbeddingsGuesser:
    - Cross-Encoders provide superior accuracy for semantic matching
    - Better at understanding nuanced relationships between clues and words
    - Hybrid approach maintains reasonable speed (~1-2s per guess)
    """

    def __init__(
        self,
        bi_encoder_model: str = EMBEDDING_MODEL,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        confidence_threshold: float = 0.25,
        top_k_retrieve: int = 10,  # Top candidates to re-rank
        seed: Optional[int] = None
    ):
        """Initialize Cross-Encoder-based guesser.
        
        Args:
            bi_encoder_model: SentenceTransformer model for fast retrieval
            cross_encoder_model: CrossEncoder model for accurate re-ranking
            confidence_threshold: Minimum score to make a guess (else STOP)
            top_k_retrieve: Number of top candidates to re-rank with Cross-Encoder
            seed: Random seed for tie-breaking
        """
        self.confidence_threshold = confidence_threshold
        self.top_k_retrieve = top_k_retrieve
        self.rng = np.random.default_rng(seed)
        
        # Load models
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)

    def get_guess(self, obs: Observation) -> GuesserAction:
        """Make a guess using hybrid Retrieve & Re-Rank approach."""
        if obs.current_clue is None:
            return GuesserAction(word_index=None)
        
        # Find unrevealed words
        unrevealed_indices = [
            i for i, revealed in enumerate(obs.revealed_mask) if not revealed
        ]
        
        if not unrevealed_indices:
            return GuesserAction(word_index=None)
        
        unrevealed_words = [obs.board_words[i].lower() for i in unrevealed_indices]
        
        # STEP 1: Fast retrieval with Bi-Encoder
        clue_emb = self.bi_encoder.encode([obs.current_clue.lower()], convert_to_numpy=True)[0]
        word_embs = self.bi_encoder.encode(unrevealed_words, convert_to_numpy=True)
        
        similarities = self._cosine_similarity(clue_emb, word_embs)
        
        # Get top-k candidates for re-ranking
        top_k_indices = np.argsort(similarities)[-self.top_k_retrieve:][::-1]
        top_k_words = [unrevealed_words[i] for i in top_k_indices]
        top_k_original_indices = [unrevealed_indices[i] for i in top_k_indices]
        
        # STEP 2: Accurate re-ranking with Cross-Encoder
        pairs = [[obs.current_clue.lower(), word] for word in top_k_words]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Get best match
        best_idx = np.argmax(cross_scores)
        best_score = cross_scores[best_idx]
        
        # Note: Cross-Encoder scores are logits, not normalized similarities
        # We use a relative threshold: score should be positive and above mean
        mean_score = np.mean(cross_scores)
        threshold = mean_score + (self.confidence_threshold * (np.max(cross_scores) - mean_score))
        
        # Decide: guess or pass
        if best_score >= threshold:
            return GuesserAction(word_index=top_k_original_indices[best_idx])
        else:
            return GuesserAction(word_index=None)  # STOP

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return np.dot(matrix_norm, vec_norm)
