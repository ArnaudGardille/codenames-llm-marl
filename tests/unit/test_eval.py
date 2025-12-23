"""Tests for evaluation harness and metrics."""

import tempfile
from pathlib import Path

import pytest

from codenames_rl.agents.baselines import (
    EmbeddingsGuesser,
    EmbeddingsSpymaster,
    RandomGuesser,
    RandomSpymaster,
)
from codenames_rl.eval import (
    EvaluationHarness,
    EvaluationMetrics,
    GameResult,
    compare_agents,
    compute_metrics,
)


@pytest.fixture
def temp_vocabulary():
    """Create a temporary vocabulary file for testing."""
    vocab_words = [
        "apple", "banana", "cherry", "dog", "elephant",
        "forest", "guitar", "house", "island", "jungle",
        "kitchen", "laptop", "mountain", "notebook", "ocean",
        "piano", "queen", "river", "sunset", "tree",
        "umbrella", "violin", "water", "xylophone", "yellow",
        "zebra", "animal", "fruit", "music", "nature"
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(vocab_words))
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_wordlist():
    """Create a temporary wordlist file for testing."""
    words = [
        "car", "book", "phone", "chair", "lamp",
        "desk", "window", "door", "wall", "floor",
        "ceiling", "table", "pen", "paper", "ink",
        "light", "dark", "sun", "moon", "star",
        "cloud", "rain", "snow", "wind", "storm",
        "fire", "ice", "metal", "wood", "stone"
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(words))
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestGameResult:
    """Tests for GameResult data structure."""

    def test_initialization(self):
        """Test GameResult can be created."""
        result = GameResult(
            seed=42,
            outcome="win",
            total_turns=5,
            total_guesses=12,
            score=9,
            illegal_clues=0,
            clue_history=[("animal", 2), ("place", 3)]
        )
        
        assert result.seed == 42
        assert result.outcome == "win"
        assert result.score == 9

    def test_to_dict(self):
        """Test GameResult can be serialized."""
        result = GameResult(
            seed=42,
            outcome="win",
            total_turns=5,
            total_guesses=12,
            score=9,
            illegal_clues=0,
            clue_history=[("animal", 2)]
        )
        
        d = result.to_dict()
        assert d["seed"] == 42
        assert d["outcome"] == "win"
        assert d["score"] == 9


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics."""

    def test_initialization(self):
        """Test EvaluationMetrics can be created."""
        metrics = EvaluationMetrics(
            num_games=100,
            win_rate=0.75,
            avg_score=7.5,
            avg_turns=6.2,
            illegal_clue_rate=0.02,
            assassin_rate=0.15,
            opponent_win_rate=0.10,
            avg_guesses=14.3
        )
        
        assert metrics.num_games == 100
        assert metrics.win_rate == 0.75

    def test_to_dict(self):
        """Test EvaluationMetrics can be serialized."""
        metrics = EvaluationMetrics(
            num_games=100,
            win_rate=0.75,
            avg_score=7.5,
            avg_turns=6.2,
            illegal_clue_rate=0.02,
            assassin_rate=0.15,
            opponent_win_rate=0.10,
            avg_guesses=14.3
        )
        
        d = metrics.to_dict()
        assert d["num_games"] == 100
        assert d["win_rate"] == 0.75

    def test_str_representation(self):
        """Test string representation is human-readable."""
        metrics = EvaluationMetrics(
            num_games=100,
            win_rate=0.75,
            avg_score=7.5,
            avg_turns=6.2,
            illegal_clue_rate=0.02,
            assassin_rate=0.15,
            opponent_win_rate=0.10,
            avg_guesses=14.3
        )
        
        s = str(metrics)
        assert "100 games" in s
        assert "75.0%" in s or "75%" in s


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_empty_results(self):
        """Test compute_metrics with no results."""
        metrics = compute_metrics([])
        
        assert metrics.num_games == 0
        assert metrics.win_rate == 0.0

    def test_all_wins(self):
        """Test compute_metrics with all wins."""
        results = [
            GameResult(
                seed=i,
                outcome="win",
                total_turns=5,
                total_guesses=10,
                score=9,
                illegal_clues=0
            )
            for i in range(10)
        ]
        
        metrics = compute_metrics(results)
        
        assert metrics.num_games == 10
        assert metrics.win_rate == 1.0
        assert metrics.avg_score == 9.0

    def test_mixed_outcomes(self):
        """Test compute_metrics with mixed outcomes."""
        results = [
            GameResult(seed=0, outcome="win", total_turns=5, total_guesses=10, score=9, illegal_clues=0),
            GameResult(seed=1, outcome="loss_assassin", total_turns=3, total_guesses=6, score=4, illegal_clues=0),
            GameResult(seed=2, outcome="loss_opponent_won", total_turns=8, total_guesses=15, score=7, illegal_clues=1),
            GameResult(seed=3, outcome="win", total_turns=6, total_guesses=12, score=9, illegal_clues=0),
        ]
        
        metrics = compute_metrics(results)
        
        assert metrics.num_games == 4
        assert metrics.win_rate == 0.5
        assert metrics.assassin_rate == 0.25
        assert metrics.opponent_win_rate == 0.25


class TestEvaluationHarness:
    """Tests for EvaluationHarness."""

    def test_initialization(self, temp_wordlist, temp_vocabulary):
        """Test harness can be initialized."""
        spymaster = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = RandomGuesser(seed=42)
        
        harness = EvaluationHarness(
            wordlist_path=temp_wordlist,
            spymaster=spymaster,
            guesser=guesser
        )
        
        assert harness.wordlist_path == temp_wordlist
        assert harness.spymaster is spymaster
        assert harness.guesser is guesser

    def test_run_episode(self, temp_wordlist, temp_vocabulary):
        """Test running a single episode."""
        spymaster = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = RandomGuesser(seed=42)
        
        harness = EvaluationHarness(
            wordlist_path=temp_wordlist,
            spymaster=spymaster,
            guesser=guesser,
            verbose=False
        )
        
        result = harness.run_episode(seed=123)
        
        assert isinstance(result, GameResult)
        assert result.seed == 123
        assert result.outcome in ["win", "loss_assassin", "loss_opponent_won", "truncated"]
        assert result.total_turns >= 0
        assert result.score >= 0

    def test_evaluate(self, temp_wordlist, temp_vocabulary):
        """Test evaluating over multiple games."""
        spymaster = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = RandomGuesser(seed=42)
        
        harness = EvaluationHarness(
            wordlist_path=temp_wordlist,
            spymaster=spymaster,
            guesser=guesser,
            verbose=False
        )
        
        metrics = harness.evaluate(num_games=5, start_seed=0)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.num_games == 5
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_evaluate_with_specific_seeds(self, temp_wordlist, temp_vocabulary):
        """Test evaluation with specific seed list."""
        spymaster = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = RandomGuesser(seed=42)
        
        harness = EvaluationHarness(
            wordlist_path=temp_wordlist,
            spymaster=spymaster,
            guesser=guesser,
            verbose=False
        )
        
        seeds = [10, 20, 30]
        metrics = harness.evaluate(num_games=3, seeds=seeds)
        
        assert metrics.num_games == 3

    def test_evaluate_with_details(self, temp_wordlist, temp_vocabulary):
        """Test evaluate_with_details returns both metrics and results."""
        spymaster = RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42)
        guesser = RandomGuesser(seed=42)
        
        harness = EvaluationHarness(
            wordlist_path=temp_wordlist,
            spymaster=spymaster,
            guesser=guesser,
            verbose=False
        )
        
        metrics, results = harness.evaluate_with_details(num_games=3, start_seed=0)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert len(results) == 3
        assert all(isinstance(r, GameResult) for r in results)


@pytest.mark.slow
class TestCompareAgents:
    """Tests for compare_agents function."""

    def test_compare_two_configs(self, temp_wordlist, temp_vocabulary):
        """Test comparing two agent configurations."""
        configs = {
            "Random": {
                "spymaster": RandomSpymaster(vocabulary_path=temp_vocabulary, seed=42),
                "guesser": RandomGuesser(seed=42)
            },
            "Embeddings": {
                "spymaster": EmbeddingsSpymaster(vocabulary_path=temp_vocabulary, seed=42, top_k=10),
                "guesser": EmbeddingsGuesser(seed=42)
            }
        }
        
        results = compare_agents(
            agent_configs=configs,
            wordlist_path=temp_wordlist,
            num_games=3,
            verbose=False
        )
        
        assert "Random" in results
        assert "Embeddings" in results
        assert isinstance(results["Random"], EvaluationMetrics)
        assert isinstance(results["Embeddings"], EvaluationMetrics)

