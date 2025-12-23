"""Tests for adversarial Codenames environments."""

import pytest
import numpy as np

from codenames_rl.env.adversarial_core import CodenamesAdversarialCore
from codenames_rl.env.adversarial_pz import CodenamesAdversarialPZ
from codenames_rl.env.adversarial_gym import CodenamesAdversarialGym
from codenames_rl.env.spaces import SpymasterAction, GuesserAction
from codenames_rl.agents.baselines import RandomSpymaster, RandomGuesser


class TestCodenamesAdversarialCore:
    """Tests for the core adversarial game logic."""
    
    def test_initialization(self, temp_vocabulary):
        """Test that core initializes correctly."""
        rng = np.random.default_rng(42)
        core = CodenamesAdversarialCore(temp_vocabulary, rng)
        
        assert len(core.board_words) == 25
        assert len(core.board_colors_absolute) == 25
        assert core.team_a_remaining == 9
        assert core.team_b_remaining == 8
        assert core.current_team == 'team_a'
        assert core.current_phase == 'spymaster'
        assert not core.game_over
    
    def test_board_generation(self, temp_vocabulary):
        """Test board has correct card distribution."""
        rng = np.random.default_rng(42)
        core = CodenamesAdversarialCore(temp_vocabulary, rng)
        
        from codenames_rl.env.spaces import CardColor
        
        team_count = sum(1 for c in core.board_colors_absolute if c == CardColor.TEAM)
        opponent_count = sum(1 for c in core.board_colors_absolute if c == CardColor.OPPONENT)
        neutral_count = sum(1 for c in core.board_colors_absolute if c == CardColor.NEUTRAL)
        assassin_count = sum(1 for c in core.board_colors_absolute if c == CardColor.ASSASSIN)
        
        assert team_count == 9
        assert opponent_count == 8
        assert neutral_count == 7
        assert assassin_count == 1
    
    def test_spymaster_action(self, temp_vocabulary):
        """Test spymaster giving a clue."""
        rng = np.random.default_rng(42)
        core = CodenamesAdversarialCore(temp_vocabulary, rng)
        
        reward, done, info = core.execute_spymaster_action('team_a', 'test', 2)
        
        assert reward == 0.0
        assert not done
        assert core.current_clue == 'test'
        assert core.current_count == 2
        assert core.current_phase == 'guesser'
    
    def test_team_alternation(self, temp_vocabulary):
        """Test that teams alternate correctly."""
        rng = np.random.default_rng(42)
        core = CodenamesAdversarialCore(temp_vocabulary, rng)
        
        # Team A spymaster
        assert core.current_team == 'team_a'
        core.execute_spymaster_action('team_a', 'test', 1)
        
        # Team A guesser passes
        assert core.current_phase == 'guesser'
        core.execute_guesser_action('team_a', None)
        
        # Should be Team B's turn now
        assert core.current_team == 'team_b'
        assert core.current_phase == 'spymaster'
    
    def test_flipped_colors(self, temp_vocabulary):
        """Test that board colors flip for team B."""
        rng = np.random.default_rng(42)
        core = CodenamesAdversarialCore(temp_vocabulary, rng)
        
        from codenames_rl.env.spaces import CardColor
        
        team_a_colors = core.get_board_colors_for_team('team_a')
        team_b_colors = core.get_board_colors_for_team('team_b')
        
        for i in range(25):
            if core.board_colors_absolute[i] == CardColor.TEAM:
                assert team_a_colors[i] == CardColor.TEAM
                assert team_b_colors[i] == CardColor.OPPONENT
            elif core.board_colors_absolute[i] == CardColor.OPPONENT:
                assert team_a_colors[i] == CardColor.OPPONENT
                assert team_b_colors[i] == CardColor.TEAM


class TestCodenamesAdversarialPZ:
    """Tests for PettingZoo wrapper."""
    
    def test_initialization(self, temp_vocabulary):
        """Test that PettingZoo env initializes."""
        env = CodenamesAdversarialPZ(temp_vocabulary)
        env.reset(seed=42)
        
        assert len(env.agents) == 4
        assert 'team_a_spymaster' in env.agents
        assert 'team_a_guesser' in env.agents
        assert 'team_b_spymaster' in env.agents
        assert 'team_b_guesser' in env.agents
    
    def test_step_execution(self, temp_vocabulary):
        """Test that step executes correctly."""
        env = CodenamesAdversarialPZ(temp_vocabulary)
        env.reset(seed=42)
        
        agent = env.agent_selection
        obs = env.observe(agent)
        
        # Execute a spymaster action
        action = SpymasterAction(clue='test', count=2)
        env.step(action)
        
        assert env.core.current_clue == 'test'


class TestCodenamesAdversarialGym:
    """Tests for Gymnasium self-play wrapper."""
    
    def test_initialization(self, temp_vocabulary):
        """Test that Gymnasium env initializes."""
        opponent_spy = RandomSpymaster(temp_vocabulary, seed=42)
        opponent_guess = RandomGuesser(seed=42)
        
        env = CodenamesAdversarialGym(
            temp_vocabulary,
            opponent_spy.get_clue,
            opponent_guess.get_guess
        )
        
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert len(obs.board_words) == 25
    
    def test_step_execution(self, temp_vocabulary):
        """Test that step executes and auto-runs opponent."""
        opponent_spy = RandomSpymaster(temp_vocabulary, seed=42)
        opponent_guess = RandomGuesser(seed=42)
        
        env = CodenamesAdversarialGym(
            temp_vocabulary,
            opponent_spy.get_clue,
            opponent_guess.get_guess
        )
        
        obs, info = env.reset(seed=42)
        
        # Team A spymaster gives clue
        action = SpymasterAction(clue='test', count=2)
        obs, reward, done, truncated, info = env.step(action)
        
        assert obs is not None
        assert not done
    
    def test_full_episode(self, temp_vocabulary):
        """Test a full episode runs without errors."""
        opponent_spy = RandomSpymaster(temp_vocabulary, seed=42)
        opponent_guess = RandomGuesser(seed=42)
        
        env = CodenamesAdversarialGym(
            temp_vocabulary,
            opponent_spy.get_clue,
            opponent_guess.get_guess
        )
        
        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Simple policy: random clue or pass
            if env.core.current_phase == 'spymaster':
                action = SpymasterAction(clue='test', count=1)
            else:
                action = GuesserAction(word_index=None)  # Always pass
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        assert steps < max_steps  # Should terminate


@pytest.fixture
def temp_vocabulary(temp_vocabulary):
    """Reuse the existing temp_vocabulary fixture from test_baselines."""
    return temp_vocabulary


