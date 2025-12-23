"""Streamlit frontend for Codenames game visualization."""

import streamlit as st
from pathlib import Path

from codenames_rl.env import CodenamesEnv, SpymasterAction, GuesserAction, GamePhase, CardColor
from codenames_rl.utils.config import get_language_paths

# Page config
st.set_page_config(page_title="Codenames", page_icon="🎯", layout="centered")

# Custom CSS for the board
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; }
    .card {
        padding: 12px 8px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 13px;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: transform 0.1s;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .card:hover { transform: scale(1.02); }
    .card-hidden { background: #2d2d44; color: #e0e0e0; border: 2px solid #404060; }
    .card-team { background: #1e88e5; color: white; }
    .card-opponent { background: #e53935; color: white; }
    .card-neutral { background: #8d6e63; color: white; }
    .card-assassin { background: #212121; color: #ff5252; border: 2px solid #ff5252; }
    .phase-banner {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 18px;
        font-weight: bold;
    }
    .spymaster-phase { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .guesser-phase { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
    .game-over { background: linear-gradient(135deg, #434343 0%, #000000 100%); color: white; }
    h1 { color: #e0e0e0 !important; text-align: center; }
    .score { color: #e0e0e0; text-align: center; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

def init_game(wordlist_path: str, seed: int = None):
    """Initialize a new game.
    
    Args:
        wordlist_path: Path to wordlist file
        seed: Random seed for game generation
    """
    env = CodenamesEnv(str(wordlist_path))
    obs, _ = env.reset(seed=seed)
    return env, obs


def get_card_class(color: CardColor, revealed: bool) -> str:
    """Get CSS class for a card."""
    if not revealed:
        return "card-hidden"
    return {
        CardColor.TEAM: "card-team",
        CardColor.OPPONENT: "card-opponent",
        CardColor.NEUTRAL: "card-neutral",
        CardColor.ASSASSIN: "card-assassin",
    }[color]


# Session state initialization
if "env" not in st.session_state:
    st.session_state.language = "en"
    wordlist_path, _ = get_language_paths(st.session_state.language)
    st.session_state.env, st.session_state.obs = init_game(wordlist_path)
    st.session_state.show_colors = False
    st.session_state.history = []

env = st.session_state.env
obs = st.session_state.obs

# Header
st.markdown("# 🎯 Codenames")

# Controls row
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("🔄 New Game", use_container_width=True):
        wordlist_path, _ = get_language_paths(st.session_state.language)
        st.session_state.env, st.session_state.obs = init_game(wordlist_path)
        st.session_state.history = []
        st.rerun()
with col2:
    st.session_state.show_colors = st.toggle("👁 Spymaster View", st.session_state.show_colors)
with col3:
    language_options = ["en", "fr"]
    current_index = 0 if st.session_state.language == "en" else 1
    new_language = st.selectbox("Language", language_options, index=current_index)
    if new_language != st.session_state.language:
        st.session_state.language = new_language
        wordlist_path, _ = get_language_paths(st.session_state.language)
        st.session_state.env, st.session_state.obs = init_game(wordlist_path)
        st.session_state.history = []
        st.rerun()
with col4:
    seed = st.number_input("Seed", value=None, step=1, placeholder="Random")
    if st.button("Set Seed", use_container_width=True):
        wordlist_path, _ = get_language_paths(st.session_state.language)
        st.session_state.env, st.session_state.obs = init_game(wordlist_path, seed=int(seed) if seed else None)
        st.session_state.history = []
        st.rerun()

# Phase banner
phase_class = {
    GamePhase.SPYMASTER_TURN: "spymaster-phase",
    GamePhase.GUESSER_TURN: "guesser-phase",
    GamePhase.GAME_OVER: "game-over",
}[obs.phase]

phase_text = {
    GamePhase.SPYMASTER_TURN: "🎭 Spymaster's Turn",
    GamePhase.GUESSER_TURN: f"🔍 Guesser's Turn — Clue: {obs.current_clue} ({obs.current_count}) — {obs.remaining_guesses} guesses left",
    GamePhase.GAME_OVER: "🏁 Game Over",
}[obs.phase]

st.markdown(f'<div class="phase-banner {phase_class}">{phase_text}</div>', unsafe_allow_html=True)

# Score
st.markdown(f'<p class="score">🔵 Team: {obs.team_remaining} left &nbsp;|&nbsp; 🔴 Opponent: {obs.opponent_remaining} left</p>', unsafe_allow_html=True)

# Board
for row in range(5):
    cols = st.columns(5)
    for col in range(5):
        idx = row * 5 + col
        word = obs.board_words[idx]
        revealed = obs.revealed_mask[idx]
        color = obs.board_colors[idx]
        
        # Determine display
        show_color = revealed or st.session_state.show_colors
        card_class = get_card_class(color, show_color)
        
        with cols[col]:
            # Clickable card during guesser turn
            if obs.phase == GamePhase.GUESSER_TURN and not revealed:
                if st.button(word.upper(), key=f"card_{idx}", use_container_width=True):
                    action = GuesserAction(word_index=idx)
                    new_obs, reward, terminated, _, info = env.step(action)
                    st.session_state.obs = new_obs
                    st.session_state.history.append(f"Guessed: {word} → {info.get('color', 'unknown')}")
                    st.rerun()
            else:
                st.markdown(f'<div class="card {card_class}">{word}</div>', unsafe_allow_html=True)

# Spymaster input
if obs.phase == GamePhase.SPYMASTER_TURN:
    st.markdown("---")
    st.markdown("### Give a Clue")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        clue = st.text_input("Clue word", placeholder="Enter clue...", label_visibility="collapsed")
    with col2:
        count = st.number_input("Count", min_value=1, max_value=9, value=1, label_visibility="collapsed")
    with col3:
        if st.button("Submit", use_container_width=True):
            if clue:
                action = SpymasterAction(clue=clue.strip(), count=int(count))
                new_obs, reward, terminated, _, info = env.step(action)
                st.session_state.obs = new_obs
                if "error" in info:
                    st.error(info["error"])
                else:
                    st.session_state.history.append(f"Clue: {clue} ({count})")
                    st.rerun()

# Pass button during guesser turn
if obs.phase == GamePhase.GUESSER_TURN:
    st.markdown("---")
    if st.button("⏭ Pass Turn", use_container_width=True):
        action = GuesserAction(word_index=None)
        new_obs, _, _, _, _ = env.step(action)
        st.session_state.obs = new_obs
        st.session_state.history.append("Passed turn")
        st.rerun()

# History sidebar
with st.sidebar:
    st.markdown("### 📜 History")
    for item in reversed(st.session_state.history[-10:]):
        st.markdown(f"- {item}")


