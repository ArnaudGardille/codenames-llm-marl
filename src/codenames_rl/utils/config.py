"""Configuration management using environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env")


def _get(key: str, default, type_fn):
    """Get environment variable with type conversion and fallback."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return type_fn(value)
    except (ValueError, TypeError):
        return default


def _get_path(key: str, default: str) -> str:
    """Get path from environment variable, relative to project root if not absolute."""
    value = os.getenv(key, default)
    path = Path(value)
    return str(path if path.is_absolute() else project_root / value)


def get_language_paths(lang: str) -> tuple[str, str]:
    """Get wordlist and vocabulary paths for a given language code.
    
    Args:
        lang: Language code ("en" or "fr")
        
    Returns:
        Tuple of (wordlist_path, vocabulary_path) as absolute paths
        
    Raises:
        ValueError: If language code is not supported
    """
    supported_languages = {
        "en": ("configs/wordlist_en.txt", "configs/vocabulary_en.txt"),
        "fr": ("configs/wordlist_fr.txt", "configs/vocabulary_fr.txt"),
    }
    
    if lang not in supported_languages:
        raise ValueError(
            f"Unsupported language code: {lang}. Supported languages: {list(supported_languages.keys())}"
        )
    
    wordlist_rel, vocabulary_rel = supported_languages[lang]
    # Resolve paths relative to project root (don't check env vars for language-specific paths)
    wordlist_path = Path(wordlist_rel)
    vocabulary_path = Path(vocabulary_rel)
    
    return str(wordlist_path if wordlist_path.is_absolute() else project_root / wordlist_path), \
           str(vocabulary_path if vocabulary_path.is_absolute() else project_root / vocabulary_path)


# Game Environment Settings
MAX_TURNS = _get("MAX_TURNS", 20, int)
MAX_GUESSES = _get("MAX_GUESSES", 9, int)

# Evaluation Settings
NUM_GAMES = _get("NUM_GAMES", 100, int)
START_SEED = _get("START_SEED", 0, int)
AGENT_SEED = _get("AGENT_SEED", 42, int)

# File Paths
WORDLIST_PATH = _get_path("WORDLIST_PATH", "configs/wordlist_en.txt")
VOCABULARY_PATH = _get_path("VOCABULARY_PATH", "configs/vocabulary_en.txt")

# Model Settings
# Construct full embedding model name: if env var contains '/', use as-is (custom path),
# otherwise prepend 'sentence-transformers/' prefix
_embedding_model_raw = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
if "/" in _embedding_model_raw:
    EMBEDDING_MODEL = _embedding_model_raw
else:
    EMBEDDING_MODEL = f"sentence-transformers/{_embedding_model_raw}"
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "")

# LLM Model Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LLM_TEMPERATURE = _get("LLM_TEMPERATURE", 0.7, float)
LLM_MAX_NEW_TOKENS = _get("LLM_MAX_NEW_TOKENS", 128, int)
LLM_QUANTIZATION = os.getenv("LLM_QUANTIZATION", "none")

# Performance Settings
DEVICE = os.getenv("DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = _get("EMBEDDING_BATCH_SIZE", 32, int)

