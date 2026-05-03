"""Agent implementations for Codenames."""

from typing import Optional, Union

from .baselines import (
    BaseGuesser,
    BaseSpymaster,
    EmbeddingsGuesser,
    EmbeddingsSpymaster,
    LLMGuesser,
    LLMSpymaster,
    QwenEmbeddingGuesser,
    QwenEmbeddingSpymaster,
    RandomGuesser,
    RandomSpymaster,
)

AGENT_KINDS = ("random", "embeddings", "qwen_embedding", "llm")

_SPYMASTER_CLASSES = {
    "random": RandomSpymaster,
    "embeddings": EmbeddingsSpymaster,
    "qwen_embedding": QwenEmbeddingSpymaster,
    "llm": LLMSpymaster,
}

_GUESSER_CLASSES = {
    "random": RandomGuesser,
    "embeddings": EmbeddingsGuesser,
    "qwen_embedding": QwenEmbeddingGuesser,
    "llm": LLMGuesser,
}


def create_agent(
    agent_type: str,
    vocabulary_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Union[BaseSpymaster, BaseGuesser]:
    """Build an agent from a string like ``"embeddings_spymaster"`` / ``"llm_guesser"``.

    Spymasters that score over a vocabulary require ``vocabulary_path``.
    """
    if agent_type.endswith("_spymaster"):
        kind = agent_type[: -len("_spymaster")]
        cls = _SPYMASTER_CLASSES.get(kind)
        if cls is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        if kind in ("random", "embeddings", "qwen_embedding"):
            if vocabulary_path is None:
                raise ValueError(f"vocabulary_path required for {agent_type}")
            return cls(vocabulary_path=vocabulary_path, seed=seed)
        return cls(seed=seed)

    if agent_type.endswith("_guesser"):
        kind = agent_type[: -len("_guesser")]
        cls = _GUESSER_CLASSES.get(kind)
        if cls is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return cls(seed=seed)

    raise ValueError(f"Unknown agent type: {agent_type}")


__all__ = [
    "BaseSpymaster",
    "BaseGuesser",
    "RandomSpymaster",
    "RandomGuesser",
    "EmbeddingsSpymaster",
    "EmbeddingsGuesser",
    "QwenEmbeddingSpymaster",
    "QwenEmbeddingGuesser",
    "LLMSpymaster",
    "LLMGuesser",
    "create_agent",
    "AGENT_KINDS",
]
