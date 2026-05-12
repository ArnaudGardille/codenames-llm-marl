"""Wordlist loading and clue legality.

``is_valid_clue`` is the single source of truth for what counts as a legal
Spymaster clue — every agent and env layer routes through it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple


def load_wordlist(path: str) -> List[str]:
    """Read a wordlist file, one word per line, stripped and uppercased.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Wordlist not found: {path}")
    with p.open(encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]


# A clue must be a single alphanumeric token. We accept letters from any
# alphabet (e.g. accented characters in French) plus ASCII digits.
_ALPHANUMERIC_RE = re.compile(r"^[^\W_]+$", flags=re.UNICODE)


def is_valid_clue(clue: str, board_words: List[str]) -> Tuple[bool, str]:
    """Validate a clue against the standard Codenames rules.

    Rules: non-empty, single token (no spaces), no hyphens, alphanumeric
    only, and not a word that currently sits on the board (case-insensitive).

    Returns
    -------
    (ok, error_message)
        ``error_message`` is ``""`` when ``ok`` is True.
    """
    if not isinstance(clue, str) or not clue.strip():
        return False, "Clue is empty"
    stripped = clue.strip()
    if " " in stripped:
        return False, "Clue contains a space (must be a single word)"
    if "-" in stripped:
        return False, "Clue contains a hyphen (must be a single bare word)"
    if not _ALPHANUMERIC_RE.match(stripped):
        return False, "Clue must be alphanumeric (letters and digits only)"
    board_lower = {w.lower() for w in board_words}
    if stripped.lower() in board_lower:
        return False, "Clue matches a word on the board"
    return True, ""
