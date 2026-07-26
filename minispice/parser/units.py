"""Parsing of SPICE-style numeric values with engineering suffixes."""

from __future__ import annotations

import re

# Longest/most specific suffixes must be checked before their prefixes
# (e.g. "meg" before "m") since matching is done via a plain string suffix
# search rather than a single regex alternation with backtracking.
_SUFFIXES = [
    ("meg", 1e6),
    ("t", 1e12),
    ("g", 1e9),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
]

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def parse_value(text: str) -> float:
    """Parse a SPICE numeric literal such as ``4.7k``, ``10u``, ``1e-9``.

    Trailing unit letters after a recognized suffix (e.g. the "F" in
    "10uF") are ignored, matching common SPICE conventions.
    """
    text = text.strip()
    match = _NUMBER_RE.match(text)
    if not match:
        raise ValueError(f"invalid numeric value: {text!r}")

    number = float(match.group(0))
    rest = text[match.end():].strip().lower()

    for suffix, multiplier in _SUFFIXES:
        if rest.startswith(suffix):
            return number * multiplier

    return number
