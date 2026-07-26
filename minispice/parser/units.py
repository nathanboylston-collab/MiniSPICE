"""Parsing of SPICE-style numeric values with engineering suffixes.

SPICE netlists write component values as a plain number optionally
followed by an engineering-scale suffix and, sometimes, a cosmetic unit
name that carries no numeric meaning (e.g. the "F" in "10uF" or the
"Hz" in "1kHz"). This module turns that textual convention into a
plain ``float`` that the rest of MiniSPICE works with.

The conversion happens in three steps, all inside :func:`parse_value`:

1. A regex pulls the leading numeric literal (optional sign, decimal
   point, and exponent) off the front of the token, e.g. "4.7" out of
   "4.7k".
2. Whatever text remains is matched, longest-suffix-first, against a
   table of SPICE engineering-scale suffixes (``k``, ``meg``, ``u``,
   ...). "meg" must be checked before "m", or "10meg" would be
   misread as "10m" (milli) followed by a stray, unrecognized "eg".
3. Anything left over after the suffix is removed must be a cosmetic
   unit name -- letters only (e.g. "F", "Hz") -- and is discarded. If
   it contains digits or symbols instead (a likely typo, such as
   "10k5"), or if no suffix matched and the leftover text isn't a
   plain unit name at all, :func:`parse_value` raises ``ValueError``
   rather than silently truncating the mistake.
"""

from __future__ import annotations

import re

# Ordered longest/most-specific suffix first: "meg" must be tested
# before "m", "t", "g", etc., otherwise "10meg" would match the
# single-letter "m" (milli) suffix first and leave a dangling,
# unrecognized "eg" behind it.
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

# Leading signed float/int, with optional exponent: 5, -3.2, .5, 1e-9
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")

# A cosmetic unit name left after the number (and optional scale
# suffix) is stripped -- e.g. the "F" in "10uF" or the "Hz" in "1kHz".
# SPICE ignores these entirely; we only require that they be letters,
# so a genuine typo (stray digits or symbols) is rejected instead of
# being silently swallowed.
_UNIT_RE = re.compile(r"^[a-zA-Z]*$")


def parse_value(text: str) -> float:
    """Parse a SPICE numeric literal such as ``4.7k``, ``10u``, ``1e-9``.

    Returns the value as a plain float in base units (ohms, farads,
    henries, volts, or amps, depending on the caller's context).

    Raises:
        ValueError: if ``text`` isn't a recognizable SPICE numeric
            literal -- e.g. it has no leading number, or it has
            trailing characters that are neither a known engineering
            suffix nor a plain alphabetic unit name.
    """
    original = text
    text = text.strip()
    if not text:
        raise ValueError("empty numeric value")

    # Step 1: pull the leading number off the front of the token.
    match = _NUMBER_RE.match(text)
    if not match:
        raise ValueError(f"invalid numeric value: {original!r}")

    number = float(match.group(0))
    remainder = text[match.end():]

    # Step 2: check the text right after the number against each known
    # engineering-scale suffix, longest first (see _SUFFIXES comment).
    for suffix, multiplier in _SUFFIXES:
        if remainder[: len(suffix)].lower() == suffix:
            unit = remainder[len(suffix):]
            # Step 3: whatever is left must be a plain unit name.
            if not _UNIT_RE.match(unit):
                raise ValueError(
                    f"invalid unit suffix {remainder!r} in value: {original!r}"
                )
            return number * multiplier

    # No recognized magnitude suffix: the remainder (if any) must still
    # be a bare cosmetic unit name (e.g. "V", "Hz"), applied with no
    # scale factor. Anything else -- digits, symbols, unrecognized
    # letter runs mixed with punctuation -- is treated as malformed.
    if not _UNIT_RE.match(remainder):
        raise ValueError(f"invalid unit suffix {remainder!r} in value: {original!r}")

    return number
