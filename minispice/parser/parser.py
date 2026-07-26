"""Parser for a small subset of SPICE netlist syntax.

Supported element lines (one per line, whitespace-separated):

    R<name> node+ node- value      resistor (value must be > 0)
    C<name> node+ node- value      capacitor (value must be > 0)
    L<name> node+ node- value      inductor (value must be > 0)
    V<name> node+ node- value      independent DC voltage source
    I<name> node+ node- value      independent DC current source

The first non-blank, non-comment line is treated as the circuit title,
matching standard SPICE convention. Full-line comments start with
``*``; inline comments start with ``;`` and run to the end of the
line. ``.end`` (case-insensitive) terminates the netlist.

--------------------------------------------------------------------
How a netlist becomes Python objects
--------------------------------------------------------------------
Turning SPICE text into the object graph the rest of MiniSPICE
operates on happens in four stages, mirrored by the methods below:

1. **Line splitting & cleanup** (:meth:`SpiceParser.parse_text`): the
   raw text is split into physical lines. Each line has any inline
   ``;`` comment chopped off and is stripped of surrounding
   whitespace. Blank lines and full-line ``*`` comments are skipped
   entirely -- they never reach the element parser. The very first
   surviving line is special-cased as the deck title and used to
   construct the :class:`~minispice.circuit.Circuit` container;
   everything after that is a candidate element line, until ``.end``
   is seen.

2. **Tokenization & dispatch** (:meth:`SpiceParser._parse_element_line`):
   each element line is split on whitespace into tokens. The first
   character of the first token (e.g. the ``R`` in ``R1``) is the
   SPICE "designator" that says what kind of device the line
   describes; it is looked up in ``self._builders`` to find the
   right factory function. This mirrors real SPICE, where the letter
   prefix of an element's name -- not any keyword -- determines its
   type.

3. **Field validation & construction** (the ``_build_*`` static
   methods, via the shared :meth:`SpiceParser._split_two_terminal`
   helper): the remaining tokens are validated as exactly two node
   names and one value field, rejecting a component that references
   the same node twice (a short that contributes nothing to the
   circuit) or has the wrong number of fields. The value field's text
   (e.g. ``"4.7k"``) is converted to a plain float by
   :func:`minispice.parser.units.parse_value`. The result is a fully
   constructed :class:`~minispice.components.base.Component`
   subclass instance (e.g. :class:`~minispice.components.Resistor`).

4. **Registration** (:meth:`SpiceParser.parse_text` again): each
   constructed component is checked against the set of component
   names already seen (SPICE names must be unique) and then handed to
   :meth:`Circuit.add_component <minispice.circuit.Circuit.add_component>`,
   which is what actually discovers and numbers the circuit's nodes.
   The parser itself never assigns node indices -- that bookkeeping
   belongs to ``Circuit``.

Any failure along the way -- an unrecognized designator, a malformed
value, a wrong field count, a duplicate name -- is reported as a
:class:`SpiceParseError` that includes the offending line number, so a
user editing a netlist by hand gets a precise pointer to the mistake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple, Union

from minispice.circuit import Circuit
from minispice.components import (
    Capacitor,
    CurrentSource,
    Inductor,
    Resistor,
    VoltageSource,
)
from minispice.components.base import Component

from .units import parse_value


class SpiceParseError(Exception):
    """Raised when a netlist cannot be parsed.

    The exception message always includes the 1-based source line
    number so a hand-edited netlist can be corrected quickly.
    """


class SpiceParser:
    """Parses SPICE-like netlist text into a :class:`Circuit`.

    An instance is stateless between calls to :meth:`parse_text` /
    :meth:`parse_file`; the only per-instance state is the dispatch
    table mapping element-name designators (``R``, ``C``, ``L``, ``V``,
    ``I``) to the factory function that builds the matching
    :class:`~minispice.components.base.Component` subclass.
    """

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[str, List[str]], Component]] = {
            "R": self._build_resistor,
            "C": self._build_capacitor,
            "L": self._build_inductor,
            "V": self._build_voltage_source,
            "I": self._build_current_source,
        }

    def parse_file(self, path: Union[str, Path]) -> Circuit:
        """Read a netlist file from disk and parse it."""
        text = Path(path).read_text()
        return self.parse_text(text)

    def parse_text(self, text: str) -> Circuit:
        """Parse full netlist text into a :class:`Circuit`.

        See the module docstring for a full description of the
        line-by-line conversion process.
        """
        circuit: Circuit | None = None
        seen_names: Set[str] = set()

        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            # Strip a trailing "; comment" (common ngspice convention)
            # before anything else, then apply classic SPICE rules:
            # blank lines and full-line "*" comments are ignored.
            line = raw_line.split(";", 1)[0].strip()

            if not line or line.startswith("*"):
                continue
            if line.lower().startswith(".end"):
                break

            if circuit is None:
                # The first substantive line of a SPICE deck is always
                # the title -- never an element line. This is the one
                # piece of positional (rather than prefix-based) syntax
                # in the format.
                circuit = Circuit(title=line)
                continue

            component = self._parse_element_line(line, line_no)

            # SPICE element names must be unique within a deck; SPICE
            # itself treats names case-insensitively, so we compare
            # upper-cased names while preserving the original casing
            # on the component for display purposes.
            key = component.name.upper()
            if key in seen_names:
                raise SpiceParseError(
                    f"line {line_no}: duplicate component name {component.name!r}"
                )
            seen_names.add(key)

            circuit.add_component(component)

        if circuit is None:
            # An empty (or all-comment) netlist still yields a valid,
            # if empty, circuit rather than raising.
            circuit = Circuit(title="")

        return circuit

    def _parse_element_line(self, line: str, line_no: int) -> Component:
        """Tokenize one element line and dispatch to its builder.

        The designator is the first character of the element's name
        (``R1`` -> ``R``), matching the SPICE convention that an
        element's type is encoded in its name rather than declared
        with a separate keyword.
        """
        tokens = line.split()
        name = tokens[0]
        designator = name[0].upper()

        builder = self._builders.get(designator)
        if builder is None:
            supported = ", ".join(sorted(self._builders))
            raise SpiceParseError(
                f"line {line_no}: unsupported element type {name!r} "
                f"(supported prefixes: {supported})"
            )

        try:
            return builder(name, tokens[1:])
        except (IndexError, ValueError) as exc:
            raise SpiceParseError(f"line {line_no}: {exc}") from exc

    @staticmethod
    def _split_two_terminal(name: str, args: List[str]) -> Tuple[str, str, str]:
        """Validate and unpack the common ``node+ node- value`` fields.

        Shared by every builder below since all currently-supported
        elements are two-terminal devices. Raises ``ValueError`` (which
        :meth:`_parse_element_line` turns into a ``SpiceParseError``)
        if the field count is wrong or the two nodes are identical --
        the latter describes a component shorted to itself, which
        contributes nothing to the circuit and is almost always a typo.
        """
        if len(args) != 3:
            raise ValueError(
                f"{name} expects exactly 3 fields (node+, node-, value), "
                f"got {len(args)}"
            )

        node_pos, node_neg, value_text = args
        if node_pos == node_neg:
            raise ValueError(f"{name} connects node {node_pos!r} to itself")

        return node_pos, node_neg, value_text

    @staticmethod
    def _build_resistor(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value_text = SpiceParser._split_two_terminal(name, args)
        resistance = parse_value(value_text)
        if resistance <= 0:
            raise ValueError(f"{name} resistance must be positive, got {resistance!r}")
        return Resistor(name, node_pos, node_neg, resistance)

    @staticmethod
    def _build_capacitor(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value_text = SpiceParser._split_two_terminal(name, args)
        capacitance = parse_value(value_text)
        if capacitance <= 0:
            raise ValueError(f"{name} capacitance must be positive, got {capacitance!r}")
        return Capacitor(name, node_pos, node_neg, capacitance)

    @staticmethod
    def _build_inductor(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value_text = SpiceParser._split_two_terminal(name, args)
        inductance = parse_value(value_text)
        if inductance <= 0:
            raise ValueError(f"{name} inductance must be positive, got {inductance!r}")
        return Inductor(name, node_pos, node_neg, inductance)

    @staticmethod
    def _build_voltage_source(name: str, args: List[str]) -> Component:
        # Unlike R/L/C, a source's value may legitimately be zero
        # (e.g. modeling a short/off state) or negative (reversed
        # polarity), so no positivity check applies here.
        node_pos, node_neg, value_text = SpiceParser._split_two_terminal(name, args)
        return VoltageSource(name, node_pos, node_neg, parse_value(value_text))

    @staticmethod
    def _build_current_source(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value_text = SpiceParser._split_two_terminal(name, args)
        return CurrentSource(name, node_pos, node_neg, parse_value(value_text))
