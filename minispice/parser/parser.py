"""Parser for a small subset of SPICE netlist syntax.

Supported element lines (one per line, whitespace-separated):

    R<name> node+ node- value      resistor
    C<name> node+ node- value      capacitor
    L<name> node+ node- value      inductor
    V<name> node+ node- value      independent DC voltage source
    I<name> node+ node- value      independent DC current source

The first non-blank, non-comment line is treated as the circuit title,
matching standard SPICE convention. Comment lines start with ``*``, and
``.end`` (case-insensitive) terminates the netlist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Union

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
    """Raised when a netlist cannot be parsed."""


class SpiceParser:
    """Parses SPICE-like netlist text into a :class:`Circuit`."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[str, List[str]], Component]] = {
            "R": self._build_resistor,
            "C": self._build_capacitor,
            "L": self._build_inductor,
            "V": self._build_voltage_source,
            "I": self._build_current_source,
        }

    def parse_file(self, path: Union[str, Path]) -> Circuit:
        text = Path(path).read_text()
        return self.parse_text(text)

    def parse_text(self, text: str) -> Circuit:
        lines = text.splitlines()

        title = ""
        circuit: Circuit | None = None

        for raw_line in lines:
            line = raw_line.strip()

            if not line or line.startswith("*"):
                continue
            if line.lower().startswith(".end"):
                break

            if circuit is None:
                title = line
                circuit = Circuit(title=title)
                continue

            component = self._parse_element_line(line)
            circuit.add_component(component)

        if circuit is None:
            circuit = Circuit(title=title)

        return circuit

    def _parse_element_line(self, line: str) -> Component:
        tokens = line.split()
        name = tokens[0]
        designator = name[0].upper()

        builder = self._builders.get(designator)
        if builder is None:
            raise SpiceParseError(f"unsupported element type: {name!r}")

        try:
            return builder(name, tokens[1:])
        except (IndexError, ValueError) as exc:
            raise SpiceParseError(f"malformed element line: {line!r}") from exc

    @staticmethod
    def _build_resistor(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value = args[0], args[1], args[2]
        return Resistor(name, node_pos, node_neg, parse_value(value))

    @staticmethod
    def _build_capacitor(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value = args[0], args[1], args[2]
        return Capacitor(name, node_pos, node_neg, parse_value(value))

    @staticmethod
    def _build_inductor(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value = args[0], args[1], args[2]
        return Inductor(name, node_pos, node_neg, parse_value(value))

    @staticmethod
    def _build_voltage_source(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value = args[0], args[1], args[2]
        return VoltageSource(name, node_pos, node_neg, parse_value(value))

    @staticmethod
    def _build_current_source(name: str, args: List[str]) -> Component:
        node_pos, node_neg, value = args[0], args[1], args[2]
        return CurrentSource(name, node_pos, node_neg, parse_value(value))
