"""Passive two-terminal components: resistors, capacitors, inductors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TwoTerminalComponent

if TYPE_CHECKING:
    from minispice.solver.mna import MNASystem


class Resistor(TwoTerminalComponent):
    """A linear resistor, value given in ohms."""

    def __init__(self, name: str, node_pos: str, node_neg: str, resistance: float) -> None:
        super().__init__(name, node_pos, node_neg, resistance)

    @property
    def resistance(self) -> float:
        return self.value

    def stamp(self, system: "MNASystem") -> None:
        raise NotImplementedError


class Capacitor(TwoTerminalComponent):
    """A linear capacitor, value given in farads."""

    def __init__(self, name: str, node_pos: str, node_neg: str, capacitance: float) -> None:
        super().__init__(name, node_pos, node_neg, capacitance)

    @property
    def capacitance(self) -> float:
        return self.value

    def stamp(self, system: "MNASystem") -> None:
        raise NotImplementedError


class Inductor(TwoTerminalComponent):
    """A linear inductor, value given in henries."""

    def __init__(self, name: str, node_pos: str, node_neg: str, inductance: float) -> None:
        super().__init__(name, node_pos, node_neg, inductance)

    @property
    def inductance(self) -> float:
        return self.value

    def stamp(self, system: "MNASystem") -> None:
        raise NotImplementedError
