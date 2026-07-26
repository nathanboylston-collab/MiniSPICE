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
        """Add this resistor's conductance to the MNA system.

        A resistor obeys Ohm's law: the current flowing from
        ``node_pos`` to ``node_neg`` is ``g * (V_pos - V_neg)``, where
        ``g = 1 / resistance`` is the conductance. That is exactly the
        relationship ``MNASystem.stamp_conductance`` encodes, so the
        resistor's only job is to compute ``g`` and hand its two node
        *names* off to the system -- ``stamp_conductance`` (via
        ``Circuit.node_index()``) is what figures out which row/column
        of the matrix each name corresponds to, or skips a terminal
        entirely if it is tied to ground.
        """
        conductance = 1.0 / self.resistance
        system.stamp_conductance(self.node_pos, self.node_neg, conductance)


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
