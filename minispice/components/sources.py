"""Independent voltage and current sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TwoTerminalComponent

if TYPE_CHECKING:
    from minispice.solver.mna import MNASystem


class VoltageSource(TwoTerminalComponent):
    """An independent DC voltage source, value given in volts.

    Voltage sources introduce an auxiliary branch current unknown in
    Modified Nodal Analysis, since (unlike a resistor) the current
    through an ideal source cannot be written as a function of its own
    terminal voltages. ``MNASystem`` reserves a row/column for that
    unknown for every voltage source in the circuit; ``stamp`` fills it
    in via ``MNASystem.stamp_voltage_source``.
    """

    def __init__(self, name: str, node_pos: str, node_neg: str, voltage: float) -> None:
        super().__init__(name, node_pos, node_neg, voltage)

    @property
    def voltage(self) -> float:
        return self.value

    def stamp(self, system: "MNASystem") -> None:
        system.stamp_voltage_source(self.node_pos, self.node_neg, self.name, self.voltage)


class CurrentSource(TwoTerminalComponent):
    """An independent DC current source, value given in amps.

    Current flows from ``node_pos`` to ``node_neg`` through the source.
    Unlike ``VoltageSource``, a current source's current is already
    known, so it needs no auxiliary branch-current unknown -- ``stamp``
    contributes only to the system's right-hand side, via
    ``MNASystem.stamp_current_source``.
    """

    def __init__(self, name: str, node_pos: str, node_neg: str, current: float) -> None:
        super().__init__(name, node_pos, node_neg, current)

    @property
    def current(self) -> float:
        return self.value

    def stamp(self, system: "MNASystem") -> None:
        system.stamp_current_source(self.node_pos, self.node_neg, self.current)
