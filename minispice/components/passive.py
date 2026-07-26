"""Passive two-terminal components: resistors, capacitors, inductors.

MiniSPICE currently only solves the DC operating point (no transient or
AC analysis), and that flattens capacitors and inductors into limiting
cases of elements already stamped elsewhere in this file/module:

- A capacitor's impedance, ``1 / (j*omega*C)``, goes to infinity as
  ``omega -> 0``: at DC it's an open circuit, i.e. zero conductance --
  exactly ``Resistor``'s stamp with ``g = 0``.
- An inductor's impedance, ``j*omega*L``, goes to zero as ``omega -> 0``:
  at DC it's a plain wire, forcing ``V_pos == V_neg`` regardless of
  current -- exactly what an ideal 0V voltage source enforces. Unlike
  the capacitor case, that can't be written as a conductance (a "wire"
  would need infinite conductance), so it reuses
  ``VoltageSource``'s stamp instead, and needs the same kind of
  auxiliary branch-current unknown a voltage source does (see
  ``MNASystem.__init__``).

Neither element's actual capacitance/inductance value matters for a DC
solve -- both would only come into play once transient/AC analysis
exists, since only there does ``omega`` stop being zero.
"""

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
        """Add this capacitor's DC contribution to the MNA system.

        At DC a capacitor is an open circuit (see module docstring):
        zero conductance. That's the same stamp a resistor uses, just
        with ``g = 0`` -- which, being zero, leaves the matrix
        unchanged. The capacitance value itself plays no role here; it
        only matters once transient/AC analysis exists.
        """
        system.stamp_conductance(self.node_pos, self.node_neg, 0.0)


class Inductor(TwoTerminalComponent):
    """A linear inductor, value given in henries."""

    def __init__(self, name: str, node_pos: str, node_neg: str, inductance: float) -> None:
        super().__init__(name, node_pos, node_neg, inductance)

    @property
    def inductance(self) -> float:
        return self.value

    def stamp(self, system: "MNASystem") -> None:
        """Add this inductor's DC contribution to the MNA system.

        At DC an inductor is a plain wire (see module docstring):
        ``V_pos == V_neg``, exactly what an ideal 0V voltage source
        enforces. So rather than a conductance, this reuses
        ``stamp_voltage_source`` with ``voltage=0.0`` -- which is also
        why ``MNASystem`` reserves an auxiliary branch-current unknown
        for every inductor exactly as it does for real voltage sources
        (see ``MNASystem.__init__``). The inductance value itself plays
        no role here; it only matters once transient/AC analysis exists.
        """
        system.stamp_voltage_source(self.node_pos, self.node_neg, self.name, 0.0)
