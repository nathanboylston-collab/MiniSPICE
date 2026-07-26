"""Human-readable, engineering-notation formatting of solved circuits.

``MNASolution`` (see ``minispice.solver.mna``) is a plain data container:
node voltages and voltage-source currents, keyed by name, as raw floats
in base SI units. ``SimulationResult`` builds on top of it for display
purposes -- deriving the per-resistor quantities MNA's unknown vector
doesn't carry (current and power dissipation, both computable from
Ohm's law once node voltages are known), and rendering everything in
engineering notation (``5 V``, ``2.5 mA``, ``6.25 mW``) instead of raw
floats like ``0.00625``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minispice.circuit import Circuit
from minispice.components import Resistor
from minispice.solver.mna import MNASolution

# Engineering notation scales in powers of 1000, largest first. The
# formatter picks the largest prefix whose scale the value's magnitude
# reaches, so e.g. 2500 V becomes "2.5 kV" rather than "2500 V", and
# 0.0025 A becomes "2.5 mA" rather than "0.0025 A".
_ENGINEERING_PREFIXES = [
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "u"),
    (1e-9, "n"),
    (1e-12, "p"),
]


def format_engineering(value: float, unit: str) -> str:
    """Format ``value`` in engineering notation with the given SI unit.

    E.g. ``format_engineering(0.0025, "A")`` -> ``"2.5 mA"``,
    ``format_engineering(2500.0, "V")`` -> ``"2.5 kV"``.
    """
    if value == 0.0:
        return f"0 {unit}"

    magnitude = abs(value)
    scale, prefix = _ENGINEERING_PREFIXES[-1]
    for candidate_scale, candidate_prefix in _ENGINEERING_PREFIXES:
        if magnitude >= candidate_scale:
            scale, prefix = candidate_scale, candidate_prefix
            break

    return f"{value / scale:g} {prefix}{unit}"


@dataclass
class SimulationResult:
    """Readable, engineering-notation view of a solved circuit.

    Wraps the ``Circuit`` together with the ``MNASolution`` it produced,
    so it can derive per-resistor current (``I = (V_pos - V_neg) / R``)
    and power dissipation (``P = V^2 / R``) -- quantities that live
    outside MNA's unknown vector, since they're a function of node
    voltages rather than unknowns themselves.
    """

    circuit: Circuit
    solution: MNASolution

    @property
    def node_voltages(self) -> Dict[str, float]:
        """Every node's DC voltage, keyed by name (see ``MNASolution``)."""
        return self.solution.node_voltages

    def _resistors(self) -> List[Resistor]:
        return [c for c in self.circuit.components if isinstance(c, Resistor)]

    @property
    def resistor_currents(self) -> Dict[str, float]:
        """Current through each resistor, keyed by name.

        Follows the same sign convention used throughout MNA stamping:
        current is positive when flowing from ``node_pos`` to
        ``node_neg``.
        """
        return {
            r.name: (self.solution.voltage(r.node_pos) - self.solution.voltage(r.node_neg))
            / r.resistance
            for r in self._resistors()
        }

    @property
    def resistor_power(self) -> Dict[str, float]:
        """Power dissipated by each resistor (``V^2 / R``), keyed by name.

        Always non-negative, regardless of current direction, since
        dissipation doesn't depend on the sign convention above.
        """
        return {
            r.name: (self.solution.voltage(r.node_pos) - self.solution.voltage(r.node_neg)) ** 2
            / r.resistance
            for r in self._resistors()
        }

    def describe(self) -> str:
        """Build (and print) a readable engineering-notation summary."""
        lines: List[str] = [f"Simulation Results: {self.circuit.title or '(untitled)'}"]

        lines.append("")
        lines.append("Node Voltages:")
        for node in self.circuit.list_nodes():
            lines.append(f"  {node:<8} {format_engineering(self.node_voltages[node], 'V')}")

        resistor_currents = self.resistor_currents
        resistor_power = self.resistor_power
        if resistor_currents:
            lines.append("")
            lines.append("Resistor Currents:")
            for name, current in resistor_currents.items():
                lines.append(f"  {name:<8} {format_engineering(current, 'A')}")

            lines.append("")
            lines.append("Resistor Power Dissipation:")
            for name, power in resistor_power.items():
                lines.append(f"  {name:<8} {format_engineering(power, 'W')}")

        summary = "\n".join(lines)
        print(summary)
        return summary
