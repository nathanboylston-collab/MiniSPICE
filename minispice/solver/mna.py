"""Modified Nodal Analysis (MNA) solver.

This module currently only defines the shape of the linear system and
the solver's public interface. Actual matrix stamping and solving are
not implemented yet.

Background: MNA builds a linear system ``A x = z`` where the unknown
vector ``x`` holds one entry per non-ground node voltage, plus one
extra entry per independent voltage source (and other elements that
require a branch-current unknown, e.g. inductors in some formulations).
Each component contributes ("stamps") entries into ``A`` and ``z``
based on its type and connectivity.
"""

from __future__ import annotations

import numpy as np

from minispice.circuit import Circuit


class MNASystem:
    """The linear system ``A x = z`` produced by stamping a circuit.

    Attributes:
        size: Total number of unknowns (node voltages plus auxiliary
            branch currents).
        A: The (size x size) system matrix.
        z: The right-hand-side vector of length ``size``.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.A = np.zeros((size, size))
        self.z = np.zeros(size)


class MNASolver:
    """Builds and solves the MNA system for a given circuit.

    This is a placeholder: it establishes the interface future work will
    fill in (component stamping, DC operating point solve, and later
    transient/AC analyses).
    """

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit

    def build_system(self) -> MNASystem:
        """Construct the (currently empty) MNA system for the circuit."""
        raise NotImplementedError("MNA matrix stamping is not implemented yet")

    def solve(self) -> np.ndarray:
        """Solve the DC operating point and return the unknown vector."""
        raise NotImplementedError("MNA solving is not implemented yet")
