"""Modified Nodal Analysis (MNA) solver.

This module defines the shape of the linear system (``MNASystem``) and
the solver's public interface (``MNASolver``). Full circuit assembly
and solving are not implemented yet -- ``MNASolver.build_system()`` and
``MNASolver.solve()`` remain placeholders -- but ``MNASystem`` already
supports the conductance stamp used by purely resistive elements, since
that is enough to demonstrate (and test) how a single component's
``stamp()`` method places values into the matrix.

Background: MNA builds a linear system ``A x = z`` where the unknown
vector ``x`` holds one entry per non-ground node voltage, plus one
extra entry per independent voltage source (and other elements that
require a branch-current unknown, e.g. inductors in some formulations).
Each component contributes ("stamps") entries into ``A`` and ``z``
based on its type and connectivity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from minispice.circuit import Circuit


class MNASystem:
    """The linear system ``A x = z`` produced by stamping a circuit.

    Attributes:
        circuit: The circuit this system was built for. Kept on the
            system so that a component's ``stamp()`` method can turn
            the node *names* it was constructed with (e.g. ``"in"``,
            ``"out"``) into matrix *row/column indices* by calling
            back into ``circuit.node_index()``.
        size: Total number of unknowns. Currently just
            ``circuit.num_nodes`` (one per non-ground node voltage);
            this will grow once elements that need an auxiliary
            branch-current unknown (independent voltage sources,
            inductors in some formulations) are stamped.
        A: The (size x size) system matrix.
        z: The right-hand-side vector of length ``size``.
    """

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit
        self.size = circuit.num_nodes
        self.A = np.zeros((self.size, self.size))
        self.z = np.zeros(self.size)

    def _unknown_index(self, node: str) -> Optional[int]:
        """Translate a node name into a matrix row/column, or ``None``.

        Delegates to ``Circuit.node_index()``, which returns ``-1``
        for the ground node and a 0-based index for every other node.
        Ground has no row or column in the MNA matrix -- its voltage
        is the fixed 0V reference, not an unknown -- so the ``-1``
        sentinel is remapped to ``None`` here, making "this terminal
        has no matrix entry" explicit at every call site instead of
        relying on Python silently treating ``-1`` as a valid (and
        very wrong) index.
        """
        index = self.circuit.node_index(node)
        return None if index < 0 else index

    def stamp_conductance(self, node_a: str, node_b: str, conductance: float) -> None:
        """Add a conductance ``g`` between ``node_a`` and ``node_b``.

        This is the standard MNA stamp shared by any purely resistive
        two-terminal element carrying a current of ``g * (V_a - V_b)``
        from ``node_a`` to ``node_b``. Writing Kirchhoff's Current Law
        at each node in terms of the unknown node voltages gives four
        symmetric contributions:

            A[a][a] += g
            A[b][b] += g
            A[a][b] -= g
            A[b][a] -= g

        If a terminal is tied to ground, ``_unknown_index`` returns
        ``None`` for it: that terminal contributes no row/column at
        all, so both the diagonal entry for that terminal and the two
        off-diagonal entries linking it to the other terminal are
        skipped, leaving only the other terminal's diagonal entry
        updated.
        """
        idx_a = self._unknown_index(node_a)
        idx_b = self._unknown_index(node_b)

        if idx_a is not None:
            self.A[idx_a, idx_a] += conductance
        if idx_b is not None:
            self.A[idx_b, idx_b] += conductance
        if idx_a is not None and idx_b is not None:
            self.A[idx_a, idx_b] -= conductance
            self.A[idx_b, idx_a] -= conductance


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
