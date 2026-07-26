"""Modified Nodal Analysis (MNA) solver.

This module defines the shape of the linear system (``MNASystem``) and
the solver's public interface (``MNASolver``). ``MNASolver.build_system()``
assembles the system by stamping every component in a circuit;
``MNASolver.solve()`` -- the actual linear-algebra solve -- is still a
placeholder. ``MNASystem`` supports the three stamps needed for simple
resistive DC circuits: ``stamp_conductance`` (resistors),
``stamp_voltage_source`` (independent voltage sources), and
``stamp_current_source`` (independent current sources). ``Capacitor``
and ``Inductor`` don't implement ``stamp()`` yet, so ``build_system()``
will raise ``NotImplementedError`` for circuits that contain one.

Background: plain nodal analysis writes one KCL equation per node, with
every branch current expressed as a function of node voltages (Ohm's
law for a resistor). That breaks down for an *ideal voltage source*: it
pins ``V_pos - V_neg`` to a fixed value, but the current flowing
through it is whatever the rest of the circuit demands -- it cannot be
written as a function of node voltages at all. MNA's fix, and the
"Modified" in its name, is to augment the unknown vector with one extra
entry per such element: its branch current. That turns the source's
voltage constraint into an explicit equation (an extra matrix row) and
its unknown current into an extra matrix column, wired into the KCL
rows of the two nodes it touches. ``MNASystem`` reserves that space up
front (see ``__init__``); ``stamp_voltage_source`` fills it in.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from minispice.circuit import Circuit
from minispice.components.sources import VoltageSource


class MNASystem:
    """The linear system ``A x = z`` produced by stamping a circuit.

    The unknown vector ``x`` is laid out as ``[node voltages | branch
    currents]``: the first ``circuit.num_nodes`` entries are node
    voltages (indexed via ``circuit.node_index()``), followed by one
    entry per independent voltage source in the circuit, in the order
    those sources appear in ``circuit.components``. Reserving that
    space at construction time -- rather than growing the matrix as
    sources are stamped -- means every stamp only ever *fills in*
    entries of an already correctly-sized ``A``/``z``, regardless of
    the order components are stamped in.

    Attributes:
        circuit: The circuit this system was built for. Kept on the
            system so that a component's ``stamp()`` method can turn
            the node *names* it was constructed with (e.g. ``"in"``,
            ``"out"``) into matrix *row/column indices* by calling
            back into ``circuit.node_index()``.
        size: Total number of unknowns: node voltages plus one branch
            current per independent voltage source.
        A: The (size x size) system matrix.
        z: The right-hand-side vector of length ``size``.
    """

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit

        # Reserve one auxiliary row/column per voltage source, appended
        # after all the node-voltage unknowns, in the order the sources
        # appear in the circuit. Doing this scan up front means the
        # matrix never needs to be resized mid-stamp.
        node_count = circuit.num_nodes
        self._branch_index: Dict[str, int] = {}
        for component in circuit.components:
            if isinstance(component, VoltageSource):
                self._branch_index[component.name] = node_count + len(self._branch_index)

        self.size = node_count + len(self._branch_index)
        self.A = np.zeros((self.size, self.size))
        self.z = np.zeros(self.size)

    def branch_index(self, name: str) -> int:
        """Row/column reserved for a named voltage source's branch current.

        Raises ``KeyError`` if ``name`` doesn't refer to a voltage
        source that was present in the circuit when this system was
        constructed.
        """
        return self._branch_index[name]

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

    def stamp_voltage_source(
        self, node_pos: str, node_neg: str, name: str, voltage: float
    ) -> None:
        """Add an ideal voltage source's contribution to the MNA system.

        An ideal source can't be stamped with a conductance -- its
        current isn't a function of its own terminal voltages -- so
        instead this fills in the branch-current column/row reserved
        for it at construction time (see ``branch_index``):

            A[pos][branch] += 1        A[branch][pos] += 1
            A[neg][branch] -= 1        A[branch][neg] -= 1
            z[branch]      += voltage

        The first two lines wire the new unknown, the source's branch
        current, into the KCL equations at its two nodes: by
        convention the current is defined as flowing from ``node_pos``
        to ``node_neg`` through the source, so it counts as current
        *leaving* ``node_pos`` (+1) and current *entering* ``node_neg``
        (-1) -- exactly the same sign convention used for a resistor's
        current in ``stamp_conductance``, just with an unknown in place
        of ``g * (V_a - V_b)``.

        The last two lines are the new equation itself, occupying the
        branch's own row: ``A[branch][pos] - A[branch][neg] = z[branch]``
        reads as ``V_pos - V_neg = voltage``, the source's defining
        constraint.

        As with ``stamp_conductance``, a terminal tied to ground has no
        row/column at all, so its two entries are simply skipped --
        note that the branch row's entry for that terminal is skipped
        too, since it is the same coefficient used symmetrically in
        both places.
        """
        idx_pos = self._unknown_index(node_pos)
        idx_neg = self._unknown_index(node_neg)
        branch = self.branch_index(name)

        if idx_pos is not None:
            self.A[idx_pos, branch] += 1.0
            self.A[branch, idx_pos] += 1.0
        if idx_neg is not None:
            self.A[idx_neg, branch] -= 1.0
            self.A[branch, idx_neg] -= 1.0

        self.z[branch] += voltage

    def stamp_current_source(self, node_pos: str, node_neg: str, current: float) -> None:
        """Add an independent current source's contribution to the system.

        Unlike a voltage source, a current source's current is already
        known -- it doesn't depend on any unknown node voltage, and it
        needs no auxiliary branch-current unknown of its own. So there
        is nothing to add to ``A``; the whole contribution lands on the
        right-hand side ``z``.

        By convention (matching the sign used for a resistor's or
        voltage source's branch current), ``current`` flows from
        ``node_pos`` to ``node_neg`` through the source: that is
        current *leaving* ``node_pos`` and current *entering*
        ``node_neg``. KCL says currents leaving a node sum to zero, so
        moving this known term to the other side of the equation gives:

            z[pos] -= current
            z[neg] += current

        As with the other stamps, a terminal tied to ground has no
        row at all, so its entry is simply skipped.
        """
        idx_pos = self._unknown_index(node_pos)
        idx_neg = self._unknown_index(node_neg)

        if idx_pos is not None:
            self.z[idx_pos] -= current
        if idx_neg is not None:
            self.z[idx_neg] += current


class MNASolver:
    """Builds and solves the MNA system for a given circuit.

    ``build_system()`` assembles the linear system; ``solve()`` runs the
    actual DC operating-point solve. (Transient/AC analyses are future
    work, not covered by either method.)
    """

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit

    def build_system(self) -> MNASystem:
        """Assemble the MNA system by stamping every component in turn.

        Creates an ``MNASystem`` sized for the circuit's nodes and
        voltage sources (see ``MNASystem.__init__``), then asks each
        component to add its own contribution via ``stamp()``, in the
        order the components appear in ``circuit.components``. Stamp
        order doesn't matter for correctness -- every stamp only adds
        (``+=``/``-=``) to matrix entries, so the result is the same
        regardless of which component is stamped first.

        Components whose ``stamp()`` is not yet implemented (currently
        ``Capacitor`` and ``Inductor``) raise ``NotImplementedError``
        from within ``stamp()`` itself; this method makes no attempt to
        catch or work around that; it simply propagates, since a
        circuit containing an unstamped component genuinely can't be
        assembled yet.
        """
        system = MNASystem(self.circuit)
        for component in self.circuit.components:
            component.stamp(system)
        return system

    def solve(self) -> np.ndarray:
        """Solve the DC operating point and return the unknown vector.

        Assembles the system with ``build_system()`` and solves the
        linear system ``A x = z`` for ``x`` using ``numpy.linalg.solve``.
        ``x`` is laid out exactly as described in ``MNASystem``: node
        voltages first (in ``Circuit.node_index()`` order), followed by
        one branch current per voltage source. Raises
        ``numpy.linalg.LinAlgError`` if ``A`` is singular -- e.g. a node
        with no DC path to ground, or a circuit with no voltage
        reference at all.
        """
        system = self.build_system()
        return np.linalg.solve(system.A, system.z)
