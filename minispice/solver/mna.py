"""Modified Nodal Analysis (MNA) solver.

This module defines the shape of the linear system (``MNASystem``), the
solved-result container (``MNASolution``), and the solver's public
interface (``MNASolver``). ``MNASolver.build_system()`` assembles the
system by stamping every component in a circuit; ``MNASolver.solve()``
solves it with ``numpy.linalg.solve`` and returns an ``MNASolution``
that exposes node voltages and branch currents by *name* rather than by
``MNASystem``'s internal matrix index. ``MNASystem`` supports the three
stamps every currently-implemented component reduces to:
``stamp_conductance`` (resistors, and capacitors at DC),
``stamp_voltage_source`` (independent voltage sources, and inductors at
DC), and ``stamp_current_source`` (independent current sources).

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

MiniSPICE only solves the DC operating point (no transient/AC analysis
yet), and reactive elements degenerate to something already on this
list at DC: a capacitor's impedance is infinite as frequency -> 0, i.e.
an open circuit (zero conductance -- see ``Capacitor.stamp()``); an
inductor's impedance is zero, i.e. a plain wire, which is exactly what
an ideal 0V voltage source enforces (see ``Inductor.stamp()``). Neither
element needed a new kind of stamp -- see the module docstring for
``minispice.components.passive`` for the reasoning in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from minispice.circuit import Circuit
from minispice.components.passive import Inductor
from minispice.components.sources import VoltageSource


class MNASystem:
    """The linear system ``A x = z`` produced by stamping a circuit.

    The unknown vector ``x`` is laid out as ``[node voltages | branch
    currents]``: the first ``circuit.num_nodes`` entries are node
    voltages (indexed via ``circuit.node_index()``), followed by one
    entry per element that needs an auxiliary branch-current unknown --
    every independent voltage source, plus every inductor (which, at
    DC, is stamped identically to a 0V voltage source; see
    ``Inductor.stamp()``) -- in the order those elements appear in
    ``circuit.components``. Reserving that space at construction time
    -- rather than growing the matrix as elements are stamped -- means
    every stamp only ever *fills in* entries of an already
    correctly-sized ``A``/``z``, regardless of the order components are
    stamped in.

    Attributes:
        circuit: The circuit this system was built for. Kept on the
            system so that a component's ``stamp()`` method can turn
            the node *names* it was constructed with (e.g. ``"in"``,
            ``"out"``) into matrix *row/column indices* by calling
            back into ``circuit.node_index()``.
        size: Total number of unknowns: node voltages plus one branch
            current per voltage source and per inductor.
        A: The (size x size) system matrix.
        z: The right-hand-side vector of length ``size``.
    """

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit

        # Reserve one auxiliary row/column per element that needs a
        # branch-current unknown -- voltage sources and (at DC)
        # inductors alike -- appended after all the node-voltage
        # unknowns, in the order those elements appear in the circuit.
        # Doing this scan up front means the matrix never needs to be
        # resized mid-stamp.
        node_count = circuit.num_nodes
        self._branch_index: Dict[str, int] = {}
        for component in circuit.components:
            if isinstance(component, (VoltageSource, Inductor)):
                self._branch_index[component.name] = node_count + len(self._branch_index)

        self.size = node_count + len(self._branch_index)
        self.A = np.zeros((self.size, self.size))
        self.z = np.zeros(self.size)

    def branch_index(self, name: str) -> int:
        """Row/column reserved for a named element's branch current.

        Applies to any voltage source or inductor. Raises ``KeyError``
        if ``name`` doesn't refer to one that was present in the
        circuit when this system was constructed.
        """
        return self._branch_index[name]

    @property
    def voltage_source_names(self) -> List[str]:
        """Names of every voltage source with a reserved branch row, in index order."""
        return [c.name for c in self.circuit.components if isinstance(c, VoltageSource)]

    @property
    def inductor_names(self) -> List[str]:
        """Names of every inductor with a reserved DC branch row, in index order.

        At DC, an inductor needs an auxiliary branch-current unknown
        for the same reason a voltage source does -- see
        ``Inductor.stamp()`` and ``stamp_voltage_source``.
        """
        return [c.name for c in self.circuit.components if isinstance(c, Inductor)]

    def unknown_labels(self) -> List[str]:
        """A human-readable name for each row/column of ``A`` (and each
        entry of ``z``/a solved ``x``), in the exact order they appear
        in the unknown vector.

        Node-voltage unknowns are labeled ``"V(<node>)"``; voltage-source
        and inductor branch-current unknowns are labeled ``"I(<name>)"``.
        Meant for displaying the raw linear system to a human (e.g. a
        matrix-viewer UI) without them needing to know the index layout.
        """
        node_names = sorted(
            (node for node in self.circuit.list_nodes() if self.circuit.node_index(node) >= 0),
            key=self.circuit.node_index,
        )
        branch_names = sorted(self._branch_index, key=self._branch_index.get)
        return [f"V({name})" for name in node_names] + [f"I({name})" for name in branch_names]

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

        Also used, with ``voltage=0.0``, by ``Inductor.stamp()``: at DC
        an inductor is a plain wire, which is exactly what a 0V ideal
        source enforces (``V_pos == V_neg``) -- see the module
        docstring and ``Inductor.stamp()`` for the full reasoning.

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


@dataclass
class MNASolution:
    """The DC operating-point solution of a circuit.

    Wraps the raw unknown vector produced by ``numpy.linalg.solve`` so
    callers look results up by node/source *name* instead of needing to
    know ``MNASystem``'s ``[node voltages | branch currents]`` index
    layout.

    Attributes:
        node_voltages: Every node's DC voltage, keyed by name --
            including ground, which is always ``0.0``.
        source_currents: The DC current through every independent
            voltage source, keyed by name. Sign convention: current is
            defined flowing from ``node_pos`` to ``node_neg`` through
            the source (see ``VoltageSource`` /
            ``MNASystem.stamp_voltage_source``).
        inductor_currents: The DC current through every inductor, keyed
            by name, using the same ``node_pos -> node_neg`` sign
            convention. An inductor is not a source, so it gets its own
            dict rather than being folded into ``source_currents`` --
            but under the hood it shares the same kind of matrix
            row/column (see ``Inductor.stamp()``).
        raw: The raw solution vector, in ``MNASystem``'s index layout --
            kept for advanced use; the named dicts above cover ordinary
            usage.
    """

    node_voltages: Dict[str, float]
    source_currents: Dict[str, float]
    inductor_currents: Dict[str, float]
    raw: np.ndarray

    def voltage(self, node: str) -> float:
        """Equivalent to ``node_voltages[node]``."""
        return self.node_voltages[node]

    def current(self, source_name: str) -> float:
        """Equivalent to ``source_currents[source_name]``."""
        return self.source_currents[source_name]

    @classmethod
    def from_system(cls, system: "MNASystem", x: np.ndarray) -> "MNASolution":
        """Unpack a raw solved unknown vector ``x`` for ``system`` into
        a name-keyed ``MNASolution``.

        Factored out of ``MNASolver.solve()`` so anything else that
        already has an ``MNASystem`` and a solved ``x`` -- e.g. a caller
        that also wants direct access to ``system.A``/``system.z``, such
        as a matrix-viewer UI -- can reuse this instead of re-deriving
        the same node/branch lookups.
        """
        circuit = system.circuit
        node_voltages = {
            node: 0.0 if circuit.node_index(node) < 0 else float(x[circuit.node_index(node)])
            for node in circuit.list_nodes()
        }
        source_currents = {
            name: float(x[system.branch_index(name)]) for name in system.voltage_source_names
        }
        inductor_currents = {
            name: float(x[system.branch_index(name)]) for name in system.inductor_names
        }
        return cls(
            node_voltages=node_voltages,
            source_currents=source_currents,
            inductor_currents=inductor_currents,
            raw=x,
        )


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
        branch-current unknowns (see ``MNASystem.__init__``), then asks
        each component to add its own contribution via ``stamp()``, in
        the order the components appear in ``circuit.components``.
        Stamp order doesn't matter for correctness -- every stamp only
        adds (``+=``/``-=``) to matrix entries, so the result is the
        same regardless of which component is stamped first.
        """
        system = MNASystem(self.circuit)
        for component in self.circuit.components:
            component.stamp(system)
        return system

    def solve(self) -> MNASolution:
        """Solve the DC operating point and return a structured MNASolution.

        Assembles the system with ``build_system()`` and solves the
        linear system ``A x = z`` for ``x`` using ``numpy.linalg.solve``
        -- the raw ``x`` is laid out as described in ``MNASystem``: node
        voltages first (in ``Circuit.node_index()`` order), followed by
        one branch current per voltage source and per inductor. That
        raw vector is then unpacked into an ``MNASolution`` keyed by
        node/element name, so callers don't need to know anything about
        the index layout.

        Raises ``numpy.linalg.LinAlgError`` if ``A`` is singular -- e.g.
        a node with no DC path to ground, or a circuit with no voltage
        reference at all.
        """
        system = self.build_system()
        x = np.linalg.solve(system.A, system.z)
        return MNASolution.from_system(system, x)
