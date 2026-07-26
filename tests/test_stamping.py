"""Tests for Resistor.stamp() and the MNASystem conductance stamp it uses.

These exercise the matrix-assembly logic directly (constructing an
MNASystem and calling stamp() by hand) rather than through the solver,
since MNASolver.build_system()/solve() are still unimplemented
placeholders.
"""

import pytest

from minispice.circuit import Circuit
from minispice.components import Resistor, VoltageSource
from minispice.solver.mna import MNASystem


def test_resistor_between_two_nodes_stamps_symmetric_entries():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "a", "b", 1000.0))
    system = MNASystem(circuit)

    circuit.components[0].stamp(system)

    g = 1.0 / 1000.0
    a, b = circuit.node_index("a"), circuit.node_index("b")
    assert system.A[a, a] == pytest.approx(g)
    assert system.A[b, b] == pytest.approx(g)
    assert system.A[a, b] == pytest.approx(-g)
    assert system.A[b, a] == pytest.approx(-g)


def test_resistor_to_ground_only_touches_its_own_diagonal():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "a", "0", 500.0))
    system = MNASystem(circuit)

    circuit.components[0].stamp(system)

    g = 1.0 / 500.0
    a = circuit.node_index("a")
    assert system.size == 1  # ground never gets a row/column
    assert system.A[a, a] == pytest.approx(g)


def test_parallel_resistors_accumulate_conductance():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "a", "b", 1000.0))
    circuit.add_component(Resistor("R2", "a", "b", 1000.0))
    system = MNASystem(circuit)

    for component in circuit.components:
        component.stamp(system)

    combined_g = 1.0 / 1000.0 + 1.0 / 1000.0
    a, b = circuit.node_index("a"), circuit.node_index("b")
    assert system.A[a, a] == pytest.approx(combined_g)
    assert system.A[b, b] == pytest.approx(combined_g)
    assert system.A[a, b] == pytest.approx(-combined_g)
    assert system.A[b, a] == pytest.approx(-combined_g)


def test_voltage_divider_matrix_matches_expected_kcl_equations():
    """R1 in-out (1k), R2 out-0 (1k): the textbook two-resistor divider.

    Only the resistors are stamped (VoltageSource.stamp() is still a
    placeholder); this checks exactly the 2x2 conductance matrix
    described in the stamping walkthrough.
    """
    circuit = Circuit(title="Voltage Divider")
    circuit.add_component(VoltageSource("V1", "in", "0", 5.0))
    r1 = Resistor("R1", "in", "out", 1000.0)
    r2 = Resistor("R2", "out", "0", 1000.0)
    circuit.add_component(r1)
    circuit.add_component(r2)

    system = MNASystem(circuit)
    r1.stamp(system)
    r2.stamp(system)

    in_idx = circuit.node_index("in")
    out_idx = circuit.node_index("out")
    assert (in_idx, out_idx) == (0, 1)
    assert system.size == 2

    g = 1.0 / 1000.0
    expected = [
        [g, -g],
        [-g, 2 * g],
    ]
    assert system.A[in_idx, in_idx] == pytest.approx(expected[0][0])
    assert system.A[in_idx, out_idx] == pytest.approx(expected[0][1])
    assert system.A[out_idx, in_idx] == pytest.approx(expected[1][0])
    assert system.A[out_idx, out_idx] == pytest.approx(expected[1][1])


def test_mna_system_size_excludes_ground():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "a", "b", 100.0))

    system = MNASystem(circuit)

    assert system.size == circuit.num_nodes == 2
    assert system.A.shape == (2, 2)
    assert system.z.shape == (2,)
