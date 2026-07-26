"""Tests for Resistor.stamp()/VoltageSource.stamp() and the MNASystem
stamps they use (stamp_conductance, stamp_voltage_source).

These exercise the matrix-assembly logic directly (constructing an
MNASystem and calling stamp() by hand) rather than through the solver,
since MNASolver.build_system()/solve() are still unimplemented
placeholders.
"""

import numpy as np
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


def test_mna_system_size_excludes_ground():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "a", "b", 100.0))

    system = MNASystem(circuit)

    assert system.size == circuit.num_nodes == 2
    assert system.A.shape == (2, 2)
    assert system.z.shape == (2,)


# ---------------------------------------------------------------------------
# VoltageSource.stamp() / MNASystem.stamp_voltage_source()
# ---------------------------------------------------------------------------


def test_mna_system_reserves_one_row_per_voltage_source():
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", "a", "0", 5.0))
    circuit.add_component(Resistor("R1", "a", "0", 1000.0))

    system = MNASystem(circuit)

    # 1 node ("a") + 1 voltage source branch current.
    assert system.size == 2
    assert system.branch_index("V1") == 1


def test_voltage_source_to_ground_stamps_single_row_and_column():
    circuit = Circuit()
    v1 = VoltageSource("V1", "a", "0", 5.0)
    circuit.add_component(v1)
    system = MNASystem(circuit)

    v1.stamp(system)

    a = circuit.node_index("a")
    branch = system.branch_index("V1")

    # Branch column, wired into node "a"'s KCL row; ground contributes
    # no row/column so its entries are simply absent (left at 0).
    assert system.A[a, branch] == pytest.approx(1.0)
    assert system.A[branch, a] == pytest.approx(1.0)
    # The branch's own row is the constraint V_a - V_ground = 5.
    assert system.z[branch] == pytest.approx(5.0)


def test_voltage_source_between_two_nodes_stamps_both_terminals():
    circuit = Circuit()
    v1 = VoltageSource("V1", "a", "b", 12.0)
    circuit.add_component(v1)
    system = MNASystem(circuit)

    v1.stamp(system)

    a, b = circuit.node_index("a"), circuit.node_index("b")
    branch = system.branch_index("V1")

    assert system.A[a, branch] == pytest.approx(1.0)
    assert system.A[branch, a] == pytest.approx(1.0)
    assert system.A[b, branch] == pytest.approx(-1.0)
    assert system.A[branch, b] == pytest.approx(-1.0)
    assert system.z[branch] == pytest.approx(12.0)


def test_full_voltage_divider_augmented_system_solves_correctly():
    """The complete worked example: V1 (5V) feeding R1/R2 (1k each).

    Builds the full 3x3 augmented MNA system (2 node voltages + 1
    voltage-source branch current), stamps every component, and solves
    it with plain linear algebra to confirm the assembled system
    actually describes the circuit: Vout should come out to 2.5V (the
    textbook divider result) and the source current to -2.5mA (current
    is defined flowing node_pos->node_neg through the source, which is
    the opposite of the direction current actually flows here).
    """
    circuit = Circuit(title="Voltage Divider")
    v1 = VoltageSource("V1", "in", "0", 5.0)
    r1 = Resistor("R1", "in", "out", 1000.0)
    r2 = Resistor("R2", "out", "0", 1000.0)
    circuit.add_component(v1)
    circuit.add_component(r1)
    circuit.add_component(r2)

    system = MNASystem(circuit)
    for component in circuit.components:
        component.stamp(system)

    in_idx = circuit.node_index("in")
    out_idx = circuit.node_index("out")
    branch = system.branch_index("V1")
    assert (in_idx, out_idx, branch) == (0, 1, 2)
    assert system.size == 3

    g = 1.0 / 1000.0
    expected_A = np.array(
        [
            [g, -g, 1.0],
            [-g, 2 * g, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    expected_z = np.array([0.0, 0.0, 5.0])
    assert system.A == pytest.approx(expected_A)
    assert system.z == pytest.approx(expected_z)

    solution = np.linalg.solve(system.A, system.z)
    assert solution[in_idx] == pytest.approx(5.0)
    assert solution[out_idx] == pytest.approx(2.5)
    assert solution[branch] == pytest.approx(-0.0025)
