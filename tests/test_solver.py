import numpy as np
import pytest

from minispice.circuit import Circuit
from minispice.components import Capacitor, CurrentSource, Inductor, Resistor, VoltageSource
from minispice.solver import MNASolver


def test_build_system_on_empty_circuit_returns_zero_sized_system():
    system = MNASolver(Circuit()).build_system()

    assert system.size == 0
    assert system.A.shape == (0, 0)
    assert system.z.shape == (0,)


def test_build_system_stamps_a_single_resistor():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "a", "b", 1000.0))

    system = MNASolver(circuit).build_system()

    g = 1.0 / 1000.0
    a, b = circuit.node_index("a"), circuit.node_index("b")
    assert system.size == 2
    assert system.A[a, a] == pytest.approx(g)
    assert system.A[b, b] == pytest.approx(g)
    assert system.A[a, b] == pytest.approx(-g)
    assert system.A[b, a] == pytest.approx(-g)


def test_build_system_assembles_full_voltage_divider():
    """The same worked example as tests/test_stamping.py, but built end
    to end through MNASolver.build_system() instead of stamping by hand.
    """
    circuit = Circuit(title="Voltage Divider")
    circuit.add_component(VoltageSource("V1", "in", "0", 5.0))
    circuit.add_component(Resistor("R1", "in", "out", 1000.0))
    circuit.add_component(Resistor("R2", "out", "0", 1000.0))

    system = MNASolver(circuit).build_system()

    in_idx = circuit.node_index("in")
    out_idx = circuit.node_index("out")
    branch = system.branch_index("V1")
    assert system.size == 3

    solution = np.linalg.solve(system.A, system.z)
    assert solution[in_idx] == pytest.approx(5.0)
    assert solution[out_idx] == pytest.approx(2.5)
    assert solution[branch] == pytest.approx(-0.0025)


def test_build_system_stamps_current_source_into_resistor():
    circuit = Circuit()
    circuit.add_component(CurrentSource("I1", "0", "a", 0.001))
    circuit.add_component(Resistor("R1", "a", "0", 1000.0))

    system = MNASolver(circuit).build_system()

    solution = np.linalg.solve(system.A, system.z)
    a = circuit.node_index("a")
    assert solution[a] == pytest.approx(1.0)


def test_build_system_stamp_order_does_not_matter():
    forward = Circuit()
    forward.add_component(VoltageSource("V1", "in", "0", 5.0))
    forward.add_component(Resistor("R1", "in", "out", 1000.0))
    forward.add_component(Resistor("R2", "out", "0", 1000.0))

    backward = Circuit()
    backward.add_component(Resistor("R2", "out", "0", 1000.0))
    backward.add_component(Resistor("R1", "in", "out", 1000.0))
    backward.add_component(VoltageSource("V1", "in", "0", 5.0))

    forward_system = MNASolver(forward).build_system()
    backward_system = MNASolver(backward).build_system()

    forward_solution = np.linalg.solve(forward_system.A, forward_system.z)
    backward_solution = np.linalg.solve(backward_system.A, backward_system.z)

    assert forward_solution[forward.node_index("out")] == pytest.approx(
        backward_solution[backward.node_index("out")]
    )


def test_build_system_capacitor_to_ground_is_a_dc_open_circuit():
    """A capacitor stamps zero conductance at DC, so it should leave the
    matrix exactly as if it weren't in the circuit at all.
    """
    circuit = Circuit()
    circuit.add_component(Capacitor("C1", "a", "0", 1e-6))

    system = MNASolver(circuit).build_system()

    assert system.size == 1
    assert system.A == pytest.approx(np.zeros((1, 1)))


def test_build_system_inductor_reserves_a_branch_row_like_a_voltage_source():
    circuit = Circuit()
    circuit.add_component(Inductor("L1", "a", "0", 1e-3))
    circuit.add_component(Resistor("R1", "a", "0", 1000.0))

    system = MNASolver(circuit).build_system()

    # 1 node ("a") + 1 inductor branch current, same as a voltage source would reserve.
    assert system.size == 2
    assert system.branch_index("L1") == 1
    assert system.inductor_names == ["L1"]


def test_solve_on_empty_circuit_returns_ground_only_solution():
    solution = MNASolver(Circuit()).solve()

    # Circuit always registers ground, even with zero components.
    assert solution.node_voltages == {"0": 0.0}
    assert solution.source_currents == {}
    assert solution.inductor_currents == {}
    assert solution.raw.shape == (0,)


def test_solve_voltage_divider_matches_expected_node_voltages():
    circuit = Circuit(title="Voltage Divider")
    circuit.add_component(VoltageSource("V1", "in", "0", 5.0))
    circuit.add_component(Resistor("R1", "in", "out", 1000.0))
    circuit.add_component(Resistor("R2", "out", "0", 1000.0))

    solution = MNASolver(circuit).solve()

    assert solution.node_voltages["in"] == pytest.approx(5.0)
    assert solution.node_voltages["out"] == pytest.approx(2.5)
    assert solution.node_voltages["0"] == pytest.approx(0.0)
    assert solution.source_currents["V1"] == pytest.approx(-0.0025)
    assert solution.voltage("out") == pytest.approx(2.5)
    assert solution.current("V1") == pytest.approx(-0.0025)


def test_solve_current_source_into_resistor_matches_ohms_law():
    circuit = Circuit()
    circuit.add_component(CurrentSource("I1", "0", "a", 0.001))
    circuit.add_component(Resistor("R1", "a", "0", 1000.0))

    solution = MNASolver(circuit).solve()

    assert solution.node_voltages["a"] == pytest.approx(1.0)
    assert solution.source_currents == {}


def test_solve_capacitor_in_parallel_with_resistor_does_not_change_dc_result():
    """A capacitor is a DC open circuit, so adding one in parallel with
    a resistor should have zero effect on the operating point.
    """
    without_capacitor = Circuit(title="Voltage Divider")
    without_capacitor.add_component(VoltageSource("V1", "in", "0", 5.0))
    without_capacitor.add_component(Resistor("R1", "in", "out", 1000.0))
    without_capacitor.add_component(Resistor("R2", "out", "0", 1000.0))

    with_capacitor = Circuit(title="Voltage Divider with Capacitor")
    with_capacitor.add_component(VoltageSource("V1", "in", "0", 5.0))
    with_capacitor.add_component(Resistor("R1", "in", "out", 1000.0))
    with_capacitor.add_component(Resistor("R2", "out", "0", 1000.0))
    with_capacitor.add_component(Capacitor("C1", "out", "0", 1e-6))

    baseline = MNASolver(without_capacitor).solve()
    solution = MNASolver(with_capacitor).solve()

    assert solution.node_voltages["out"] == pytest.approx(baseline.node_voltages["out"])


def test_solve_inductor_acts_as_a_wire_at_dc():
    """V1 (5V) feeds an inductor in series with R1: at DC the inductor
    should drop no voltage at all (out == in), and the inductor's own
    branch current should equal the current flowing through R1.
    """
    circuit = Circuit(title="RL at DC")
    circuit.add_component(VoltageSource("V1", "in", "0", 5.0))
    circuit.add_component(Inductor("L1", "in", "out", 1e-3))
    circuit.add_component(Resistor("R1", "out", "0", 1000.0))

    solution = MNASolver(circuit).solve()

    assert solution.node_voltages["in"] == pytest.approx(5.0)
    assert solution.node_voltages["out"] == pytest.approx(5.0)
    assert solution.inductor_currents["L1"] == pytest.approx(0.005)
    assert solution.source_currents["V1"] == pytest.approx(-0.005)


def test_solve_raises_linalg_error_for_ungrounded_circuit():
    """A resistor floating between two nodes with no ground reference
    at all has infinitely many equally-valid solutions (only the
    voltage *difference* is determined), which shows up as a singular
    system matrix -- numpy.linalg.solve reports that as LinAlgError
    rather than silently returning a wrong answer.
    """
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "a", "b", 1000.0))

    with pytest.raises(np.linalg.LinAlgError):
        MNASolver(circuit).solve()
