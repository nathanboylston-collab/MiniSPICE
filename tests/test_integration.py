"""End-to-end tests: SPICE text/file -> Circuit -> MNASolver -> solution.

The unit tests elsewhere (test_stamping.py, test_solver.py) build
Circuit objects by hand and stamp/solve them directly. These tests
instead go through the full pipeline a real user exercises: parsing a
netlist and asking the solver for a DC operating point.
"""

from pathlib import Path

import pytest

from minispice.parser import SpiceParser
from minispice.solver import MNASolver

VOLTAGE_DIVIDER = """\
Voltage Divider
V1 in 0 5
R1 in out 1k
R2 out 0 1k
.end
"""


def test_parsed_voltage_divider_solves_to_expected_node_voltages():
    circuit = SpiceParser().parse_text(VOLTAGE_DIVIDER)

    solution = MNASolver(circuit).solve()

    assert solution.node_voltages["in"] == pytest.approx(5.0)
    assert solution.node_voltages["out"] == pytest.approx(2.5)


def test_parsed_voltage_divider_source_current_matches_ohms_law():
    circuit = SpiceParser().parse_text(VOLTAGE_DIVIDER)

    solution = MNASolver(circuit).solve()

    # Current is defined flowing node_pos -> node_neg through the
    # source (see VoltageSource docstring); physically 2.5mA flows the
    # other way (out of the source into R1), hence the minus sign.
    assert solution.source_currents["V1"] == pytest.approx(-0.0025)


def test_example_divider_file_solves_correctly():
    example_path = Path(__file__).resolve().parent.parent / "examples" / "divider.cir"
    circuit = SpiceParser().parse_file(example_path)

    solution = MNASolver(circuit).solve()

    assert solution.node_voltages["in"] == pytest.approx(5.0)
    assert solution.node_voltages["out"] == pytest.approx(2.5)


def test_asymmetric_divider_matches_voltage_divider_formula():
    """A non-1:1 divider (2k/1k) to confirm the result generalizes,
    not just for the special case where R1 == R2.
    """
    text = "Divider\nV1 in 0 9\nR1 in out 2k\nR2 out 0 1k\n.end\n"
    circuit = SpiceParser().parse_text(text)

    solution = MNASolver(circuit).solve()

    expected_out = 9.0 * 1000.0 / (2000.0 + 1000.0)
    assert solution.node_voltages["out"] == pytest.approx(expected_out)
