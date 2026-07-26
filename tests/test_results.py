import pytest

from minispice.circuit import Circuit
from minispice.components import Resistor, VoltageSource
from minispice.solver import MNASolver, SimulationResult, format_engineering


# ---------------------------------------------------------------------------
# format_engineering()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        (0.0, "V", "0 V"),
        (5.0, "V", "5 V"),
        (2.5, "V", "2.5 V"),
        (0.0025, "V", "2.5 mV"),
        (2500.0, "V", "2.5 kV"),
        (0.001, "A", "1 mA"),
        (0.5, "A", "500 mA"),
        (1.0, "A", "1 A"),
        (2.0, "A", "2 A"),
        (0.00625, "W", "6.25 mW"),
        (6.25, "W", "6.25 W"),
        (-0.0025, "A", "-2.5 mA"),
    ],
)
def test_format_engineering(value, unit, expected):
    assert format_engineering(value, unit) == expected


def test_format_engineering_picks_largest_applicable_prefix():
    # 1,500,000 V should read as 1.5 MV, not 1500 kV or 1500000 V.
    assert format_engineering(1_500_000.0, "V") == "1.5 MV"


# ---------------------------------------------------------------------------
# SimulationResult -- voltage divider
# ---------------------------------------------------------------------------


def _voltage_divider_result() -> SimulationResult:
    circuit = Circuit(title="Voltage Divider")
    circuit.add_component(VoltageSource("V1", "in", "0", 5.0))
    circuit.add_component(Resistor("R1", "in", "out", 1000.0))
    circuit.add_component(Resistor("R2", "out", "0", 1000.0))

    solution = MNASolver(circuit).solve()
    return SimulationResult(circuit=circuit, solution=solution)


def test_node_voltages():
    result = _voltage_divider_result()

    assert result.node_voltages["in"] == pytest.approx(5.0)
    assert result.node_voltages["out"] == pytest.approx(2.5)
    assert result.node_voltages["0"] == pytest.approx(0.0)


def test_resistor_currents():
    result = _voltage_divider_result()

    # Same 2.5mA flows through both resistors in a simple series divider.
    assert result.resistor_currents["R1"] == pytest.approx(0.0025)
    assert result.resistor_currents["R2"] == pytest.approx(0.0025)


def test_resistor_power_dissipation():
    result = _voltage_divider_result()

    assert result.resistor_power["R1"] == pytest.approx(0.00625)
    assert result.resistor_power["R2"] == pytest.approx(0.00625)


def test_describe_uses_engineering_notation_and_includes_everything(capsys):
    result = _voltage_divider_result()

    summary = result.describe()

    assert "Voltage Divider" in summary
    assert "in" in summary and "5 V" in summary
    assert "out" in summary and "2.5 V" in summary
    assert "R1" in summary and "2.5 mA" in summary
    assert "R2" in summary and "6.25 mW" in summary

    printed = capsys.readouterr().out
    assert printed.strip() == summary.strip()


def test_describe_on_circuit_with_no_resistors_omits_those_sections():
    circuit = Circuit(title="Bare Source")
    circuit.add_component(VoltageSource("V1", "a", "0", 5.0))
    solution = MNASolver(circuit).solve()
    result = SimulationResult(circuit=circuit, solution=solution)

    summary = result.describe()

    assert "Resistor Currents" not in summary
    assert "Resistor Power Dissipation" not in summary
    assert result.resistor_currents == {}
    assert result.resistor_power == {}
