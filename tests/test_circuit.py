from minispice.circuit import Circuit
from minispice.components import Resistor, VoltageSource


def test_new_circuit_has_only_ground():
    circuit = Circuit(title="empty")
    assert circuit.nodes == ["0"]
    assert circuit.num_nodes == 0


def test_add_component_registers_nodes():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "n1", "n2", 100.0))

    assert circuit.num_nodes == 2
    assert circuit.node_index("n1") == 0
    assert circuit.node_index("n2") == 1
    assert circuit.node_index("0") == -1


def test_repeated_node_reuses_index():
    circuit = Circuit()
    circuit.add_component(Resistor("R1", "n1", "0", 100.0))
    circuit.add_component(Resistor("R2", "n1", "n2", 100.0))

    assert circuit.node_index("n1") == 0
    assert circuit.node_index("n2") == 1
    assert circuit.num_nodes == 2
    assert len(circuit.components) == 2


def _voltage_divider() -> Circuit:
    circuit = Circuit(title="Voltage Divider")
    circuit.add_component(VoltageSource("V1", "in", "0", 5.0))
    circuit.add_component(Resistor("R1", "in", "out", 1000.0))
    circuit.add_component(Resistor("R2", "out", "0", 1000.0))
    return circuit


def test_list_components_returns_all_components_in_order():
    circuit = _voltage_divider()

    components = circuit.list_components()

    assert [c.name for c in components] == ["V1", "R1", "R2"]
    # Mutating the returned list must not affect the circuit's internals.
    components.clear()
    assert len(circuit.components) == 3


def test_list_nodes_returns_ground_first_then_discovery_order():
    circuit = _voltage_divider()

    assert circuit.list_nodes() == ["0", "in", "out"]


def test_list_nodes_on_empty_circuit_is_just_ground():
    circuit = Circuit()

    assert circuit.list_nodes() == ["0"]


def test_components_at_returns_only_attached_components():
    circuit = _voltage_divider()

    at_ground = circuit.components_at("0")
    at_in = circuit.components_at("in")
    at_out = circuit.components_at("out")

    assert [c.name for c in at_ground] == ["V1", "R2"]
    assert [c.name for c in at_in] == ["V1", "R1"]
    assert [c.name for c in at_out] == ["R1", "R2"]


def test_components_at_unknown_node_returns_empty_list():
    circuit = _voltage_divider()

    assert circuit.components_at("nonexistent") == []


def test_describe_includes_every_component_and_node(capsys):
    circuit = _voltage_divider()

    summary = circuit.describe()

    for name in ("V1", "R1", "R2"):
        assert name in summary
    for node in ("0", "in", "out"):
        assert node in summary
    assert "Voltage Divider" in summary
    assert "(ground)" in summary

    # describe() also prints the same summary for interactive/CLI use.
    printed = capsys.readouterr().out
    assert printed.strip() == summary.strip()


def test_describe_on_empty_circuit_reports_no_components(capsys):
    circuit = Circuit(title="Empty")

    summary = circuit.describe()

    assert "(none)" in summary
    assert "0 (ground)" in summary
    capsys.readouterr()
