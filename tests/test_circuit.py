from minispice.circuit import Circuit
from minispice.components import Resistor


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
