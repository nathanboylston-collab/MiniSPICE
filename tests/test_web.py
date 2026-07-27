"""Tests for the optional browser UI's Flask routes (minispice.web).

These use Flask's test client, so no real server/network is involved.
Requires the 'web' extra (flask) to be installed.
"""

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from minispice.web import create_app
from minispice.web.app import DEFAULT_NETLIST


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_page_loads_and_embeds_default_netlist(client):
    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "MiniSPICE" in body
    assert "Voltage Divider" in body
    assert 'id="netlist"' in body


def test_simulate_voltage_divider_returns_expected_circuit_and_results(client):
    response = client.post("/simulate", json={"netlist": DEFAULT_NETLIST})

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True

    circuit = data["circuit"]
    assert circuit["title"] == "Voltage Divider"
    assert circuit["nodes"] == ["0", "in", "out"]
    names = {c["name"] for c in circuit["components"]}
    assert names == {"V1", "R1", "R2"}

    results = data["results"]
    assert results["node_voltages"]["in"]["value"] == pytest.approx(5.0)
    assert results["node_voltages"]["out"]["value"] == pytest.approx(2.5)
    assert results["node_voltages"]["out"]["display"] == "2.5 V"
    assert results["resistor_currents"]["R1"]["display"] == "2.5 mA"
    assert results["resistor_power"]["R1"]["display"] == "6.25 mW"
    assert results["source_currents"]["V1"]["value"] == pytest.approx(-0.0025)
    assert results["inductor_currents"] == {}

    matrix = data["matrix"]
    assert matrix["labels"] == ["V(in)", "V(out)", "I(V1)"]
    g = 1.0 / 1000.0
    expected_A = np.array([[g, -g, 1.0], [-g, 2 * g, 0.0], [1.0, 0.0, 0.0]])
    assert np.array(matrix["A"]) == pytest.approx(expected_A)
    assert matrix["z"] == pytest.approx([0.0, 0.0, 5.0])
    assert matrix["x"] == pytest.approx([5.0, 2.5, -0.0025])


def test_simulate_matrix_omitted_when_circuit_is_unsolvable(client):
    """No successful solve means no matrix -- the 'x' half of A x = z
    was never computed, so there's nothing consistent to show.
    """
    response = client.post("/simulate", json={"netlist": "Floating\nR1 a b 1k\n"})

    data = response.get_json()
    assert data["ok"] is False
    assert "matrix" not in data


def test_simulate_reports_parse_errors_without_500(client):
    response = client.post("/simulate", json={"netlist": "Bad\nR1 a b\n"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is False
    assert "expects exactly 3 fields" in data["error"]


def test_simulate_reports_singular_system_without_500(client):
    response = client.post("/simulate", json={"netlist": "Floating\nR1 a b 1k\n"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is False
    assert "floating node" in data["error"]


def test_simulate_with_missing_netlist_key_treats_it_as_empty(client):
    """An absent 'netlist' key falls back to "", which parses to a
    valid (trivial, ground-only) circuit rather than an error -- same
    behavior as SpiceParser().parse_text("") on its own.
    """
    response = client.post("/simulate", json={})

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert data["circuit"]["nodes"] == ["0"]
    assert data["circuit"]["components"] == []


def test_simulate_includes_capacitor_and_inductor_in_circuit_json(client):
    netlist = "RLC\nV1 in 0 5\nL1 in mid 1m\nR1 mid out 1k\nC1 out 0 1u\n.end\n"

    response = client.post("/simulate", json={"netlist": netlist})

    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    components = {c["name"]: c for c in data["circuit"]["components"]}
    assert components["L1"]["type"] == "Inductor"
    assert components["C1"]["type"] == "Capacitor"
    assert components["C1"]["value_display"] == "1 uF"
