"""Flask application: interactive browser UI for MiniSPICE.

Wraps the existing parser/solver/results pipeline (unchanged) in a
small JSON API -- ``POST /simulate`` -- plus a single HTML page that
renders the circuit as an auto-laid-out diagram and shows the DC solve
results. This is a local development/testing tool, not a production
service: see ``minispice/web/__main__.py`` for how it's launched.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, render_template, request

from minispice.components import (
    Capacitor,
    CurrentSource,
    Inductor,
    Resistor,
    VoltageSource,
)
from minispice.parser import SpiceParseError, SpiceParser
from minispice.solver import MNASolution, MNASolver, MNASystem, SimulationResult, format_engineering

DEFAULT_NETLIST = """\
Voltage Divider
V1 in 0 5
R1 in out 1k
R2 out 0 1k
.end
"""

# The unit each component's raw value should be displayed in -- same
# base units the parser/solver already work in (see minispice.parser.units).
_COMPONENT_UNITS = {
    Resistor: "Ω",  # Ohm
    Capacitor: "F",
    Inductor: "H",
    VoltageSource: "V",
    CurrentSource: "A",
}


def create_app() -> Flask:
    """Build the Flask app. Kept as a factory so tests can create fresh
    instances instead of sharing global state.
    """
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html", default_netlist=DEFAULT_NETLIST)

    @app.post("/simulate")
    def simulate():
        payload = request.get_json(silent=True) or {}
        netlist_text = payload.get("netlist", "")

        try:
            circuit = SpiceParser().parse_text(netlist_text)
        except SpiceParseError as exc:
            return jsonify(ok=False, error=f"Error parsing netlist: {exc}")

        try:
            system = MNASolver(circuit).build_system()
        except NotImplementedError:
            return jsonify(
                ok=False,
                error=(
                    "This circuit uses a component type that isn't supported "
                    "yet for DC analysis."
                ),
            )

        try:
            x = np.linalg.solve(system.A, system.z)
        except np.linalg.LinAlgError:
            return jsonify(
                ok=False,
                error=(
                    'Could not solve this circuit -- check for a floating '
                    'node or a missing ground (node "0") reference.'
                ),
            )

        solution = MNASolution.from_system(system, x)
        result = SimulationResult(circuit=circuit, solution=solution)
        return jsonify(
            ok=True,
            circuit=_circuit_json(circuit),
            results=_results_json(result),
            matrix=_matrix_json(system, x),
        )

    return app


def _circuit_json(circuit) -> Dict[str, Any]:
    """The circuit's topology, in a form the frontend can lay out and draw."""
    return {
        "title": circuit.title,
        "nodes": circuit.list_nodes(),
        "components": [
            {
                "name": component.name,
                "type": type(component).__name__,
                "node_pos": component.node_pos,
                "node_neg": component.node_neg,
                "value": component.value,
                "value_display": format_engineering(
                    component.value, _COMPONENT_UNITS.get(type(component), "")
                ),
            }
            for component in circuit.components
        ],
    }


def _named_values_json(values: Dict[str, float], unit: str) -> Dict[str, Dict[str, Any]]:
    """Wrap a {name: raw_float} dict with an engineering-notation display string."""
    return {
        name: {"value": value, "display": format_engineering(value, unit)}
        for name, value in values.items()
    }


def _results_json(result: SimulationResult) -> Dict[str, Any]:
    return {
        "node_voltages": _named_values_json(result.node_voltages, "V"),
        "resistor_currents": _named_values_json(result.resistor_currents, "A"),
        "resistor_power": _named_values_json(result.resistor_power, "W"),
        "source_currents": _named_values_json(result.solution.source_currents, "A"),
        "inductor_currents": _named_values_json(result.solution.inductor_currents, "A"),
    }


def _matrix_json(system: MNASystem, x: np.ndarray) -> Dict[str, Any]:
    """The raw MNA linear system (A, z) plus its solution (x), labeled
    by unknown name so a matrix-viewer UI can show a human-readable
    "V(in)" / "I(V1)" row/column instead of a bare index.
    """
    return {
        "labels": system.unknown_labels(),
        "A": system.A.tolist(),
        "z": system.z.tolist(),
        "x": x.tolist(),
    }
