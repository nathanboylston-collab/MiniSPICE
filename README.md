# MiniSPICE

A minimal, educational SPICE-like circuit simulator written in Python.

## Project goals

- Provide clean, object-oriented models of basic electrical components
  (resistors, capacitors, inductors, independent sources).
- Parse simple SPICE-style netlists describing a circuit's topology and
  element values.
- Solve circuits using Modified Nodal Analysis (MNA), the standard
  approach used by production SPICE simulators.
- Stay small and readable enough to serve as a reference for how circuit
  simulators work, rather than a full-featured replacement for ngspice.

The parser and DC solver are functional for circuits built from
resistors, capacitors, inductors, independent voltage sources, and
independent current sources. Since MiniSPICE only solves the DC
operating point (no transient/AC analysis yet), capacitors and
inductors are stamped as their DC limiting cases -- an open circuit and
a plain wire, respectively -- rather than using their actual
capacitance/inductance value; those values only matter once
transient/AC analysis exists.

## Project layout

```
minispice/
    components/     Component class hierarchy (Resistor, Capacitor, ...)
    parser/          SPICE-like netlist parser
    circuit.py       In-memory circuit/netlist representation
    solver/
        mna.py       MNASystem/MNASolution/MNASolver (matrix stamping + DC solve)
        results.py   SimulationResult (engineering-notation formatting)
    __main__.py      `python -m minispice <file>` CLI
    web/             Optional browser UI (`python -m minispice.web`)
tests/               pytest test suite
```

## Command-line usage

Solve a netlist directly from the command line:

```bash
python -m minispice examples/divider.cir
```

This parses the file, solves its DC operating point, and prints
progress messages followed by a readable summary:

```
Parsing netlist: examples/divider.cir
Parsed circuit 'Voltage Divider': 3 component(s), 2 node(s)
Solving DC operating point...

Simulation Results: Voltage Divider

Node Voltages:
  0        0 V
  in       5 V
  out      2.5 V

Resistor Currents:
  R1       2.5 mA
  R2       2.5 mA

Resistor Power Dissipation:
  R1       6.25 mW
  R2       6.25 mW
```

Problems are reported as a short message on stderr (exit code 1)
instead of a raw traceback: a missing file, a malformed netlist, a
circuit using a component whose stamp isn't implemented (none of the
current component types, but reserved for future devices like diodes),
or a circuit that can't be solved (e.g. a floating node with no path to
ground).

## Browser UI

An optional local web UI wraps the same parser/solver pipeline with a
netlist editor, an auto-laid-out circuit diagram, and live results:

```bash
pip install -e ".[web]"
python -m minispice.web
```

Then open <http://127.0.0.1:5000>. Edit the netlist and click
**Simulate** (or press Ctrl/Cmd+Enter) to re-run it; the diagram and
results update in place, and parse/solve errors show inline instead of
a stack trace. The dev server runs with auto-reload enabled, so as the
simulator itself keeps changing, the next "Simulate" click picks up the
latest code automatically -- no manual restart needed. It's a local
development tool (binds to localhost only), not meant to be deployed.

## Example netlist

```
Voltage Divider
V1 in 0 5
R1 in out 1k
R2 out 0 1k
.end
```

Using the parser and solver directly from Python:

```python
from minispice.parser import SpiceParser
from minispice.solver import MNASolver, SimulationResult

circuit = SpiceParser().parse_file("divider.cir")
solution = MNASolver(circuit).solve()

print(solution.node_voltages)       # {'0': 0.0, 'in': 5.0, 'out': 2.5}
print(solution.source_currents)     # {'V1': -0.0025}

SimulationResult(circuit=circuit, solution=solution).describe()
```

## Development

Install in editable mode with dev dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Implement MNA matrix stamping for resistors, capacitors,
      inductors, independent voltage sources, and independent current
      sources (capacitors/inductors stamped as their DC limiting cases)
- [x] DC operating point solve
- [x] Command-line interface
- [x] Browser UI (netlist editor + circuit diagram + live results)
- [ ] Transient analysis
- [ ] AC small-signal analysis
- [ ] Nonlinear devices (diodes, transistors)
