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

This project is in early architectural stages: the component models and
parser are functional, while the MNA solver currently only defines its
interface (matrix stamping and solving are not implemented yet).

## Project layout

```
minispice/
    components/     Component class hierarchy (Resistor, Capacitor, ...)
    parser/          SPICE-like netlist parser
    circuit.py       In-memory circuit/netlist representation
    solver/          Modified Nodal Analysis solver (placeholder)
tests/               pytest test suite
```

## Example netlist

```
Voltage Divider
V1 in 0 5
R1 in out 1k
R2 out 0 1k
.end
```

```python
from minispice.parser import SpiceParser

circuit = SpiceParser().parse_file("divider.cir")
print(circuit)
```

## Development

Install in editable mode with dev dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

- [ ] Implement MNA matrix stamping for each component type
- [ ] DC operating point solve
- [ ] Transient analysis
- [ ] AC small-signal analysis
- [ ] Nonlinear devices (diodes, transistors)
