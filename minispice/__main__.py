"""Command-line entry point: ``python -m minispice <netlist-file>``.

Ties the whole pipeline together for interactive use -- parse a
SPICE-like netlist, solve its DC operating point, and print a readable
engineering-notation summary -- while turning the ways that can go
wrong (a missing file, a malformed netlist, an unsolvable circuit, an
unsupported component) into a short, user-facing message instead of a
raw traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from minispice.parser import SpiceParseError, SpiceParser
from minispice.solver import MNASolver, SimulationResult


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI end to end. Returns the process exit code."""
    args = _parse_args(argv)
    path = Path(args.netlist)

    print(f"Parsing netlist: {path}")
    try:
        circuit = SpiceParser().parse_file(path)
    except FileNotFoundError:
        print(f"Error: netlist file not found: {path}", file=sys.stderr)
        return 1
    except SpiceParseError as exc:
        print(f"Error parsing netlist: {exc}", file=sys.stderr)
        return 1

    print(
        f"Parsed circuit {circuit.title!r}: "
        f"{len(circuit.components)} component(s), {circuit.num_nodes} node(s)"
    )

    print("Solving DC operating point...")
    try:
        solution = MNASolver(circuit).solve()
    except NotImplementedError:
        print(
            "Error: this circuit uses a component type that isn't "
            "supported yet for DC analysis (only resistors, voltage "
            "sources, and current sources can currently be simulated).",
            file=sys.stderr,
        )
        return 1
    except np.linalg.LinAlgError:
        print(
            'Error: could not solve this circuit -- check for a floating '
            'node or a missing ground (node "0") reference.',
            file=sys.stderr,
        )
        return 1

    print()
    SimulationResult(circuit=circuit, solution=solution).describe()
    return 0


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m minispice",
        description="Parse a SPICE-like netlist and solve its DC operating point.",
    )
    parser.add_argument("netlist", help="path to a SPICE-like netlist file (e.g. divider.cir)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
