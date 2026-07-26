import pytest

from minispice.circuit import Circuit
from minispice.solver import MNASolver


def test_build_system_not_yet_implemented():
    solver = MNASolver(Circuit())
    with pytest.raises(NotImplementedError):
        solver.build_system()


def test_solve_not_yet_implemented():
    solver = MNASolver(Circuit())
    with pytest.raises(NotImplementedError):
        solver.solve()
