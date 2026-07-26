"""Base classes shared by all circuit components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from minispice.solver.mna import MNASystem


class Component(ABC):
    """Base class for every element that can appear in a netlist.

    Subclasses represent a specific device (resistor, capacitor, source,
    ...) and are responsible for contributing their own entries to the
    Modified Nodal Analysis system via :meth:`stamp`.
    """

    def __init__(self, name: str, nodes: Tuple[str, ...], value: float) -> None:
        self.name = name
        self.nodes = nodes
        self.value = value

    @abstractmethod
    def stamp(self, system: "MNASystem") -> None:
        """Add this component's contribution to the MNA system.

        Left unimplemented for now; the solver module currently only
        defines the shape of ``MNASystem`` and does not perform stamping.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, nodes={self.nodes}, value={self.value})"


class TwoTerminalComponent(Component):
    """Convenience base for components with exactly two terminals."""

    def __init__(self, name: str, node_pos: str, node_neg: str, value: float) -> None:
        super().__init__(name, (node_pos, node_neg), value)

    @property
    def node_pos(self) -> str:
        return self.nodes[0]

    @property
    def node_neg(self) -> str:
        return self.nodes[1]
