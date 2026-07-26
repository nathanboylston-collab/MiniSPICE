"""In-memory representation of a parsed circuit (netlist)."""

from __future__ import annotations

from typing import Dict, List

from minispice.components.base import Component

GROUND = "0"


class Circuit:
    """A collection of components plus the node set they connect."""

    def __init__(self, title: str = "") -> None:
        self.title = title
        self.components: List[Component] = []
        self._nodes: Dict[str, int] = {}
        self._register_node(GROUND)

    @property
    def nodes(self) -> List[str]:
        """All node names, ground first, in discovery order."""
        return list(self._nodes.keys())

    @property
    def num_nodes(self) -> int:
        """Number of nodes excluding ground."""
        return len(self._nodes) - 1

    def add_component(self, component: Component) -> None:
        for node in component.nodes:
            self._register_node(node)
        self.components.append(component)

    def node_index(self, node: str) -> int:
        """Index of a node within the MNA unknown vector.

        Ground is index -1 (i.e. not part of the unknown vector); all
        other nodes are numbered from 0 in order of first appearance.
        """
        if node == GROUND:
            return -1
        return self._nodes[node] - 1

    def _register_node(self, node: str) -> None:
        if node not in self._nodes:
            self._nodes[node] = len(self._nodes)

    def __repr__(self) -> str:
        return f"Circuit(title={self.title!r}, components={len(self.components)}, nodes={self.num_nodes})"
