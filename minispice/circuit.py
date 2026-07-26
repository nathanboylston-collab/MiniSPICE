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

    def list_components(self) -> List[Component]:
        """Return all components in the circuit, in insertion order."""
        return list(self.components)

    def list_nodes(self) -> List[str]:
        """Return all node names, ground first, in discovery order."""
        return list(self._nodes.keys())

    def components_at(self, node: str) -> List[Component]:
        """Return every component with a terminal connected to ``node``."""
        return [component for component in self.components if node in component.nodes]

    def describe(self) -> str:
        """Build (and print) a human-readable summary of the circuit.

        The summary lists every component with its type, terminals, and
        value, then every node with the components attached to it. It is
        printed for interactive/CLI use and also returned as a string so
        it can be inspected or logged without re-parsing stdout.
        """
        lines: List[str] = []

        title = self.title or "(untitled)"
        lines.append(f"Circuit: {title}")

        lines.append(f"Components ({len(self.components)}):")
        if self.components:
            for component in self.components:
                kind = type(component).__name__
                node_list = ", ".join(component.nodes)
                lines.append(
                    f"  {component.name:<6} {kind:<14} nodes=({node_list})  value={component.value}"
                )
        else:
            lines.append("  (none)")

        node_names = self.list_nodes()
        lines.append(f"Nodes ({self.num_nodes} + ground):")
        for node in node_names:
            label = f"{node} (ground)" if node == GROUND else node
            attached = ", ".join(c.name for c in self.components_at(node))
            attached = attached or "(unconnected)"
            lines.append(f"  {label}: {attached}")

        summary = "\n".join(lines)
        print(summary)
        return summary

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
