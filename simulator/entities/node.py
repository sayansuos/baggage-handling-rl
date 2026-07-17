from __future__ import annotations


class Node:
    """
    Represents a node used by the A* pathfinding algorithm.
    """

    def __init__(
        self, pos: tuple[int, int], g: float, h: float, parent: dict | None = None
    ):
        """
        Constructor
        """
        self.position: tuple[int, int] = pos
        self.g: float = g
        self.h: float = h
        self.parent: Node = parent

    def __lt__(self, other: Node) -> bool:
        """
        Compare two nodes using their total cost.
        """

        return self.f < other.f

    @property
    def f(self) -> float:
        """
        Total estimated cost of the node.
        """

        return self.g + self.h
