from __future__ import annotations


class Node:
    """
    Represents a node used by the A* pathfinding algorithm.

    A node stores:
    - its position in the grid,
    - the accumulated cost from the start node,
    - the heuristic cost to the target,
    - a reference to its parent node.

    Attributes
    ----------
    position : tuple
        Grid position of the node.
        Format: (row, col).
    g : float
        Cost from the start node to this node.
    h : float
        Heuristic estimated cost from this node
        to the target node.
    parent : Node
        Parent node used to reconstruct the final path.
    f : float
        Total estimated cost of the node.
    """

    def __init__(
        self, pos: tuple[int, int], g: float, h: float, parent: dict | None = None
    ):
        """
        Constructor
        """
        self.position = pos
        self.g = g
        self.h = h
        self.parent = parent

    def __lt__(self, other: Node):
        """
        Compare two nodes using their total cost.

        This method allows nodes to be sorted
        inside a priority queue (`heapq`).
        """

        return self.f < other.f

    @property
    def f(self):
        """
        Total estimated cost of the node.
        """
        return self.g + self.h
