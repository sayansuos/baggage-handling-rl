import numpy as np
import heapq
from entities.node import Node


def get_distance(pos1: tuple, pos2: tuple) -> float:
    """
    Compute Euclidean distance between two positions.
    """

    return np.linalg.norm(np.array(pos1) - np.array(pos2))


class AStar:
    """
    A* path planning algorithm on a 2D occupancy grid.

    The algorithm finds the shortest path between a start and a goal
    while avoiding obstacles defined in the grid.

    Grid format:
    - 0 → free cell
    - 1 → obstacle

    Attributes
    ----------
    grid : np.ndarray
        Occupancy grid used for navigation.
    """

    def __init__(self, grid: np.ndarray, radius: int = None):
        """
        Builder
        """
        self.grid = grid
        if radius:
            self.inflate_obstacles(radius)

    def inflate_obstacles(self, radius: int):
        """ """
        inflated = self.grid.copy()
        rows, cols = inflated.shape
        obstacle_cells = np.argwhere(self.grid == 1)
        for r, c in obstacle_cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        inflated[nr, nc] = 1
        self.grid = inflated

    def get_valid_neighbors(self, pos: tuple) -> list[tuple]:
        """
        Return valid neighboring cells.

        A valid neighbor:
        - is inside the grid
        - is not an obstacle (grid value = 0)
        """

        x, y = pos
        rows, cols = self.grid.shape
        possible_moves = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
        ]
        valid_moves = []
        for r, c in possible_moves:
            if 0 <= r < rows and 0 <= c < cols and self.grid[r][c] == 0:
                valid_moves.append((r, c))
        return valid_moves

    def reconstruct_path(self, target: Node) -> list[tuple]:
        """
        Reconstruct path from target node.
        """

        path = []
        current = target
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]

    def find_path(self, start_pos: tuple, target_pos: tuple):
        """
        Find the shortest path between two positions using A*.
        """

        start_node = Node(start_pos, 0, get_distance(start_pos, target_pos))
        open_list = []
        heapq.heappush(open_list, start_node)
        open_dict = {start_pos: start_node}
        closed_set = set()

        while open_list:
            current_node = heapq.heappop(open_list)
            current_pos = current_node.position

            if get_distance(current_pos, target_pos) < 1e-6:
                return self.reconstruct_path(current_node)

            closed_set.add(current_pos)

            for neighbor_pos in self.get_valid_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue
                new_g = current_node.g + get_distance(current_pos, neighbor_pos)
                if neighbor_pos not in open_dict:
                    neighbor_node = Node(
                        neighbor_pos,
                        new_g,
                        get_distance(neighbor_pos, target_pos),
                        parent=current_node,
                    )
                    heapq.heappush(open_list, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node
                elif new_g < open_dict[neighbor_pos].g:
                    neighbor_node = open_dict[neighbor_pos]
                    neighbor_node.g = new_g
                    neighbor_node.parent = current_node
                    heapq.heappush(open_list, neighbor_node)
        return []
