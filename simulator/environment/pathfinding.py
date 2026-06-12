import heapq

import numpy as np

from simulator.entities import agent, moving_entity, node
from simulator.environment.gridmap import GridMap
from simulator.geometry import get_relative_distance


def compute_astar_paths(
    grid_map: GridMap,
    radius_max: float,
    moving_obstacles: list[moving_entity.MovingEntity],
    agents: list[agent.Agent],
):
    """
    Return the paths found by A* algorithm for each moving obstacle.
    """

    pathfinder = AStar(grid_map.grid, int(radius_max // 2 + 1))

    for entity in list(moving_obstacles + agents):
        if not entity.target_positions or not entity.start_position:
            continue
        start = entity.start_position
        for goal in entity.target_positions:
            start_grid = _to_grid(grid_map, start)
            goal_grid = _to_grid(grid_map, goal)
            path = pathfinder.find_path(start_grid, goal_grid)
            if path:
                path = [_from_grid(grid_map, pos) for pos in path]
                entity.path.extend(path)
                start = goal
            else:
                path = [entity.current_position]


def _from_grid(grid_map: GridMap, pos: tuple[int, int]) -> tuple[float, float]:
    """
    Convert grid coordinates back to world coordinates.
    """
    return grid_map.grid_to_world(pos)


def _to_grid(grid_map: GridMap, pos: tuple[float, float]) -> tuple[int, int]:
    """
    Convert world coordinates to grid coordinates.
    """
    return grid_map.world_to_grid(pos)


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

    def __init__(self, grid: np.ndarray, margin: int | None = None):
        """
        Constructor
        """
        self.grid = grid
        if margin:
            self.inflate_obstacles(margin)

    def inflate_obstacles(self, margin: int):
        """
        Inflate all obstacle cells in the occupancy grid.

        Neighboring cells around each obstacle are marked as occupied
        in order to create a safety margin for path planning.
        """

        inflated = self.grid.copy()
        rows, cols = inflated.shape
        obstacle_cells = np.argwhere(self.grid == 1)
        for r, c in obstacle_cells:
            for dr in range(-margin, margin + 1):
                for dc in range(-margin, margin + 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        inflated[nr, nc] = 1
        self.grid = inflated

    def get_valid_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
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

    def reconstruct_path(self, target: node.Node) -> list[tuple[int, int]]:
        """
        Reconstruct path from target node.
        """

        path = []
        current = target
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]

    def find_path(
        self, start_pos: tuple[int, int], target_pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Find the shortest path between two positions using A*.
        """

        start_node = node.Node(
            start_pos, 0, get_relative_distance(start_pos, target_pos)
        )
        open_list = []
        heapq.heappush(open_list, start_node)
        open_dict = {start_pos: start_node}
        closed_set = set()

        while open_list:
            current_node = heapq.heappop(open_list)
            current_pos = current_node.position

            if get_relative_distance(current_pos, target_pos) < 1e-6:
                return self.reconstruct_path(current_node)

            closed_set.add(current_pos)

            for neighbor_pos in self.get_valid_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue
                new_g = current_node.g + get_relative_distance(
                    current_pos, neighbor_pos
                )
                if neighbor_pos not in open_dict:
                    neighbor_node = node.Node(
                        neighbor_pos,
                        new_g,
                        get_relative_distance(neighbor_pos, target_pos),
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
