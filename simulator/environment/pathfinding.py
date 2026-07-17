import heapq

import numpy as np

from simulator.entities import moving_entity, node
from simulator.environment.gridmap import GridMap
from simulator.geometry import get_relative_distance


def compute_astar_paths(
    grid_map: GridMap,
    radius_max: float,
    moving_obstacles: list[moving_entity.MovingEntity],
    min_length: int | None = None,
) -> None:
    """
    Compute A* paths for all moving obstacles.
    """

    # Create the pathfinder
    pathfinder = AStar(grid=grid_map.grid, margin=int(radius_max // 2 + 1))

    for entity in list(moving_obstacles):
        if not entity.target_positions or not entity.start_position:
            continue

        # Initialize start and path
        start = entity.start_position
        full_path = []

        # Compute a path between each pair of consecutive targets
        for target in entity.target_positions:
            start_grid = _to_grid(grid_map=grid_map, pos=start)
            target_grid = _to_grid(grid_map=grid_map, pos=target)
            path = pathfinder.find_path(start_pos=start_grid, _pos=target_grid)

            # Convert the grid coordinates to world ones
            if path:
                path = [_from_grid(grid_map=grid_map, cell=pos) for pos in path]
                full_path.extend(path)
                start = target

            # If no path is found, gives the current position
            else:
                path = [entity.current_position]

        # Extend the trajectory with back-and-forth, until chosen length
        if min_length is not None and len(full_path) > 1:
            back_and_forth = full_path + full_path[::-1]
            extended_path = []

            while len(extended_path) < min_length:
                extended_path.extend(back_and_forth)

            full_path = extended_path[:min_length]

        # Assign the trajectory to entity
        entity.path = full_path


def _from_grid(grid_map: GridMap, cell: tuple[int, int]) -> tuple[float, float]:
    """
    Convert grid coordinates to world coordinates.
    """

    return grid_map.grid_to_world(cell=cell)


def _to_grid(grid_map: GridMap, pos: tuple[float, float]) -> tuple[int, int]:
    """
    Convert world coordinates to grid coordinates.
    """

    return grid_map.world_to_grid(pos=pos)


class AStar:
    """
    A* path planner on a 2D occupancy grid.
    """

    def __init__(self, grid: np.ndarray, margin: int | None = None):
        """
        Constructor
        """

        self.grid: np.ndarray = grid
        if margin:
            self.inflate_obstacles(margin)

    def inflate_obstacles(self, margin: int) -> None:
        """
        Inflate the obstacles in the occupancy grid.
        """

        # Copy the grid
        inflated = self.grid.copy()
        rows, cols = inflated.shape

        # Retrieve occupied cells.
        obstacle_cells = np.argwhere(self.grid == 1)

        # Mark all cells located within the safety margin as occupied.
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
        Return the valid neighboring cells of a grid position.
        """

        x, y = pos
        rows, cols = self.grid.shape
        valid_moves = []

        # Generate horizontal, vertical, and diagonal neighbors
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

        # Keep only free cells
        for r, c in possible_moves:
            if 0 <= r < rows and 0 <= c < cols and self.grid[r][c] == 0:
                valid_moves.append((r, c))

        return valid_moves

    def reconstruct_path(self, target: node.Node) -> list[tuple[int, int]]:
        """
        Reconstruct path from  node.
        """

        path = []
        current = target

        # Follow the parent links from the target to the start node
        while current is not None:
            path.append(current.position)
            current = current.parent

        # Reverse the path
        return path[::-1]

    def find_path(
        self, start_pos: tuple[int, int], target_pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Find the shortest path between two positions using A*.
        """

        # Initialize the start node and the A* data structures
        h = get_relative_distance(pos1=start_pos, pos2=target_pos)
        start_node = node.Node(pos=start_pos, g=0, h=h, parent=None)
        open_list = []
        heapq.heappush(open_list, start_node)
        open_dict = {start_pos: start_node}
        closed_set = set()

        while open_list:
            # Select the node with the lowest estimated total cost
            current_node = heapq.heappop(open_list)
            current_pos = current_node.position

            # Stop when the target position is reached.
            if get_relative_distance(pos1=current_pos, pos2=target_pos) < 1e-6:
                return self.reconstruct_path(target=current_node)

            # Mark the current node as explored
            closed_set.add(current_pos)

            for neighbor_pos in self.get_valid_neighbors(pos=current_pos):
                # Ignore previously explored neighbors
                if neighbor_pos in closed_set:
                    continue

                # Compute the cost from the start to the neighboring cell
                new_g = current_node.g + get_relative_distance(
                    pos1=current_pos, pos2=neighbor_pos
                )

                # Add discovered neighbors to the open list
                if neighbor_pos not in open_dict:
                    new_h = get_relative_distance(pos1=neighbor_pos, pos2=target_pos)
                    neighbor_node = node.Node(
                        pos=neighbor_pos,
                        g=new_g,
                        h=new_h,
                        parent=current_node,
                    )
                    heapq.heappush(open_list, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node

                # Update the neighbor when a shorter path is found
                elif new_g < open_dict[neighbor_pos].g:
                    neighbor_node = open_dict[neighbor_pos]
                    neighbor_node.g = new_g
                    neighbor_node.parent = current_node
                    heapq.heappush(open_list, neighbor_node)

        # Return an empty path when the target is unreachable
        return []
