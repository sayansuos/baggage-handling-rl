import numpy as np


class GridMap:
    """
    Occupancy grid representation of a continuous environment.

    This class converts a continuous 2D environment into a discrete grid
    suitable for path planning algorithms such as A*.

    The grid encodes:
    - 0 → free space
    - 1 → occupied space (static obstacles)

    Attributes
    ----------
    env : Environment
        The simulation environment containing obstacles and dimensions.
    resolution : int
        Size of one grid cell in world units.
        Higher resolution → finer grid but higher computation cost.
    _grid : np.ndarray
        Internal occupancy grid representation.
    """

    def __init__(self, env: Environment, resolution: int = 1):
        """
        Builder
        """
        self.env = env
        self.resolution = resolution
        self._grid = self._build_grid()

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def shape(self) -> tuple:
        return self._grid.shape

    def _build_grid(self) -> np.ndarray:
        """
        Build the occupancy grid from the environment.

        Static obstacles are projected into grid cells by converting
        their world-space bounding boxes into grid indices.
        """

        rows = int(self.env.env_height / self.resolution)
        cols = int(self.env.env_width / self.resolution)

        grid = np.zeros((rows, cols), dtype=np.uint8)

        for obstacle in self.env.static_obstacles:

            x_min, y_min, x_max, y_max = obstacle.bounds

            gx_min = max(0, int(x_min / self.resolution))
            gy_min = max(0, int(y_min / self.resolution))
            gx_max = min(cols - 1, int(x_max / self.resolution))
            gy_max = min(rows - 1, int(y_max / self.resolution))

            grid[gy_min : gy_max + 1, gx_min : gx_max + 1] = 1

        return grid

    def world_to_grid(self, pos: tuple[float, float]) -> tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        """
        x, y = pos
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return row, col

    def grid_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        """
        Convert grid coordinates back to world coordinates.
        """
        row, col = cell
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y
