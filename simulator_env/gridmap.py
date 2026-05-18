import numpy as np
from entities.agent import Agent


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

    @property
    def current_grid(self) -> np.ndarray:
        return self._update_grid()

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

            r_min, c_min = self.world_to_grid((x_min, y_min))
            r_max, c_max = self.world_to_grid((x_max, y_max))

            r_min = max(0, r_min)
            c_min = max(0, c_min)
            r_max = min(rows - 1, r_max)
            c_max = min(cols - 1, c_max)

            grid[r_min : r_max + 1, c_min : c_max + 1] = 1

        return grid

    def _update_grid(self) -> np.ndarray:
        """
        Return the current occupancy grid including
        moving obstacles and agents.
        """

        grid = self._grid.copy()
        for entity in self.env.moving_obstacles + self.env.agents:
            r, c = self.world_to_grid(entity.current_position)
            pad = int(round(entity.radius))
            grid[r - pad : r + pad + 1, c - pad : c + pad + 1] = 1

        return grid

    def get_local_grid(self, agent: Agent) -> np.ndarray:
        """
        Return the local occupancy grid perceived by the agent.
        """
        size = Agent.LENGTH_VIEW
        local = np.ones((size, size), dtype=self.current_grid.dtype)

        x_min, y_min, x_max, y_max = agent.get_vision_field(
            0,
            self.env.env_width,
            0,
            self.env.env_height,
        )
        r_min, c_min = self.world_to_grid((x_min, y_min))
        r_max, c_max = self.world_to_grid((x_max, y_max))

        return self.current_grid[r_min : r_max + 1, c_min : c_max + 1]

    def world_to_grid(self, pos: tuple[float, float]) -> tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        """
        x, y = pos
        col = int(round(x) / self.resolution)
        row = int(round(y) / self.resolution)
        return row, col

    def grid_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        """
        Convert grid coordinates back to world coordinates.
        """
        row, col = cell
        x = (col) * self.resolution
        y = (row) * self.resolution
        return x, y
