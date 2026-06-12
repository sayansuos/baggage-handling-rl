import warnings

import numpy as np

from configs.config import AgentConfig, EnvConfig
from simulator.entities.agent import Agent
from simulator.entities.moving_entity import MovingEntity
from simulator.entities.static_entity import StaticEntity


class GridMap:
    """
    Occupancy grid representation of the environment.

    This class converts the continuous simulation world into a
    discrete 2D occupancy grid.

    The occupancy grid uses the following convention:
    - 0 → free cell
    - 1 → occupied cell

    Attributes
    ----------
    env_config : EnvConfig
        Environment configuration parameters.
    agent_config : AgentConfig
        Agent configuration parameters.
    static_obstacles : list[StaticEntity]
        List of static obstacles in the environment.
    moving_obstacles : list[MovingEntity]
        List of dynamic obstacles in the environment.
    agents : list[Agent]
        List of agents in the environment.
    _rows : int
        Number of rows in the occupancy grid.
    _columns : int
        Number of columns in the occupancy grid.
    _grid : np.ndarray
        Static occupancy grid containing only static obstacles.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        agent_config: AgentConfig,
        static_obstacles: list[StaticEntity],
        moving_obstacles: list[MovingEntity],
        agents: list[Agent],
    ):
        """
        Constructor
        """

        self.env_config = env_config
        self.agent_config = agent_config

        self.static_obstacles = static_obstacles
        self.moving_obstacles = moving_obstacles
        self.agents = agents

        self._rows, self._columns = env_config.height, env_config.width
        self._grid = self._build_grid()

    @property
    def grid(self) -> np.ndarray:
        """
        Return the static occupancy grid.
        """
        return self._grid

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the occupancy grid.
        """
        return self._grid.shape

    @property
    def current_grid(self) -> np.ndarray:
        """
        Return the current occupancy grid.
        """
        return self._update_grid()

    def _build_grid(self) -> np.ndarray:
        """
        Build the static occupancy grid.
        """

        grid = np.zeros((self._rows, self._columns), dtype=np.uint8)

        for obstacle in self.static_obstacles:

            x_min, y_min, x_max, y_max = obstacle.bounds
            x_min = np.clip(x_min, 0, self._columns - 1)
            x_max = np.clip(x_max, 0, self._columns - 1)
            y_min = np.clip(y_min, 0, self._rows - 1)

            r1, c1 = self.world_to_grid((x_min, y_min))
            r2, c2 = self.world_to_grid((x_max, y_max))

            r_min, r_max = sorted([r1, r2])
            c_min, c_max = sorted([c1, c2])

            r_min, c_min = self._safe_cell(r_min, c_min)
            r_max, c_max = self._safe_cell(r_max, c_max)

            if r_min <= r_max and c_min <= c_max:
                grid[r_min : r_max + 1, c_min : c_max + 1] = 1

        return grid

    def _update_grid(self) -> np.ndarray:
        """
        Update the occupancy grid with dynamic entities.
        """

        grid = self._grid.copy()

        for entity in self.moving_obstacles + self.agents:

            if not entity.current_position:
                continue
            r, c = self.world_to_grid(entity.current_position)

            pad = int(round(entity.radius))
            r1 = max(0, r - pad)
            r2 = min(self._rows - 1, r + pad)
            c1 = max(0, c - pad)
            c2 = min(self._columns - 1, c + pad)

            grid[r1 : r2 + 1, c1 : c2 + 1] = 1

        return grid

    def get_local_grid(
        self, agent: Agent, current_grid: np.ndarray, size: int
    ) -> np.ndarray:
        """
        Return the local occupancy grid perceived by the agent.
        """

        local = np.ones((size, size), dtype=np.float32)

        if not size % 2 == 1:
            warnings.warn(
                f"Local grid size must be odd. Using {size - 1} instead.",
                UserWarning,
            )
            size -= 1

        half = size // 2

        if agent.current_position is None:
            return local

        cx, cy = agent.current_position
        center_r, center_c = self.world_to_grid((cx, cy))

        for local_r in range(size):
            for local_c in range(size):

                grid_r = center_r + local_r - half
                grid_c = center_c + local_c - half

                if 0 <= grid_r < self._rows and 0 <= grid_c < self._columns:
                    local[local_r, local_c] = current_grid[grid_r, grid_c]

        return local

    def _safe_cell(self, r: int, c: int) -> tuple[int, int]:
        """
        Clamp grid coordinates to valid grid boundaries.
        """

        r = np.clip(r, 0, self._rows - 1)
        c = np.clip(c, 0, self._columns - 1)
        return int(r), int(c)

    def world_to_grid(self, pos: tuple[float, float]) -> tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        """

        x, y = pos
        col = int(round(x))
        row = self._rows - 1 - int(round(y))
        return row, col

    def grid_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        """
        Convert grid coordinates back to world coordinates.
        """
        row, col = cell
        x = col
        y = self._rows - 1 - row
        return x, y
