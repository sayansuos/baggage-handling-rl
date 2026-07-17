import warnings

import numpy as np

from configs.config import AgentConfig, EnvConfig
from simulator.entities.agent import Agent
from simulator.entities.moving_entity import MovingEntity
from simulator.entities.static_entity import StaticEntity


class GridMap:
    """
    Manage the occupancy grid and the local observations of the agents.
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

        self.env_config: EnvConfig = env_config
        self.agent_config: AgentConfig = agent_config

        self.static_obstacles: list[StaticEntity] = static_obstacles
        self.moving_obstacles: list[MovingEntity] = moving_obstacles
        self.agents: list[Agent] = agents

        self._rows: int = env_config.height
        self._columns: int = env_config.width
        self._grid: np.ndarray = self._build_grid()

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

        # Initialize an empty grid
        grid = np.zeros((self._rows, self._columns), dtype=np.uint8)

        for obstacle in self.static_obstacles:
            # Initialize an empty occupancy grid
            x_min, y_min, x_max, y_max = obstacle.bounds
            x_min = np.clip(x_min, 0, self._columns - 1)
            x_max = np.clip(x_max, 0, self._columns - 1)
            y_min = np.clip(y_min, 0, self._rows - 1)

            # Convert to grid coordinates
            r1, c1 = self.world_to_grid(pos=(x_min, y_min))
            r2, c2 = self.world_to_grid(pos=(x_max, y_max))

            # Sort and clamp the grid bounds
            r_min, r_max = sorted([r1, r2])
            c_min, c_max = sorted([c1, c2])
            r_min, c_min = self._safe_cell(r_min, c_min)
            r_max, c_max = self._safe_cell(r_max, c_max)

            # Mark the cells as occupied
            if r_min <= r_max and c_min <= c_max:
                grid[r_min : r_max + 1, c_min : c_max + 1] = 1

        return grid

    def _update_grid(self) -> np.ndarray:
        """
        Update the occupancy grid with dynamic entities.
        """

        # Copy the static grid
        grid = self._grid.copy()

        for entity in self.moving_obstacles + self.agents:
            # Ignore entities without a current position
            if not entity.current_position:
                continue

            # Convert to grid coordinates.
            r, c = self.world_to_grid(pos=entity.current_position)

            # Compute the occupied area from the entity radius
            pad = int(round(entity.radius))
            r1 = max(0, r - pad)
            r2 = min(self._rows - 1, r + pad)
            c1 = max(0, c - pad)
            c2 = min(self._columns - 1, c + pad)

            # Mark the cells as occupied
            grid[r1 : r2 + 1, c1 : c2 + 1] = 1

        return grid

    def get_local_grid(
        self, agent: Agent, current_grid: np.ndarray, size: int
    ) -> np.ndarray:
        """
        Return the local occupancy grid perceived by the agent.
        """

        # Initialize a full grid
        local = np.ones((size, size), dtype=np.float32)

        # Ensure that the local grid has an odd size.
        if not size % 2 == 1:
            warnings.warn(
                f"Local grid size must be odd. Using {size - 1} instead.",
                UserWarning,
            )
            size -= 1

        half = size // 2

        # Return the grid if no current position
        if agent.current_position is None:
            return local

        # Convert to grid coordinates
        cx, cy = agent.current_position
        center_r, center_c = self.world_to_grid(pos=(cx, cy))

        #  Extract the cells centered around the agent
        for local_r in range(size):
            for local_c in range(size):
                grid_r = center_r + local_r - half
                grid_c = center_c + local_c - half

                # Copy only cells located inside the global grid
                if 0 <= grid_r < self._rows and 0 <= grid_c < self._columns:
                    local[local_r, local_c] = current_grid[grid_r, grid_c]

        return local

    def _safe_cell(self, r: int, c: int) -> tuple[int, int]:
        """
        Clamp grid coordinates to valid grid boundaries.
        """

        # Clamp the row and column to the grid boundaries
        r = np.clip(r, 0, self._rows - 1)
        c = np.clip(c, 0, self._columns - 1)

        return int(r), int(c)

    def world_to_grid(self, pos: tuple[float, float]) -> tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        """

        x, y = pos

        # Convert cartesian coordinates to row column
        col = int(round(x))
        row = self._rows - 1 - int(round(y))

        return row, col

    def grid_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        """
        Convert grid coordinates back to world coordinates.
        """

        row, col = cell

        # Convert row colum coordinates to cartesian
        x = col
        y = self._rows - 1 - row

        return x, y
