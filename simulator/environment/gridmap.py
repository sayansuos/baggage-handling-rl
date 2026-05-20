import numpy as np
from simulator.utils.config import EnvConfig, AgentConfig
from simulator.entities.static_entity import StaticEntity
from simulator.entities.moving_entity import MovingEntity
from simulator.entities.agent import Agent


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
    _grid : np.ndarray
        Internal occupancy grid representation.
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
        Builder
        """

        self.env_config = env_config
        self.agent_config = agent_config
        self._rows, self._columns = env_config.height, env_config.width

        self.static_obstacles = static_obstacles
        self.moving_obstacles = moving_obstacles
        self.agents = agents

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
        Return the current occupancy grid including
        moving obstacles and agents.
        """

        grid = self._grid.copy()

        for entity in self.moving_obstacles + self.agents:

            r, c = self.world_to_grid(entity.current_position)

            pad = int(round(entity.radius))
            r1 = max(0, r - pad)
            r2 = min(self._rows - 1, r + pad)
            c1 = max(0, c - pad)
            c2 = min(self._columns - 1, c + pad)

            grid[r1 : r2 + 1, c1 : c2 + 1] = 1

        return grid

    def get_local_grid(self, agent: Agent, size: int) -> np.ndarray:
        """
        Return the local occupancy grid perceived by the agent.
        """
        if not size % 2 == 1:
            return ValueError("The size must be impair.")

        grid = self.current_grid.copy()
        half = size // 2

        cx, cy = agent.current_position
        r, c = self.world_to_grid((cx, cy))

        r_min, r_max = r - half, r + half
        c_min, c_max = c - half, c + half

        r_min, c_min = self._safe_cell(r_min, c_min)
        r_max, c_max = self._safe_cell(r_max, c_max)

        return grid[r_min : r_max + 1, c_min : c_max + 1]

    def _safe_cell(self, r: int, c: int):
        """
        Clamp grid coordinates to valid grid boundaries.

        This method ensures that a pair of grid indices stays inside
        the occupancy grid limits by clipping:
        - rows to [0, self._rows - 1]
        - columns to [0, self._columns - 1]
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
