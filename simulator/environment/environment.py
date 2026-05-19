import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from simulator.entities.agent import Agent
from simulator.entities.entity import Entity
from simulator.motion.astar import AStar
from simulator.environment.gridmap import GridMap
from simulator.environment.environment_manager import EnvironmentManager


class Environment(gym.Env):
    """
    Multi-agent reinforcement learning environment.

    The environment contains:
    - autonomous agents
    - static rectangular obstacles
    - moving circular obstacles

    Each observation is decentralized:
    an agent only observes its own local state.

    The environment follows the Gymnasium API.
    """

    ALLOWED_V_MIN, ALLOWED_V_MAX = 0, 5
    ALLOWED_OMEGA_MIN, ALLOWED_OMEGA_MAX = -np.pi / 6, np.pi / 6

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        nb_agents: int,
        nb_static_obstacles: int,
        nb_moving_obstacles: int,
        env_width: int,
        env_height: int,
    ):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.env_width, self.env_height = env_width, env_height
        self.nb_agents = nb_agents

        # ---------------------------------------------------------------
        # Environment Setup
        # ---------------------------------------------------------------

        env_manager = EnvironmentManager(
            self.env_width,
            self.env_height,
            nb_agents,
            nb_static_obstacles,
            nb_moving_obstacles,
        )

        self.static_obstacles = env_manager.generate_static_obstacles()
        self.moving_obstacles = env_manager.generate_moving_obstacles()
        self.agents = env_manager.generate_agents()

        self.grid_map = GridMap(self)
        self._compute_astar_paths()

        # ---------------------------------------------------------------
        # Observation Space
        # ---------------------------------------------------------------

        self.observation_space = spaces.Dict(
            {
                "local_map": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1, Agent.MAP_SIZE, Agent.MAP_SIZE),
                    dtype=np.float64,
                ),
                "goal_relative_position": spaces.Box(
                    low=-max(self.env_width, self.env_height),
                    high=max(self.env_width, self.env_height),
                    shape=(2,),
                    dtype=np.float64,
                ),
                "motion": spaces.Box(
                    low=np.array([Agent.V_MIN, Agent.OMEGA_MIN]),
                    high=np.array([Agent.V_MAX, Agent.OMEGA_MAX]),
                    dtype=np.float64,
                ),
                "orientation": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
            }
        )

        # ---------------------------------------------------------------
        # Action Space
        # ---------------------------------------------------------------

        self.action_space = spaces.Box(
            low=np.array([self.ALLOWED_V_MIN, self.ALLOWED_OMEGA_MIN]),
            high=np.array([self.ALLOWED_V_MAX, self.ALLOWED_OMEGA_MAX]),
            dtype=np.float64,
        )

    def _get_obs(self, agent: Agent, local_size: int = Agent.LENGTH_VIEW) -> dict:
        """
        Compute the decentralized observation of an agent.

        The observation only contains information locally
        available to the agent.
        """

        return {
            "local_map": self._get_local_grid(agent, local_size),
            "goal_relative_position": agent._goal_relative_position,
            "motion": agent._motion,
            "orientation": agent._orientation,
        }

    def reset(self, seed=None, options=None):
        pass

    def step(self, action=None):
        self._update_obstacles()
        obs = None
        reward = None
        done = False
        info = {}
        return obs, reward, done, info

    def render(self):
        """
        Default render method for the global environment.
        """
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        colors = plt.cm.get_cmap("tab10", self.nb_agents)

        for entity in self.static_obstacles:
            entity.render(self.ax)

        for i, entity in enumerate(self.moving_obstacles):
            entity.render(self.ax)

        for i, agent in enumerate(self.agents):
            agent.render(
                self.ax, 0, self.env_width, 0, self.env_height, color=colors(i)
            )
            self.ax.scatter([], [], color=colors(i), label=f"Agent n°{agent.num}")

        self.ax.set_xlim(0, self.env_width)
        self.ax.set_ylim(0, self.env_height)
        self.ax.legend(loc="upper left")

        # plt.pause(0.1)

    # ---------------------------------------------------------------
    # Methods needed for updates
    # ---------------------------------------------------------------

    @property
    def grid(self):
        return self.grid_map.grid

    def _to_grid(self, pos):
        """
        Convert world coordinates to grid coordinates.
        """
        return self.grid_map.world_to_grid(pos)

    def _from_grid(self, pos):
        """
        Convert grid coordinates back to world coordinates.
        """
        return self.grid_map.grid_to_world(pos)

    def _compute_astar_paths(self):
        """
        Return the paths found by A* algorithm for each moving obstacle.
        """

        pathfinder = AStar(self.grid, int(Agent.RADIUS) + 1)

        for entity in self.moving_obstacles:
            # entity.path = []
            if not entity.target_positions:
                continue
            start = entity.start_position
            for goal in entity.target_positions:
                start_grid = self._to_grid(start)
                goal_grid = self._to_grid(goal)
                path = pathfinder.find_path(start_grid, goal_grid)
                path = [self._from_grid(pos) for pos in path]
                entity.path.extend(path)
                start = goal

    def _update_obstacles(self):
        """
        Update the positions of all moving obstacles.

        Each moving obstacle computes its next position according
        to its internal motion model or predefined path.

        The new position is applied only if a valid next position
        is returned.
        """

        for obs in self.moving_obstacles:
            next_pos = obs._step()
            if next_pos:
                self._is_free(obs, next_pos, 0)
                obs.current_position = next_pos

    def _get_local_grid(self, agent: Agent, size: int = Agent.LENGTH_VIEW):
        """
        Return the local occupancy grid perceived by a given agent.

        The local grid is centered around the agent's current position
        and represents nearby occupied and free cells.
        """

        return self.grid_map.get_local_grid(agent, size)

    def _get_local_grids(self, size: int = Agent.LENGTH_VIEW):
        """
        Return the local occupancy grids perceived by all agents.
        """

        local_grids = []
        for agent in self.agents:
            grid = self.grid_map.get_local_grid(agent, size)
            local_grids.append(grid)
        return local_grids

    def _is_free(
        self,
        entity: Entity,
        pos: np.ndarray,
        min_dist: float,
    ):
        """
        Check whether a position is collision-free.
        """
        for other in self.static_obstacles + self.moving_obstacles + self.agents:
            if other is entity:
                continue
            if other.current_position is None:
                continue
            if entity.collides_with(other, pos, min_dist):
                return False
        return True
