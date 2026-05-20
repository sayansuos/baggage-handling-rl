import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from simulator.utils.config import EnvConfig, AgentConfig
from simulator.entities.agent import Agent
from simulator.entities.entity import Entity
from simulator.motion.astar import AStar
from simulator.environment.gridmap import GridMap
from simulator.environment.environment_manager import EnvironmentManager
from simulator.utils.environment_space import *


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

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config: EnvConfig, agent_config: AgentConfig):
        """
        Builder
        """
        self.env_config = env_config
        self.agent_config = agent_config

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.width, self.height = env_config.width, env_config.height
        self.v_min_allowed, self.v_max_allowed = (
            env_config.v_min_allowed,
            env_config.v_max_allowed,
        )
        self.omega_min_allowed, self.omega_max_allowed = (
            env_config.omega_min_allowed,
            env_config.omega_max_allowed,
        )

        # ---------------------------------------------------------------
        # Environment Setup
        # ---------------------------------------------------------------

        env_manager = EnvironmentManager(env_config, agent_config)
        self.env_manager = env_manager
        self.grid_map = None
        self._reset_world(True)

        # ---------------------------------------------------------------
        # Environment Spaces
        # ---------------------------------------------------------------

        self.observation_space = spaces.Dict(
            {
                agent.id: get_single_observation_space(
                    self.env_config, self.agent_config
                )
                for agent in self.agents
            }
        )
        self.action_space = spaces.Dict(
            {
                agent.id: get_single_action_space(self.env_config)
                for agent in self.agents
            }
        )

    def _get_obs(self) -> dict:
        """
        Compute the decentralized observation of an agent.

        The observation only contains information locally
        available to the agent.
        """

        obs = {}
        for agent in self.agents:
            obs[agent.id] = {
                "local_map": self._get_local_grid(agent)[np.newaxis, :, :],
                "goal_relative_position": agent._goal_relative_position,
                "motion": agent._motion,
                "orientation": agent._orientation,
            }
        return obs

    def _get_info(self) -> dict:
        """
        Return auxiliary information for all agents.
        """

        pass

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        """

        super().reset(seed=seed)
        self._reset_world()
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action=None):
        """
        Advance the environment by one simulation step.

        Moving obstacles and agents are updated according
        to their motion model or the provided actions.
        """

        self._update_obstacles()
        self._update_agents(action)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._compute_state()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Default render method for the global environment.
        """
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        colors = plt.cm.get_cmap("tab10", len(self.agents))

        for entity in self.static_obstacles:
            entity.render(self.ax)

        for i, entity in enumerate(self.moving_obstacles):
            entity.render(self.ax)

        for i, agent in enumerate(self.agents):
            agent.render(self.ax, 0, self.width, 0, self.height, color=colors(i))
            self.ax.scatter([], [], color=colors(i), label=f"Agent n°{agent.num}")

        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.legend(loc="upper left")

    # ---------------------------------------------------------------
    # Methods needed for updates
    # ---------------------------------------------------------------

    def _reset_world(self, astar_for_agents: bool = False):
        """
        Regenerate the full environment state.
        """

        self.static_obstacles = self.env_manager.generate_static_obstacles()
        self.moving_obstacles = self.env_manager.generate_moving_obstacles()
        self.agents = self.env_manager.generate_agents()
        self.grid_map = GridMap(
            self.env_config,
            self.agent_config,
            self.static_obstacles,
            self.moving_obstacles,
            self.agents,
        )
        self._compute_astar_paths(astar_for_agents)

    @property
    def grid(self):
        """
        Return the static occupancy grid of the environment.
        """

        return self.grid_map.grid

    def _to_grid(self, pos: tuple):
        """
        Convert world coordinates to grid coordinates.
        """
        return self.grid_map.world_to_grid(pos)

    def _from_grid(self, pos: tuple):
        """
        Convert grid coordinates back to world coordinates.
        """
        return self.grid_map.grid_to_world(pos)

    def _compute_astar_paths(self, for_agents: bool = False):
        """
        Return the paths found by A* algorithm for each moving obstacle.
        """

        pathfinder = AStar(self.grid, int(self.agent_config.radius) + 1)

        if for_agents:
            entities = self.moving_obstacles + self.agents
        else:
            entities = self.moving_obstacles

        for entity in entities:
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

    def _update_agents(self, action=None):
        """
        Update the positions of all moving agents.

        Each moving obstacle computes its next position according
        to its internal motion model or predefined path.

        The new position is applied only if a valid next position
        is returned.
        """

        for agent in self.agents:
            next_pos = agent.step()
            if self._is_free(agent, next_pos, 0):
                agent.current_position = next_pos
                agent.path_index += 1
            else:
                continue

    def _update_obstacles(self):
        """
        Update the positions of all moving obstacles.

        Each moving obstacle computes its next position according
        to its internal motion model or predefined path.

        The new position is applied only if a valid next position
        is returned.
        """

        for obs in self.moving_obstacles:
            next_pos = obs.step()
            if self._is_free(obs, next_pos, 0):
                obs.current_position = next_pos

    def _get_local_grid(self, agent: Agent, size: int = None):
        """
        Return the local occupancy grid perceived by a given agent.

        The local grid is centered around the agent's current position
        and represents nearby occupied and free cells.
        """
        if not size:
            size = self.agent_config.length_view

        return self.grid_map.get_local_grid(agent, size)

    def _get_local_grids(self, size: int = None):
        """
        Return the local occupancy grids perceived by all agents.
        """

        if not size:
            size = self.agent_config.length_view

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

    def _compute_state(self):
        """
        Compute the termination state of all agents.

        Each agent can either:
        - continue interacting with the environment
        - terminate naturally
        - be truncated due to external conditions
        """

        terminated = {}
        truncated = {}
        for agent in self.agents:
            terminated[agent.id] = False
            truncated[agent.id] = False
            if agent.state == "terminated":
                terminated[agent.id] = True
            elif agent.state != "truncated":
                truncated[agent.id] = True
            else:
                continue
        return terminated, truncated

    def _compute_reward(self):
        """
        Compute the rewards associated with all agents.
        """
        pass
