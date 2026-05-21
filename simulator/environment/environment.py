import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces

from simulator.utils.config import EnvConfig, AgentConfig, RewardConfig
from simulator.utils.environment_space import (
    get_single_observation_space,
    get_single_action_space,
)
from simulator.environment.environment_manager import EnvironmentManager
from simulator.environment.gridmap import GridMap
from simulator.entities.static_entity import StaticEntity
from simulator.entities.moving_entity import MovingEntity
from simulator.entities.agent import Agent
from simulator.motion.astar import AStar


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

    def __init__(
        self,
        env_config: EnvConfig,
        agent_config: AgentConfig,
        reward_config: RewardConfig,
    ):
        """
        Constructor
        """

        self.env_config = env_config
        self.agent_config = agent_config
        self.reward_config = reward_config
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        self._build_environment()  # Build simulation states
        self._build_environment_spaces()  # Define Gym spaces

    # ---------------------------------------------------------------
    # GYM API
    # ---------------------------------------------------------------

    def _get_obs(self) -> dict:
        """
        Compute the decentralized observation of all agents.

        The observation only contains information locally
        available to the agent.
        """

        obs = {}
        for agent in self.agents:
            obs[agent.id] = {
                "local_map": self._get_local_grid(agent)[np.newaxis, :, :],
                "goal_relative_distance": agent._goal_relative_distance,
                "motion": agent._motion,
                "orientation": agent._orientation,
            }
        return obs

    def _get_info(self) -> dict:
        """
        Return auxiliary information for all agents.
        """

        info = {}
        closest = self._compute_closest()
        for agent in self.agents:
            closest_entity = closest[agent.id]["closest_entity"]
            closest_distance = closest[agent.id]["closest_distance"]
            info[agent.id] = {
                "closest_entity": closest_entity,
                "closest_distance": closest_distance,
                "collision": agent.state in ["truncated", "collided"],
                "terminated": agent.state == "terminated",
            }
        return info

    def reset(self, seed=None, options=None) -> (dict, dict):
        """
        Reset the environment to an initial state.
        """

        super().reset(seed=seed)
        self._reset()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action=None) -> (dict, dict, dict, dict, dict):
        """
        Advance the environment by one simulation step.
        """

        self._simulate(action)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        info = self._get_info()
        print(reward)
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Default render method for the global environment.
        """

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        colors = plt.cm.get_cmap("tab10", len(self.agents))
        W, H = self.env_config.width, self.env_config.height

        for entity in self.static_obstacles:
            entity.render(self.ax)

        for i, entity in enumerate(self.moving_obstacles):
            entity.render(self.ax)

        for i, agent in enumerate(self.agents):
            agent.render(self.ax, 0, W, 0, H, color=colors(i))
            self.ax.scatter([], [], color=colors(i), label=agent.id)

        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.legend(loc="upper left")

    # ---------------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------------

    def _simulate(self, action=None):
        """ """
        self._update_obstacles()
        if not action:
            self._update_agents()
        # TODO : add other behavior for agents

    def _update_agents(self):
        """
        Update the positions of all moving agents.

        Each moving obstacle computes its next position according
        to its internal motion model or predefined path.

        The new position is applied only if a valid next position
        is returned.
        """

        for agent in self.agents:
            agent.step()

    def _update_obstacles(self):
        """
        Update the positions of all moving obstacles.

        Each moving obstacle computes its next position according
        to its internal motion model or predefined path.
        """

        for obs in self.moving_obstacles:
            obs.step()

    def _get_local_grid(
        self, agent: Agent, size: int | None = None
    ) -> np.ndarray | None | ValueError:
        """
        Return the local occupancy grid perceived by a given agent.

        The local grid is centered around the agent's current position
        and represents nearby occupied and free cells.
        """

        if not size:
            size = self.agent_config.length_view
        return self.grid_map.get_local_grid(agent, size)

    def _compute_terminated(self) -> dict:
        """
        Compute the termination state of all agents.
        """
        return {agent.id: agent.state == "terminated" for agent in self.agents}

    def _compute_truncated(self) -> dict:
        """
        Compute the truncation state of all agents.
        """
        return {
            agent.id: agent.state in ["truncated", "collided"] for agent in self.agents
        }

    def _compute_reward(self) -> dict[str, float]:
        """
        Compute the rewards associated with all agents.
        """

        rewards = {}
        closest = self._compute_closest()

        beta1 = self.reward_config.beta1
        beta2 = self.reward_config.beta2
        beta3 = self.reward_config.beta3
        beta4 = self.reward_config.beta4

        goal_bonus = self.reward_config.goal_bonus
        collision_malus = self.reward_config.collision_malus
        angular_malus = self.reward_config.angular_malus
        safety_malus1 = self.reward_config.safety_malus1
        safety_malus2 = self.reward_config.safety_malus2
        omega_threshold = self.reward_config.omega_threshold
        safety_threshold = self.reward_config.safety_threshold

        for agent in self.agents:
            reward = 0
            rewards[agent.id] = {}

            # Goal reached/progress reward
            current = agent._goal_relative_distance
            progress = agent._old_goal_relative_distance - agent._goal_relative_distance
            reward += beta1 * (goal_bonus if current < 0.05 else progress)
            # # Abrupt rotations penalty
            # omega = abs(agent.omega)
            # reward += beta2 * (angular_malus * omega if omega > omega_threshold else 0)
            # # Non-respect of safety distance penalty
            # closest_dist = safety_threshold - closest[agent.id]["closest_distance"]
            # reward += beta3 * (
            #     safety_malus1 * np.exp(safety_malus2 * closest_dist)
            #     if closest_dist > 0
            #     else 0
            # )
            # # Collision penalty
            # reward += beta4 * (collision_malus if agent.state == "collided" else 0)

            rewards[agent.id]["R"] = reward
            rewards[agent.id]["targets"] = agent.target_positions
            rewards[agent.id]["path_idx"] = agent.path_index

        return rewards

    def _compute_closest(self) -> dict:
        """
        Compute the closest surrounding entity for each agent.

        The closest entity is determined using the Euclidean
        distance between entities.
        """

        metrics = {}
        entities = self.static_obstacles + self.moving_obstacles + self.agents
        for agent in self.agents:
            min_distance = np.inf
            closest_entity = None
            for other in entities:
                if other == agent or other.current_position is None:
                    continue
                dist = agent.get_distance(other)
                if dist < min_distance:
                    min_distance = dist
                    closest_entity = other

            metrics[agent.id] = {
                "closest_distance": min_distance,
                "closest_entity": closest_entity,
            }
        return metrics

    # ---------------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """
        Return the static occupancy grid of the environment.
        """
        return self.grid_map.grid

    def _to_grid(self, pos: tuple[float, float]) -> tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        """
        return self.grid_map.world_to_grid(pos)

    def _from_grid(self, pos: tuple[int, int]) -> tuple[float, float]:
        """
        Convert grid coordinates back to world coordinates.
        """
        return self.grid_map.grid_to_world(pos)

    def _build_environment(self):
        """
        Create or reset the full simulation world.
        """

        env_manager = EnvironmentManager(self.env_config, self.agent_config)
        self.env_manager = env_manager
        self._reset()

    def _build_environment_spaces(self):
        """
        Define observation and action spaces for each agent.
        """

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

    def _reset(self):
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
        self._compute_astar_paths()

    def _is_free(
        self,
        entity: StaticEntity | MovingEntity | Agent,
        pos: tuple[int, int] | tuple[float, float],
        min_dist: float,
    ) -> bool:
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

    def _compute_astar_paths(self):
        """
        Return the paths found by A* algorithm for each moving obstacle.
        """

        pathfinder = AStar(self.grid, int(self.env_config.radius_max) // 2 + 1)

        for entity in list(self.moving_obstacles + self.agents):
            if not entity.target_positions or not entity.start_position:
                continue
            start = entity.start_position
            for goal in entity.target_positions:
                start_grid = self._to_grid(start)
                goal_grid = self._to_grid(goal)
                path = pathfinder.find_path(start_grid, goal_grid)
                path = [self._from_grid(pos) for pos in path]
                entity.path.extend(path)
                start = goal
