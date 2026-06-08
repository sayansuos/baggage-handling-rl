import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces

from simulator.configs.config import EnvConfig, AgentConfig, RewardConfig
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
        name: str,
        env_id: int = 1,
        debug: bool = False,
    ):
        """
        Constructor
        """
        self.name = name
        self.env_id = env_id
        self.debug = debug

        self.env_config = env_config
        self.agent_config = agent_config
        self.reward_config = reward_config
        # self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig = plt.gcf()
        self.ax = plt.gca()

        self.build_environment()

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

        _return, _mean_time_travel, _success_rate, _collision_rate, _debug = (
            self._update_info()
        )
        info = {
            "environment": f"{self.name}_{self.env_id}",
            "episode": self.episode,
            "return": _return,
            "mean_time_travel": _mean_time_travel,
            "success_rate": _success_rate,
            "collision_rate": _collision_rate,
            "total_step": self.total_step,
            "debug": _debug,
        }
        self.info = info
        return info

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.
        """

        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)

        self.moving_obstacles, self.agents = self.env_manager.reset()
        self._update_environment()

        obs = self._get_obs()
        info = self._get_info()

        self.step_count = 0
        self.episode += 1

        return obs, info

    def step(self, action=None) -> tuple[dict, dict, dict, dict, dict]:
        """
        Advance the environment by one simulation step.
        """
        self.step_count += 1
        self.total_step += 1

        self._simulate(action)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        info = self._get_info()

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
        """
        Advance the simulation by one step.
        """

        self._update_obstacles()
        if not action:
            self._update_agents()
        self._closest = self._compute_closest()
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
            if agent.state == "active":
                agent.travel_time += 1

    def _update_obstacles(self):
        """
        Update the positions of all moving obstacles.

        Each moving obstacle computes its next position according
        to its internal motion model or predefined path.
        """

        for obs in self.moving_obstacles:
            obs.step()

    def _update_info(self) -> tuple[float, float, float, float, list]:
        """
        Return auxiliary information for all agents.
        """

        n = len(self.agents)

        if self.info == {}:
            self._return = 0
            self._mean_time_travel = 0
            self._success_rate = 0
            self._collision_rate = 0
            self._debug = []

        else:
            self._return += sum(r for _, r in self.reward.items())
            self._mean_time_travel = sum([agent.travel_time for agent in self.agents])
            self._success_rate = sum(
                [agent.state == "terminated" for agent in self.agents]
            )
            self._collision_rate = sum(
                [agent.state == "truncated" for agent in self.agents]
            )

        if self.debug:
            debug = {
                "step": self.step_count,
                "agents": {},
            }
            for agent in self.agents:
                debug["agents"][agent.id] = {
                    "state": agent.state,
                    "old_pos": agent.old_position,
                    "current_pos": agent.current_position,
                    "goal_relative_distance": agent._goal_relative_distance,
                    "travel_time": agent.travel_time,
                    "reward": self.reward[agent.id],
                    "action": agent._motion.tolist(),
                    "closest_obstacle_distance": self._closest[agent.id][
                        "closest_distance"
                    ],
                }
            self._debug.append(debug)

        return (
            self._return,
            self._mean_time_travel / n,
            self._success_rate / n,
            self._collision_rate / n,
            self._debug,
        )

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
        for agent in self.agents:
            if agent.state == "reached":
                agent.state = "terminated"
        return {agent.id: agent.state == "terminated" for agent in self.agents}

    def _compute_truncated(self) -> dict:
        """
        Compute the truncation state of all agents.
        """
        for agent in self.agents:
            if agent.state == "collided":
                agent.state = "truncated"
        return {agent.id: agent.state == "truncated" for agent in self.agents}

    def _compute_reward(self) -> dict[str, float]:
        """
        Compute the rewards associated with all agents.
        """

        rewards = {}

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

            if agent.state in ["truncated", "terminated"]:
                rewards[agent.id] = 0

            else:
                reward = 0

                # Goal reached/progress reward
                current = agent._goal_relative_distance
                progress = (
                    agent._old_goal_relative_distance - agent._goal_relative_distance
                )
                reward += beta1 * (goal_bonus if current < 0.5 else progress)
                # Abrupt rotations penalty
                omega = abs(agent.omega)
                reward += beta2 * (
                    angular_malus * omega if omega > omega_threshold else 0
                )
                # Non-respect of safety distance penalty
                closest_dist = (
                    safety_threshold - self._closest[agent.id]["closest_distance"]
                )
                reward += beta3 * (
                    safety_malus1 * np.exp(safety_malus2 * closest_dist)
                    if closest_dist > 0
                    else 0
                )
                # Collision penalty
                reward += beta4 * (collision_malus if agent.state == "collided" else 0)

                rewards[agent.id] = reward

        self.reward = rewards

        return rewards

    # ---------------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------------

    def _set_debug(self, to: bool):
        self.debug = to

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

    def build_environment(self):
        """
        Create the full simulation world.
        """

        env_manager = EnvironmentManager(self.env_config, self.agent_config)
        self.env_manager = env_manager
        self.static_obstacles, self.moving_obstacles, self.agents = (
            self.env_manager.generate()
        )
        self._update_environment()
        self.observation_space, self.action_space = (
            self._build_environment_spaces()
        )  # Define Gym spaces

        self.episode = 1
        self.step_count = 0
        self.total_step = 0

    def _build_environment_spaces(self) -> tuple[spaces.Dict, spaces.Dict]:
        """
        Define observation and action spaces for each agent.
        """

        observation_space = spaces.Dict(
            {
                agent.id: get_single_observation_space(
                    self.env_config, self.agent_config
                )
                for agent in self.agents
            }
        )
        action_space = spaces.Dict(
            {
                agent.id: get_single_action_space(self.env_config)
                for agent in self.agents
            }
        )

        return observation_space, action_space

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
                if dist < 1e-3:
                    agent.state = "collided"

            metrics[agent.id] = {
                "closest_distance": min_distance,
                "closest_entity": closest_entity,
            }
        return metrics

    def _update_environment(self):
        """
        Regenerate the full environment state.
        """

        self.grid_map = GridMap(
            self.env_config,
            self.agent_config,
            self.static_obstacles,
            self.moving_obstacles,
            self.agents,
        )
        self._compute_astar_paths()
        self._closest = self._compute_closest()  # Get entities interaction
        self.reward: dict = {agent.id: 0 for agent in self.agents}
        self.info: dict = {}

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
                if path:
                    path = [self._from_grid(pos) for pos in path]
                    entity.path.extend(path)
                    start = goal
                else:
                    path = [entity.current_position]
