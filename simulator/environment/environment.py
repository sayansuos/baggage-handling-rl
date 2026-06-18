import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from configs.config import AgentConfig, EnvConfig, RewardConfig
from simulator.environment.gridmap import GridMap
from simulator.environment.manager import Manager
from simulator.environment.pathfinding import compute_astar_paths
from simulator.environment.rewards import compute_rewards
from simulator.environment.terminations import compute_closest, compute_dones
from simulator.geometry import (
    get_heading_error,
    get_normalized_heading_error,
    get_normalized_motion,
    get_normalized_relative_distance,
)
from simulator.spaces import get_multi_spaces


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

        # Initialization

        # self.fig, self.ax = None, None
        self.env_config = env_config
        self.agent_config = agent_config
        self.rewards_config = reward_config
        self.env_manager = Manager(self.env_config, self.agent_config)

        # Build environment

        self.name = name
        self.env_id = env_id
        self.episode = 0
        self.total_step = 0
        self.step_count = 0
        self.debug = debug

        self.observation_space, self.action_space = get_multi_spaces(
            nb_agents=self.env_config.nb_agents,
            env_config=self.env_config,
            agent_config=self.agent_config,
        )

        # Get global structures

        self.static_obstacles = self.env_manager.generate_static_obstacles()

    # ---------------------------------------------------------------
    # PROPERTIES
    # ---------------------------------------------------------------

    @property
    def timeout(self):
        """"""
        return self.step_count >= self.env_config.max_steps

    @property
    def grid(self) -> np.ndarray:
        """
        Return the static occupancy grid of the environment.
        """
        return self.grid_map.grid

    def done(self, agent_id: str | None = None) -> bool:
        """"""
        if not agent_id:
            return all(done for done in self.dones.values())
        else:
            return self.dones[agent_id]

    # ---------------------------------------------------------------
    # GYM API
    # ---------------------------------------------------------------

    def _get_obs(self) -> dict:
        """
        Compute the decentralized observation of all agents.
        """

        obs = {}
        current_grid = self.grid

        for agent in self.agents:

            local_map = self.grid_map.get_local_grid(
                agent, current_grid, self.agent_config.length_view
            )[np.newaxis, :, :]
            goal_relative_distance = get_normalized_relative_distance(
                agent._goal_relative_distance,
                self.env_config.width,
                self.env_config.height,
            )
            heading_error = get_normalized_heading_error(
                agent._goal_relative_position, agent.theta
            )
            motion = get_normalized_motion(
                agent._motion,
                self.env_config.v_max_allowed,
                self.env_config.omega_max_allowed,
            )
            orientation = np.array(list(agent._orientation), dtype=np.float32)

            obs[agent.id] = {
                "local_map": local_map,
                "goal_relative_distance": goal_relative_distance,
                "heading_error": heading_error,
                "motion": motion,
                "orientation": orientation,
            }

        return obs

    def _get_info(self, rewards_info: dict | None = None) -> dict:
        """
        Return auxiliary information for all agents.
        """

        n = len(self.agents)

        mean_time_travel = sum(agent.travel_time for agent in self.agents) / n
        success_rate = sum(agent.state == "terminated" for agent in self.agents) / n
        collision_rate = sum(agent.state == "truncated" for agent in self.agents) / n
        mean_v = self.sum_v / self.motion_count if self.motion_count > 0 else 0.0
        mean_abs_omega = (
            self.sum_abs_omega / self.motion_count if self.motion_count > 0 else 0.0
        )

        if self.debug:

            info = {}

            for agent in self.agents:

                if agent.current_position is not None:
                    x, y = agent.current_position
                else:
                    x, y = None, None
                heading_error = get_heading_error(
                    agent._goal_relative_position, agent.theta
                )

                if rewards_info is None:
                    rewards = {
                        "reward_progress": 0.0,
                        "reward_collision": 0.0,
                        "reward_safety": 0.0,
                        "reward_rotation": 0.0,
                    }
                else:
                    rewards = rewards_info[agent.id]

                info[agent.id] = {
                    "pos_x": x,
                    "pos_y": y,
                    "distance_to_goal": agent._goal_relative_distance,
                    "heading_error": heading_error,
                    "min_obstacle_distance": self._closest[agent.id][
                        "closest_distance"
                    ],
                    "v": agent.v,
                    "omega": agent.omega,
                    "state": agent.state,
                    **rewards,
                    "experiment": self.name,
                    "episode": self.episode,
                    "return_total": self.reward_total,
                    "mean_v": mean_v,
                    "mean_abs_omega": mean_abs_omega,
                    "success_rate": success_rate,
                    "collision_rate": collision_rate,
                    "mean_time_travel": mean_time_travel,
                    **self.reward_sums,
                }

        else:

            info = {
                "experiment": self.name,
                "episode": self.episode,
                "return_total": self.reward_total,
                "mean_v": mean_v,
                "mean_abs_omega": mean_abs_omega,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "mean_time_travel": mean_time_travel,
                **self.reward_sums,
            }

        return info

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.
        """

        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)

        # Reset episode number and step count

        self.step_count = 0
        self.episode += 1

        # Change moving entities setups and compute paths

        self.moving_obstacles, self.agents = self.env_manager.reset()
        self.grid_map = GridMap(
            self.env_config,
            self.agent_config,
            self.static_obstacles,
            self.moving_obstacles,
            self.agents,
        )
        compute_astar_paths(
            grid_map=self.grid_map,
            radius_max=self.env_config.radius_max,
            moving_obstacles=self.moving_obstacles,
            agents=self.agents,
        )

        # Get entities interactions

        self._closest = compute_closest(
            static_obstacles=self.static_obstacles,
            moving_obstacles=self.moving_obstacles,
            agents=self.agents,
        )

        # Reset metrics

        self.dones: dict = {agent.id: False for agent in self.agents}
        self.rewards: dict = {agent.id: 0 for agent in self.agents}
        self.reward_total = 0
        self.reward_sums = {
            "reward_progress": 0.0,
            "reward_collision": 0.0,
            "reward_safety": 0.0,
            "reward_rotation": 0.0,
        }
        self.sum_v = 0.0
        self.sum_abs_omega = 0.0
        self.motion_count = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action=None) -> tuple[dict, dict, dict, dict, dict]:
        """
        Advance the environment by one simulation step.
        """

        # Update step counts

        self.step_count += 1
        self.total_step += 1

        # Update obstacles

        for obs in self.moving_obstacles:
            obs.step()

        # Update agents

        for agent in self.agents:
            if action is not None and action[agent.id] is not None:
                v, omega = action[agent.id]
                agent.move(v, omega)
            else:
                agent.step()
            if agent.state == "active":
                agent.travel_time += 1
                self.sum_v += agent.v
                self.sum_abs_omega += abs(agent.omega)
                self.motion_count += 1

                if agent._goal_relative_distance < 0.5:
                    agent.state = "reached"

        # Get all metrics

        self._closest = compute_closest(
            static_obstacles=self.static_obstacles,
            moving_obstacles=self.moving_obstacles,
            agents=self.agents,
        )
        obs = self._get_obs()

        rewards, rewards_info = compute_rewards(
            reward_config=self.rewards_config,
            agents=self.agents,
            closest=self._closest,
            timeout=self.timeout,
        )
        self.rewards = rewards
        self.reward_total += sum(rewards.values())
        for _, agent_rewards_info in rewards_info.items():
            for reward_name, reward_value in agent_rewards_info.items():
                self.reward_sums[reward_name] += reward_value

        terminated, truncated, self.dones = compute_dones(
            agents=self.agents, timeout=self.timeout
        )

        info = self._get_info(rewards_info)

        return obs, rewards, terminated, truncated, info

    def render(self, ax=None):
        """
        Default render method for the global environment.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        ax.clear()
        ax.set_aspect("equal", adjustable="box")

        colors = plt.colormaps["tab10"]
        W, H = self.env_config.width, self.env_config.height

        for entity in self.static_obstacles:
            entity.render(ax)

        for i, entity in enumerate(self.moving_obstacles):
            entity.render(ax)

        handles = []
        labels = []
        for i, agent in enumerate(self.agents):
            agent.render(ax, 0, W, 0, H, color=colors(i))
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=colors(i),
                    label=agent.id,
                )
            )
            labels.append(agent.id)
            ax.scatter([], [], color=colors(i), label=agent.id)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"{self.name} | Episode {self.episode} | Step {self.step_count} | Return = {self.reward_total} "
        )
        ax.legend(handles=handles, labels=labels, loc="upper left")
