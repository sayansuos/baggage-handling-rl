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
from simulator.spaces import get_action_space, get_observation_space


class Environment(gym.Env):
    """
    Multi-agent navigation environment following the Gymnasium API,
    concepted for parameter sharing.
    """

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

        # Initialize the environment state
        self.name: str = name
        self.env_id: int = env_id

        self.episode: int = 0
        self.step_count: int = 0
        self.debug: bool = debug

        # Store the environment configurations
        self.env_config: EnvConfig = env_config
        self.agent_config: AgentConfig = agent_config
        self.reward_config: RewardConfig = reward_config

        # Create the observation and action spaces
        self.observation_space = get_observation_space(agent_config=self.agent_config)
        self.action_space = get_action_space(agent_config=self.agent_config)

        # Generate the static obstacles
        self.env_manager: Manager = Manager(
            env_config=self.env_config, agent_config=self.agent_config
        )
        self.static_obstacles = self.env_manager.generate_static_obstacles()

        # Define focus agents for metrics and rewards
        self.n_focus_agents = self.env_config.nb_agents
        self.focus_agents = []

    def set_focus_agents(self, n_focus_agents: int) -> None:
        """
        Set the number of agents considered for rewards, metrics,
        and episode termination.
        """

        if n_focus_agents < 1:
            raise ValueError("n_focus_agents must be greater than 0.")

        self.n_focus_agents = min(n_focus_agents, self.env_config.nb_agents)

    # ---------------------------------------------------------------
    # PROPERTIES
    # ---------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """
        Return the static occupancy grid of the environment.
        """
        return self.grid_map.grid

    @property
    def timeout(self) -> bool:
        """
        Return whether the maximum episode length has been reached.
        """
        return bool(self.step_count >= self.env_config.max_steps)

    def done(self) -> bool:
        """
        Return whether all focus agents are done.
        """
        return all(self.dones.values())

    # ---------------------------------------------------------------
    # GYM API
    # ---------------------------------------------------------------

    def _get_obs(self) -> dict:
        """
        Compute the decentralized observation of all agents.
        """

        obs = {}

        # Retrieve the current occupancy grid.
        current_grid = self.grid_map.current_grid

        for ag in self.agents:
            # Update the history of local occupancy maps.
            local_map = self.grid_map.get_local_grid(
                agent=ag, current_grid=current_grid
            )[np.newaxis, :, :]
            if ag.local_maps == []:
                local_maps = [local_map for _ in range(self.agent_config.n_maps)]
            else:
                local_maps = ag.local_maps
                local_maps.pop(0)
                local_maps.append(local_map)
            ag.local_maps = local_maps

            # Compute the normalized observation components
            goal_relative_distance = get_normalized_relative_distance(
                distance=ag._goal_relative_distance,
                env_width=self.env_config.width,
                env_height=self.env_config.height,
            )
            heading_error = get_normalized_heading_error(
                goal_relative_position=ag._goal_relative_position, theta=ag.theta
            )
            motion = get_normalized_motion(
                motion=ag._motion,
                v_max=self.agent_config.v_max,
                omega_max=self.agent_config.omega_max,
            )
            orientation = np.array(list(ag._orientation), dtype=np.float32)

            # Store the observation
            obs[ag.id] = {
                "local_map": np.concatenate(ag.local_maps, axis=0),
                "goal_relative_distance": goal_relative_distance,
                "heading_error": heading_error,
                "motion": motion,
                "orientation": orientation,
            }

        return obs

    def _get_info(self, rewards_info: dict | None = None) -> dict:
        """
        Return auxiliary information for all selected agents.
        """
        n = len(self.focus_agents)

        mean_time_travel = sum(agent.travel_time for agent in self.focus_agents) / n
        success_rate = (
            sum(agent.state == "terminated" for agent in self.focus_agents) / n
        )
        collision_rate = (
            sum(agent.state == "truncated" for agent in self.focus_agents) / n
        )
        mean_v = self.sum_v / self.motion_count if self.motion_count > 0 else 0.0
        mean_abs_omega = (
            self.sum_abs_omega / self.motion_count if self.motion_count > 0 else 0.0
        )

        global_info = {
            "task": self.name,
            "episode": self.episode,
            "return_total": self.reward_total,
            "mean_v": mean_v,
            "mean_abs_omega": mean_abs_omega,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "mean_time_travel": mean_time_travel,
            **self.reward_sums,
        }

        # If not debug, return global environment statistics.
        if not self.debug:
            return global_info

        # Otherwise, also build detailed information for each focus agent.

        info = {}

        for ag in self.focus_agents:
            if ag.current_position is not None:
                x, y = ag.current_position
            else:
                x, y = None, None

            heading_error = get_heading_error(
                goal_relative_position=ag._goal_relative_position,
                theta=ag.theta,
            )

            closest_entity_dist = self._closest[ag.id]["closest_distance"]
            closest_entity = self._closest[ag.id]["closest_entity"]
            closest_id = None if closest_entity is None else closest_entity.id

            if rewards_info is not None:
                rewards = rewards_info[ag.id]
            else:
                rewards = {
                    "reward_progress": 0.0,
                    "reward_collision": 0.0,
                    "reward_safety": 0.0,
                    "reward_rotation": 0.0,
                }

            info[ag.id] = {
                "pos_x": x,
                "pos_y": y,
                "distance_to_goal": ag._goal_relative_distance,
                "heading_error": heading_error,
                "closest_entity_dist": closest_entity_dist,
                "closest_entity": closest_id,
                "v": ag.v,
                "omega": ag.omega,
                "state": ag.state,
                **rewards,
                **global_info,
            }

        return info

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.
        """

        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)

        # Reset the episode counter
        self.step_count = 0
        self.episode += 1

        # Reset the moving obstacles, the agents and the focus agents
        self.moving_obstacles, self.agents = self.env_manager.reset()
        self.focus_agents = self.agents[: self.n_focus_agents]

        # Rebuild the occupancy grid
        self.grid_map = GridMap(
            env_config=self.env_config,
            agent_config=self.agent_config,
            static_obstacles=self.static_obstacles,
            moving_obstacles=self.moving_obstacles,
            agents=self.agents,
        )
        compute_astar_paths(
            grid_map=self.grid_map,
            radius_max=self.env_config.radius_max,
            moving_obstacles=self.moving_obstacles,
            min_length=self.env_config.max_steps,
        )

        # Compute the interactions between entities for focus agents
        self._closest = compute_closest(
            static_obstacles=self.static_obstacles,
            moving_obstacles=self.moving_obstacles,
            agents=self.focus_agents,
        )

        # Reset the episode metrics
        self.dones: dict = {agent.id: False for agent in self.focus_agents}
        self.rewards: dict = {agent.id: 0 for agent in self.focus_agents}
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

        # Compute obs and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action=None) -> tuple[dict, dict, dict, dict, dict]:
        """
        Advance the environment by one simulation step.
        """

        # Update the episode counter
        self.step_count += 1

        # Update the moving obstacles
        for obs in self.moving_obstacles:
            obs.step()

        # Update the agents and the environment metrics
        for ag in self.agents:
            v, omega, motion_count = ag.step(action=action)
            if ag in self.focus_agents:
                self.sum_v += v
                self.sum_abs_omega += omega
                self.motion_count += motion_count

        # Compute the interactions between entities for focus agents
        self._closest = compute_closest(
            static_obstacles=self.static_obstacles,
            moving_obstacles=self.moving_obstacles,
            agents=self.focus_agents,
        )

        # Compute the episode rewards
        rewards, rewards_info = compute_rewards(
            reward_config=self.reward_config, agents=self.focus_agents
        )
        self.rewards = rewards
        self.reward_total += np.mean(list(rewards.values()))
        for r in self.reward_sums:
            self.reward_sums[r] += np.mean(
                [ag_reward[r] for ag_reward in rewards_info.values()]
            )

        # Compute the termination conditions
        terminated, truncated, dones = compute_dones(
            agents=self.agents, timeout=self.timeout
        )
        terminated = {ag.id: terminated[ag.id] for ag in self.focus_agents}
        truncated = {ag.id: truncated[ag.id] for ag in self.focus_agents}
        self.dones = {ag.id: dones[ag.id] for ag in self.focus_agents}

        # Compute obs and info
        obs = self._get_obs()
        info = self._get_info(rewards_info=rewards_info)

        return obs, rewards, terminated, truncated, info

    def render(self, ax=None):
        """
        Default render method for the global environment.
        """

        # Create a new figure when no axes are provided
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        # Configure the plot
        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        W, H = self.env_config.width, self.env_config.height

        # Draw the static obstacles
        for entity in self.static_obstacles:
            entity.render(ax=ax)

        # Draw the moving obstacles
        for i, entity in enumerate(self.moving_obstacles):
            entity.render(ax=ax, color="black")

        # Draw the agents and build the legend
        handles = []
        labels = []
        colors = plt.colormaps["tab10"]
        for i, agent in enumerate(self.agents):
            agent.render(
                ax=ax,
                width_min=0,
                width_max=W,
                height_min=0,
                height_max=H,
                color=colors(i),
            )
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

        # Configure the axes and legend
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"{self.name} | Episode {self.episode} | Step {self.step_count} | Return = {self.reward_total} "
        )
