import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from entities.agent import Agent
from simulator_env.environment_manager import EnvironmentManager


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

    def _get_obs(self, agent: Agent) -> dict:
        """
        Compute the decentralized observation of an agent.

        The observation only contains information locally
        available to the agent.
        """

        x, y = agent.current_position

        if agent.target_positions:
            gx, gy = agent.target_positions[0]
            goal = np.array([gx - x, gy - y], dtype=np.float64)
        else:
            goal = np.zeros(2, dtype=np.float64)

        return {
            "local_map": self._compute_local_map(agent),
            "goal_relative_position": goal,
            "motion": np.array([agent.v, agent.omega], dtype=np.float64),
            "orientation": np.array(
                [np.cos(agent.theta), np.sin(agent.theta)], dtype=np.float64
            ),
        }

    def _get_info_agents(self):
        pass

    def get_info_obstacles(self):
        pass

    def get_all_positions(self):

        return [object.current_position for object in self.objects]

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self):
        """
        Default render method for the global environment.
        """
        self.ax.clear()
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

        plt.pause(10)
