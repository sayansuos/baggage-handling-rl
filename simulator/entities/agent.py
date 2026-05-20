import numpy as np
from matplotlib.patches import Rectangle
from simulator.config import AgentConfig
from simulator.entities.moving_entity import MovingEntity


class Agent(MovingEntity):
    """
    Represents an autonomous moving agent in the environment.

    Agents are circular moving entities capable of:
    - navigating
    - perceiving their surroundings
    - reaching target positions

    Attributes
    ----------
    num : int
        Unique identifier of the agent.
    current_position : np.ndarray
        Current center position of the agent.
    start_position : np.ndarray | None
        Initial position of the agent.
    target_positions : list[np.ndarray]
        List of target positions to reach.
    radius : float
        Radius of the circular agent.
    v : float
        Linear velocity of the agent.
    omega : float
        Angular velocity of the agent.
    theta : float
        Current orientation angle in radians.
    length_view : int
        Perception range around the agent.
    state : bool
        State of the agent, either 'active', 'terminated' or 'truncated'.
    """

    def __init__(self, agent_config: AgentConfig, num: int = None):
        """
        Builder
        """
        super().__init__(num=num)
        self.radius = agent_config.radius
        self.length_view = agent_config.length_view
        self.state = "active"

    def __str__(self):
        return f"Agent n°{self.num}"

    @property
    def id(self):
        return f"agent_{self.num}"

    @property
    def _goal_relative_position(self):
        """
        Return the relative position of the current goal.

        The goal position is expressed in the world reference frame
        relative to the agent's current position.
        """

        x, y = self.current_position

        if self.target_positions:
            gx, gy = self.target_positions[0]
            goal = np.array([gx - x, gy - y], dtype=np.float64)
        else:
            goal = np.zeros(2, dtype=np.float64)

        return goal

    @property
    def _motion(self):
        """
        Return the current motion state of the agent.

        The motion state contains the linear and angular velocities.
        """
        return np.array([self.v, self.omega], dtype=np.float64)

    @property
    def _orientation(self):
        """
        Return the current orientation of the agent.

        The orientation is encoded using the cosine and sine
        of the heading angle in order to avoid angular
        discontinuities.
        """

        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float64)

    def move(self, v: float, omega: float, dt: float = 1):
        """
        Update the agent's position regarding a given velocity.
        """

        self.theta = omega * dt
        self.current_position = self._get_next_pos(v, omega, dt)

    def _get_next_pos(self, v: float, omega: float, dt: float = 1):
        """
        Regarding the agent's next position regarding its velocity.
        """

        x = v * np.cos(self.theta) * dt
        y = np.sin(self.theta) * dt
        return x, y

    def _step(self) -> tuple(int, int):
        """
        Move the entity one step along its current path and reset it.
        """

        if self.path_index == len(self.path) - 1:
            next_pos = self.current_position
            self.path = self.path[::-1]
            self.path_index = 0
            all_positions = self.positions[::-1]
            self.start_position = all_positions[0]
            self.target_positions = all_positions[1:]
        else:
            next_pos = self.path[self.path_index]
        return next_pos

    def get_vision_field(
        self,
        width_min: float,
        width_max: float,
        height_min: float,
        height_max: float,
    ) -> tuple[float, float, float, float]:
        """
        Compute the clipped perception area around the agent.

        The perception area is represented as an axis-aligned
        bounding box centered on the agent.
        """

        x, y = self.current_position

        x_min = np.clip(
            x - self.length_view / 2,
            width_min,
            width_max,
        )
        y_min = np.clip(
            y - self.length_view / 2,
            height_min,
            height_max,
        )
        x_max = np.clip(
            x + self.length_view / 2,
            width_min,
            width_max,
        )
        y_max = np.clip(
            y + self.length_view / 2,
            height_min,
            height_max,
        )
        return x_min, y_min, x_max, y_max

    def render(
        self,
        ax,
        width_min: float,
        width_max: float,
        height_min: float,
        height_max: float,
        color="red",
    ):
        """
        Default render method for moving agents.
        """

        super().render(ax, color=color)

        x_min, y_min, x_max, y_max = self.get_vision_field(
            width_min, width_max, height_min, height_max
        )
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor=color,
            alpha=0.1,
        )
        ax.add_patch(rect)
