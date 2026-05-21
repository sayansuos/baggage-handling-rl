import numpy as np

from matplotlib.patches import Rectangle

from simulator.utils.config import AgentConfig
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
    Attributes
    ----------
    num : int
        Unique identifier of the entity.
    current_position : tuple[float, float]
        Current center position of the entity.
        Format: [x, y].
    old_position : tuple[int, int]
        Previous center position of the entity.
    start_position : tuple[float, float]
        Initial position of the entity.
    target_positions : list[tuple[float, float]]
        List of target positions to reach.
    radius : float
        Radius of the circular entity.
    theta : float
        Current orientation angle in radians.
    v : float
        Linear velocity of the entity.
    omega : float
        Angular velocity of the entity.
    path : list[tuple]
        Path of the entity.
    path_index : int
        Position index in the path of the entity.
    length_view : int
        Perception range around the agent.
    state : bool
        State of the agent, either 'active', 'terminated' or 'truncated'.
    """

    def __init__(self, agent_config: AgentConfig, num: int | None = None):
        """
        Builder
        """
        super().__init__(num=num)
        self.radius = agent_config.radius
        self.length_view: int = agent_config.length_view
        self.state: str = "active"

    def __str__(self):
        return f"agent_{self.num}"

    @property
    def id(self) -> str:
        """
        Return the identifier of the agent.
        """

        return f"agent_{self.num}"

    def move(self, v: float, omega: float, dt: float = 1):
        """
        Update the agent's position regarding a given velocity.
        """

        self.theta += omega * dt
        self.current_position = self._get_next_pos(v, dt)

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

        assert self.current_position is not None
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
        color: str | tuple[float, float, float, float] = "red",
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

    @property
    def _motion(self) -> np.ndarray:
        """
        Return the current motion state of the agent.

        The motion state contains the linear and angular velocities.
        """
        return np.array([self.v, self.omega])

    @property
    def _orientation(self) -> np.ndarray:
        """
        Return the current orientation of the agent.

        The orientation is encoded using the cosine and sine
        of the heading angle in order to avoid angular
        discontinuities.
        """

        return np.array([np.cos(self.theta), np.sin(self.theta)])

    def _get_next_pos(self, v: float, dt: float = 1) -> tuple[float, float]:
        """
        Regarding the agent's next position regarding its velocity.
        """

        assert self.current_position is not None
        x, y = self.current_position
        x += v * np.cos(self.theta) * dt
        y += np.sin(self.theta) * dt
        return x, y
