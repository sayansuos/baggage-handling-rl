import numpy as np
from entities.moving_entity import MovingEntity
from matplotlib.patches import Rectangle


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
    map_size : int
        Size of the agent's local map.
    """

    V_MIN, V_MAX = -10, 10
    OMEGA_MIN, OMEGA_MAX = -np.pi, np.pi
    RADIUS = 1
    LENGTH_VIEW = 5
    MAP_SIZE = LENGTH_VIEW**2

    def __init__(self, num: int = None):
        """
        Builder
        """

        super().__init__(radius=self.RADIUS, num=num)
        self.length_view = self.LENGTH_VIEW
        self.map_size = self.MAP_SIZE
        self.theta = np.random.uniform(-np.pi, np.pi)

    def __str__(self):
        return f"Agent_{self.num}"

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

        x_min, y_min, x_max, y_max = self.bounds

        x_min = np.clip(
            x_min - self.length_view / 2,
            width_min,
            width_max,
        )
        y_min = np.clip(
            y_min - self.length_view / 2,
            height_min,
            height_max,
        )
        x_max = np.clip(
            x_max + self.length_view / 2,
            width_min,
            width_max,
        )
        y_max = np.clip(
            y_max + self.length_view / 2,
            height_min,
            height_max,
        )
        return x_min, y_min, x_max, y_max

    @property
    def local_map(self):
        """
        Return the local map perceived by the agent.

        The local map represents the surrounding environment
        inside the agent's perception range.
        """
        pass

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
