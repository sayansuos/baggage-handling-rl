import numpy as np
from matplotlib.patches import Circle, Rectangle

from configs.config import AgentConfig
from simulator.entities.moving_entity import MovingEntity
from simulator.geometry import get_relative_distance


class Agent(MovingEntity):
    """
    Mobile agent controlled by a navigation policy.
    """

    def __init__(self, agent_config: AgentConfig, num: int | None = None):
        """
        Constructor
        """

        super().__init__(num=num)

        self.old_position: tuple[float, float] | None = None
        self.target_index: int = 0
        self.radius = agent_config.radius
        self.length_view: int = agent_config.length_view
        self.local_maps: list[np.ndarray] = []

        self.state: str = "active"
        self.travel_time: int = 0
        self._closest_dist: float = np.inf

    def step(self, action: dict, dt: float = 1) -> tuple[float, float, int]:
        """
        Update the agent state according to the applied action.
        """

        self.old_v = self.v
        self.old_position = self.current_position

        # Update position

        v, omega = action[self.id]
        self.v = v
        self.omega = omega
        self.theta += omega * dt

        self.current_position = self._get_next_pos(v, dt)
        self.path.append(self.current_position)

        # Update states

        motion_count = 0
        if self.state == "active":
            self.travel_time += 1
            motion_count = 1

            if self._goal_relative_distance < 0.5:
                self.state = "reached"

        return (v, abs(self.omega), motion_count)

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
        Render the agent, its targets, trajectory, and field of view.
        """

        assert self.start_position is not None
        x, y = self.start_position
        circle = Circle(
            (x, y),
            0.5,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
            alpha=0.1,
        )
        ax.add_patch(circle)

        if self.target_positions:
            for i, pos in enumerate(self.target_positions):
                face_color = color if i < self.target_index else "white"
                tx, ty = pos
                target = Circle(
                    (tx, ty),
                    0.5,
                    facecolor=face_color,
                    edgecolor=color,
                    linewidth=2,
                    alpha=0.2,
                )
                ax.add_patch(target)

        assert self.current_position is not None
        x, y = self.current_position
        circle = Circle(
            (x, y),
            self.radius,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
        )
        ax.add_patch(circle)
        ax.arrow(
            x,
            y,
            self.radius * np.cos(self.theta),
            self.radius * np.sin(self.theta),
            head_width=0.1,
            head_length=0.5,
            length_includes_head=True,
            color="red",
        )

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

        if hasattr(self, "path"):
            xs = [p[0] for p in self.path[self.path_index :]]
            ys = [p[1] for p in self.path[self.path_index :]]
            ax.plot(xs, ys, "--", color=color, alpha=0.1, linewidth=2)

    def get_vision_field(
        self,
        width_min: float,
        width_max: float,
        height_min: float,
        height_max: float,
    ) -> tuple[float, float, float, float]:
        """
        Return the bounds of the agent's local field of view.
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

    @property
    def id(self) -> str:
        """
        Return the unique identifier of the agent.
        """

        return f"agent_{self.num}"

    @property
    def _goal_relative_position(self):
        """
        Return the relative position of the current target.
        """

        if not self.target_positions:
            return (0, 0)
        return self.get_relative_position(self.target_positions[self.target_index])

    @property
    def _goal_relative_distance(self):
        """
        Return the distance to the current target.
        """

        if not self.target_positions:
            return 0
        return self.get_relative_distance(self.target_positions[self.target_index])

    @property
    def _old_goal_relative_distance(self):
        """
        Return the previous distance to the current target.
        """

        assert self.old_position is not None
        if not self.target_positions:
            return 0
        return get_relative_distance(
            self.old_position, self.target_positions[self.target_index]
        )

    @property
    def _motion(self) -> tuple[float, float]:
        """
        Return the current linear and angular velocities.
        """

        return (self.v, self.omega)

    @property
    def _orientation(self) -> tuple[float, float]:
        """
        Return the current orientation as a unit vector.
        """

        return np.cos(self.theta), np.sin(self.theta)
