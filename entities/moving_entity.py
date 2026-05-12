import numpy as np
from threading import Thread
from matplotlib.patches import Circle
from entities.entity import Entity
from motion.astar import AStar


class MovingEntity(Entity, Thread):
    """
    Represents a moving entity in the environment.

    Moving entities are mobile circular objects such as:
    robots, pedestrians, dynamic obstacles, etc.

    Attributes
    ----------
    num : int
        Unique identifier of the entity.
    current_position : np.ndarray
        Current center position of the entity.
        Format: [x, y].
    start_position : np.ndarray
        Initial position of the entity.
    target_positions : list[np.ndarray]
        List of target positions to reach.
    radius : float
        Radius of the circular entity.
    v : float
        Linear velocity of the entity.
    omega : float
        Angular velocity of the entity.
    paths : list[tuple]
    """

    def __init__(self, radius: float, v: float = 0, omega: float = 0, num: int = None):
        """
        Builder
        """

        super().__init__(num)
        self.start_position = None
        self.target_positions = []
        self.radius = radius
        self.v = v
        self.omega = omega
        self.paths = []

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the axis-aligned bounding box of the circular entity.
        """

        x, y = self.current_position

        return (
            x - self.radius,
            y - self.radius,
            x + self.radius,
            y + self.radius,
        )

    def collides_with(
        self,
        other: Entity,
        new_pos: np.array,
        min_dist: float,
    ):
        """
        Check collision between this moving entity and another entity.

        This method dispatches the collision logic to the appropriate
        shape-specific implementation of the other entity.
        """
        return other._collide_circle(self, other.current_position, new_pos, min_dist)

    def _collide_circle(
        self,
        circle: MovingEntity,
        pos_self: np.array,
        pos_other: np.array,
        min_dist: float,
    ):
        """
        Check collision between this moving entity and a circular entity.
        """
        dist = np.linalg.norm(np.array(pos_self) - np.array(pos_other))
        return dist < self.radius + circle.radius + min_dist

    def _collide_rectangle(
        self,
        rect: Entity,
        pos_self: np.array,
        pos_other: np.array,
        min_dist: float,
    ):
        """
        Check collision between this moving entity and a rectangular entity.
        """
        x, y = pos_self
        x_min, y_min, x_max, y_max = rect.get_bounds_at(pos_other)

        closest_x = np.clip(x, x_min, x_max)
        closest_y = np.clip(y, y_min, y_max)

        dist = np.linalg.norm([x - closest_x, y - closest_y])

        return dist < self.radius + min_dist

    def get_distance_from_target(self) -> float:
        """
        Compute the Euclidean distance between the current
        position and the current target position.
        """

        return np.linalg.norm(
            np.array(self.current_position) - np.array(self.target_positions[0])
        )

    def render(
        self,
        ax,
        color="black",
    ):
        """
        Default render method for moving entities.
        """
        x, y = self.current_position

        circle = Circle(
            (x, y),
            self.radius,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
        )
        outline = Circle(
            (x, y),
            self.radius,
            fill=False,
            edgecolor="black",
            linewidth=1,
            linestyle="--",
        )
        ax.add_patch(circle)
        ax.add_patch(outline)

        if self.target_positions:
            for pos in self.target_positions:
                tx, ty = pos
                ax.scatter(tx, ty, color=color, marker="x")

        if hasattr(self, "paths"):
            for path in self.paths:
                if len(path) < 2:
                    continue
                ys = [p[0] for p in path]
                xs = [p[1] for p in path]
                ax.plot(xs, ys, "--", color=color, alpha=0.2, linewidth=1)
