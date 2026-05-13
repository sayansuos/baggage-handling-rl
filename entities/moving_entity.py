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
    path : list[tuple]
        Path of the entity.
    path_index : int
        Position index in the path of the entity.
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

        self.path = []
        self.path_index = 0

    def __str__(self):
        return f"MovingEntity_{self.num}"

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

    @property
    def positions(self) -> list[tuple]:
        """
        Return all the main positions of the circular entity.
        """
        return [self.start_position] + self.target_positions

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
            self.path_index += 1
        return next_pos

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
        ax.add_patch(circle)

        if self.target_positions:
            for pos in self.target_positions:
                tx, ty = pos
                target = Circle(
                    (tx, ty),
                    0.5,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0,
                    alpha=0.1,
                )
                ax.add_patch(target)

        if hasattr(self, "path"):
            if len(self.path) >= 2:
                xs = [p[0] for p in self.path[self.path_index :]]
                ys = [p[1] for p in self.path[self.path_index :]]
                ax.plot(xs, ys, "--", color=color, alpha=0.2, linewidth=1)
