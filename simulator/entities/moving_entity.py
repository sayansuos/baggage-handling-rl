from matplotlib.patches import Circle

from simulator.entities.entity import Entity
from simulator.geometry import (
    get_distance_circle_circle,
    get_distance_rectangle_circle,
    get_relative_distance,
    get_relative_position,
)


class MovingEntity(Entity):
    """
    Represents a moving entity in the environment.

    Moving entities are mobile circular objects such as:
    robots, pedestrians, dynamic obstacles, etc.

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
    target_index : int
        Position index to reach in the targets' list of the entity.
    """

    def __init__(self, num: int | None = None):
        """
        Constructor
        """

        super().__init__(num)
        self.old_position: tuple[float, float] | None = None
        self.start_position: tuple[float, float] | None = None
        self.target_positions: list[tuple[float, float]] = []

        self.radius: float = 0
        self.theta: float = 0
        self.v: float = 0
        self.omega: float = 0

        self.path: list[tuple[int, int] | tuple[float, float]] = []
        self.path_index: int = 0
        self.target_index: int = 0

    def __str__(self):
        return f"moving_entity{self.num}"

    @property
    def positions(self) -> list[tuple]:
        """
        Return all the main positions of the circular entity.
        """

        assert self.start_position is not None
        return [self.start_position] + self.target_positions

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the axis-aligned bounding box of the circular entity.
        """

        assert self.current_position is not None
        return self.bounds_at(self.current_position)

    @property
    def _goal_relative_position(self):
        """
        Return the relative position of the current goal.

        The goal position is expressed in the world reference frame
        relative to the agent's current position.
        """
        if not self.target_positions:
            return (0, 0)
        return self.get_relative_position(self.target_positions[self.target_index])

    @property
    def _goal_relative_distance(self):
        """
        Return the relative distance of the current goal.
        """
        if not self.target_positions:
            return 0
        return self.get_relative_distance(self.target_positions[self.target_index])

    @property
    def _old_goal_relative_distance(self):
        """
        Return the relative distance of the current goal.
        """
        assert self.old_position is not None
        if not self.target_positions:
            return 0
        return get_relative_distance(
            self.old_position, self.target_positions[self.target_index]
        )

    def bounds_at(self, pos: tuple[float, float]) -> tuple[float, float, float, float]:
        """
        Return the axis-aligned bounding box of the circular entity at a given position.
        """

        x, y = pos
        return (
            x - self.radius,
            y - self.radius,
            x + self.radius,
            y + self.radius,
        )

    def get_relative_position(self, pos: tuple[float, float]) -> tuple[float, float]:
        """
        Compute the relative position of a given position.
        """

        assert self.current_position is not None
        return get_relative_position(self.current_position, pos)

    def get_relative_distance(self, pos: tuple[float, float]) -> float:
        """
        Compute the Euclidean distance between the current
        position and a given position.
        """

        assert self.current_position is not None
        return get_relative_distance(self.current_position, pos)

    def get_distance(self, other: StaticEntity | MovingEntity) -> float:
        """
        Compute the Euclidean distance between this circular entity and another one.
        """
        return other._get_distance_circle(self)

    def collides_with(
        self,
        other: MovingEntity | StaticEntity,
        new_pos: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check collision between this moving entity and another entity.

        This method dispatches the collision logic to the appropriate
        shape-specific implementation of the other entity.
        """

        assert other.current_position
        return bool(
            other._collide_circle(self, other.current_position, new_pos, min_dist)
        )

    def step(self) -> tuple[int, int] | tuple[float, float] | None:
        """
        Move the entity one step along its current path and reset it.
        """

        assert self.start_position is not None

        self.old_position = self.current_position
        self.path_index += 1

        if self.path_index >= len(self.path):

            self.path = self.path[::-1]
            self.target_positions = self.target_positions[::-1]
            self.target_positions.append(self.start_position)
            self.start_position = self.target_positions.pop(0)
            self.path_index = 0
            self.target_index = 0

        self.current_position = self.path[self.path_index]
        current_target = self.target_positions[self.target_index]

        if self.current_position == current_target:
            if self.target_index <= len(self.target_positions):
                self.target_index += 1

        return None

    def render(
        self,
        ax,
        color: str | tuple[float, float, float, float] = "black",
    ):
        """
        Default render method for moving entities.
        """

        assert self.start_position is not None
        x, y = self.start_position
        circle = Circle(
            (x, y),
            0.5,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            alpha=0.1,
        )
        ax.add_patch(circle)
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

    def _get_distance_circle(self, circle: MovingEntity) -> float:
        """
        Compute the Euclidean distance between two circular entity.
        """
        assert self.current_position is not None
        assert circle.current_position is not None
        cx1, cy1 = self.current_position
        radius1 = self.radius
        cx2, cy2 = circle.current_position
        radius2 = circle.radius
        return get_distance_circle_circle(cx1, cy1, radius1, cx2, cy2, radius2)

    def _get_distance_rectangle(self, rect: StaticEntity) -> float:
        """
        Compute the Euclidean distance between a circular entity and a
        rectangular one.
        """

        assert self.current_position is not None
        assert rect.current_position is not None
        x_min, y_min, x_max, y_max = rect.get_bounds_at(rect.current_position)
        cx, cy = self.current_position
        radius = self.radius
        return get_distance_rectangle_circle(x_min, y_min, x_max, y_max, cx, cy, radius)

    def _collide_circle(
        self,
        circle: MovingEntity,
        pos_self: tuple[float, float],
        pos_other: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check collision between this moving entity and a circular entity.
        """

        cx1, cy1 = pos_self
        radius1 = self.radius
        cx2, cy2 = pos_other
        radius2 = circle.radius
        dist = get_distance_circle_circle(cx1, cy1, radius1, cx2, cy2, radius2)
        return dist <= min_dist

    def _collide_rectangle(
        self,
        rect: StaticEntity,
        pos_self: tuple[float, float],
        pos_other: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check collision between this moving entity and a rectangular entity.
        """

        x_min, y_min, x_max, y_max = rect.get_bounds_at(pos_other)
        cx, cy = pos_self
        radius = self.radius
        dist = get_distance_rectangle_circle(x_min, y_min, x_max, y_max, cx, cy, radius)
        return dist <= min_dist
