from matplotlib.patches import Rectangle

from simulator.entities.entity import Entity
from simulator.utils.get_distance import (
    get_distance_rectangle_rectangle,
    get_distance_rectangle_circle,
)


class StaticEntity(Entity):
    """
    Represents a static entity in the environment.

    Static entities are immobile rectangular objects such as:
    walls, shelves, racks and fixed obstacles.

    Attributes
    ----------
    num : int
        Unique identifier of the object.
    current_position : tuple
        Current center position of the object.
    width : float
        Width of the rectangular obstacle.
    height : float
        Height of the rectangular obstacle.
    """

    def __init__(self, width: float, height: float, num: int | None = None):
        """
        Constructor
        """

        super().__init__(num)
        self.width = width
        self.height = height

    def __str__(self):
        return f"static_entity_{self.num}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the axis-aligned bounding box of the rectangular entity.
        """

        assert self.current_position is not None
        return self.get_bounds_at(self.current_position)

    def get_bounds_at(
        self, pos: tuple[float, float]
    ) -> tuple[float, float, float, float]:
        """
        Return the axis-aligned bounding box of the rectangular entity at a given position.
        """

        x, y = pos
        return (
            x - self.width / 2,
            y - self.height / 2,
            x + self.width / 2,
            y + self.height / 2,
        )

    def collides_with(
        self,
        other: StaticEntity | MovingEntity,
        new_pos: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check collision between this static entity and another entity.

        This method dispatches the collision logic to the appropriate
        shape-specific implementation of the other entity.
        """

        assert other.current_position is not None
        return bool(
            other._collide_rectangle(self, other.current_position, new_pos, min_dist)
        )

    def render(self, ax, color: str | tuple[float, float, float, float] = "black"):
        """
        Default render method for static entities.
        """

        x_min, y_min, x_max, y_max = self.bounds
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor=color,
            linewidth=0,
        )
        ax.add_patch(rect)

    def _collide_circle(
        self,
        circle: MovingEntity,
        pos_self: tuple[float, float],
        pos_other: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check collision between this static entity and a circular entity.
        """

        x_min, y_min, x_max, y_max = self.get_bounds_at(pos_self)
        cx, cy = pos_other
        radius = circle.radius
        dist = get_distance_rectangle_circle(x_min, y_min, x_max, y_max, cx, cy, radius)
        return bool(dist <= min_dist)

    def _collide_rectangle(
        self,
        rect: StaticEntity,
        pos_self: tuple[float, float],
        pos_other: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check collision between this static entity and a rectangular entity.
        """

        x1_min, y1_min, x1_max, y1_max = self.get_bounds_at(pos_self)
        x2_min, y2_min, x2_max, y2_max = rect.get_bounds_at(pos_other)
        dist = get_distance_rectangle_rectangle(
            x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max
        )
        return bool(dist <= min_dist)
