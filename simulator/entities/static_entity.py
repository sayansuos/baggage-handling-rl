from __future__ import annotations

from matplotlib.patches import Rectangle

from simulator.entities.entity import Entity
from simulator.geometry import (
    get_distance_rectangle_circle,
    get_distance_rectangle_rectangle,
)


class StaticEntity(Entity):
    """
    Base class for rectangular entities that remain fixed in the environment.
    """

    def __init__(self, width: float, height: float, num: int | None = None):
        """
        Constructor
        """

        super().__init__(num)
        self.width = width
        self.height = height

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

    @property
    def id(self):
        """Return the unique identifier of the static entity."""

        return f"static_entity_{self.num}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the entity at its current position.
        """

        assert self.current_position is not None
        return self.get_bounds_at(self.current_position)

    def get_bounds_at(
        self, pos: tuple[float, float]
    ) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the entity at a given position.
        """

        x, y = pos
        return (
            x - self.width / 2,
            y - self.height / 2,
            x + self.width / 2,
            y + self.height / 2,
        )

    def get_distance(self, other: StaticEntity | MovingEntity) -> float:
        """
        Compute the distance between the entity and another entity.
        """

        return other._get_distance_rectangle(self)

    def collides_with(
        self,
        other: StaticEntity | MovingEntity,
        new_pos: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check whether the entity would collide with another entity at the given positions.
        """

        assert other.current_position is not None
        return bool(
            other._collide_rectangle(self, other.current_position, new_pos, min_dist)
        )

    def _get_distance_circle(self, circle: MovingEntity) -> float:
        """
        Compute the distance between a rectangular entity and a circular entity.
        """

        assert circle.current_position is not None
        assert self.current_position is not None
        x_min, y_min, x_max, y_max = self.get_bounds_at(self.current_position)
        cx, cy = circle.current_position
        radius = circle.radius
        return get_distance_rectangle_circle(x_min, y_min, x_max, y_max, cx, cy, radius)

    def _get_distance_rectangle(self, rect: MovingEntity) -> float:
        """
        Compute the distance between two rectangular entities.
        """

        assert rect.current_position is not None
        assert self.current_position is not None
        x1_min, y1_min, x1_max, y1_max = self.get_bounds_at(self.current_position)
        x2_min, y2_min, x2_max, y2_max = rect.get_bounds_at(rect.current_position)
        return get_distance_rectangle_rectangle(
            x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max
        )

    def _collide_circle(
        self,
        circle: MovingEntity,
        pos_self: tuple[float, float],
        pos_other: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check whether a rectangular entity and a circular entity collide.
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
        Check whether two rectangular entities collide.
        """

        x1_min, y1_min, x1_max, y1_max = self.get_bounds_at(pos_self)
        x2_min, y2_min, x2_max, y2_max = rect.get_bounds_at(pos_other)
        dist = get_distance_rectangle_rectangle(
            x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max
        )
        return bool(dist <= min_dist)
