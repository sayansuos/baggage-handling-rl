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

        super().__init__(num=num)

        self.num: int = num
        self.current_position: tuple[float, float] | None = None

        self.width: float = width
        self.height: float = height

    def render(
        self, ax, color: str | tuple[float, float, float, float] = "black"
    ) -> None:
        """
        Default render method for static entities.
        """

        # Draw the static entity as a rectangle
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

        return other._get_distance_rectangle(rect=self)

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

        collides_with = bool(
            other._collide_rectangle(
                rect=self,
                pos_self=other.current_position,
                pos_other=new_pos,
                min_dist=min_dist,
            )
        )

        return collides_with

    def _get_distance_circle(self, circle: MovingEntity) -> float:
        """
        Compute the distance between a rectangular entity and a circular entity.
        """

        assert circle.current_position is not None
        assert self.current_position is not None

        # Extract the rectangle bounds and the circle geometry
        x_min, y_min, x_max, y_max = self.get_bounds_at(pos=self.current_position)
        cx, cy = circle.current_position
        radius = circle.radius

        # Compute the distance
        dist = get_distance_rectangle_circle(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            cx=cx,
            cy=cy,
            radius=radius,
        )

        return dist

    def _get_distance_rectangle(self, rect: MovingEntity) -> float:
        """
        Compute the distance between two rectangular entities.
        """

        assert rect.current_position is not None
        assert self.current_position is not None

        # Extract the bounds of both rectangles.
        x1_min, y1_min, x1_max, y1_max = self.get_bounds_at(pos=self.current_position)
        x2_min, y2_min, x2_max, y2_max = rect.get_bounds_at(pos=rect.current_position)

        # Compute the distance
        dist = get_distance_rectangle_rectangle(
            x1_min=x1_min,
            y1_min=y1_min,
            x1_max=x1_max,
            y1_max=y1_max,
            x2_min=x2_min,
            y2_min=y2_min,
            x2_max=x2_max,
            y2_max=y2_max,
        )

        return dist

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

        # Extract the rectangle bounds and the circle geometry
        x_min, y_min, x_max, y_max = self.get_bounds_at(pos=pos_self)
        cx, cy = pos_other
        radius = circle.radius

        # Compute the distance
        dist = get_distance_rectangle_circle(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            cx=cx,
            cy=cy,
            radius=radius,
        )

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

        # Extract the bounds of both rectangles.
        x1_min, y1_min, x1_max, y1_max = self.get_bounds_at(pos_self)
        x2_min, y2_min, x2_max, y2_max = rect.get_bounds_at(pos_other)

        # Compute the distance
        dist = get_distance_rectangle_rectangle(
            x1_min=x1_min,
            y1_min=y1_min,
            x1_max=x1_max,
            y1_max=y1_max,
            x2_min=x2_min,
            y2_min=y2_min,
            x2_max=x2_max,
            y2_max=y2_max,
        )

        return bool(dist <= min_dist)
