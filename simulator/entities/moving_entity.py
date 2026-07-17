from __future__ import annotations

import numpy as np
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
    Base class for circular entities that can move within the environment.
    """

    def __init__(self, num: int | None = None):
        """
        Constructor
        """

        super().__init__(num=num)

        self.num: int = num

        self.current_position: tuple[float, float] | None = None
        self.start_position: tuple[float, float] | None = None
        self.target_positions: list[tuple[float, float]] = []

        self.theta: float = 0
        self.v: float = 0
        self.omega: float = 0
        self.radius: float = 0.5

        self.path: list[tuple[int, int] | tuple[float, float]] = []
        self.path_index: int = 0

    def step(self) -> None:
        """
        Advance the entity to the next position along its predefined path.
        """

        # If no predefinded path, do not move
        # Else, advance to the next position along the predefined path
        if self.path:
            if self.path_index < len(self.path) - 1:
                self.path_index += 1

            self.current_position = self.path[self.path_index]

    def render(
        self,
        ax,
        color: str | tuple[float, float, float, float] = "black",
    ) -> None:
        """
        Default render method for moving entities.
        """

        assert self.current_position is not None

        # Draw the moving entity current position
        x, y = self.current_position
        circle = Circle(
            (x, y),
            self.radius,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
        )
        ax.add_patch(circle)

    @property
    def id(self) -> str:
        """
        Return the unique identifier of the moving entity.
        """

        return f"moving_obstacle_{self.num}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the entity at its current position.
        """

        assert self.current_position is not None

        return self.bounds_at(pos=self.current_position)

    def bounds_at(self, pos: tuple[float, float]) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the entity at a given position.
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
        Return the relative position of the entity with respect to a given point.
        """

        assert self.current_position is not None

        return get_relative_position(pos1=pos, pos2=self.current_position)

    def get_relative_distance(self, pos: tuple[float, float]) -> float:
        """
        Return the Euclidean distance between the entity and a given point.
        """

        assert self.current_position is not None

        return get_relative_distance(pos1=self.current_position, pos2=pos)

    def get_distance(self, other: StaticEntity | MovingEntity) -> float:
        """
        Compute the distance between the entity and another entity.
        """

        return other._get_distance_circle(circle=self)

    def collides_with(
        self,
        other: MovingEntity | StaticEntity,
        new_pos: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check whether the entity would collide with another entity at the given
        positions.
        """

        assert other.current_position

        collides_with = bool(
            other._collide_circle(
                circle=self,
                pos_self=other.current_position,
                pos_other=new_pos,
                min_dist=min_dist,
            )
        )

        return collides_with

    def _get_distance_circle(self, circle: MovingEntity) -> float:
        """
        Compute the distance between two circular entities.
        """

        assert self.current_position is not None
        assert circle.current_position is not None

        # Extract the geometry of both circular entities
        cx1, cy1 = self.current_position
        radius1 = self.radius
        cx2, cy2 = circle.current_position
        radius2 = circle.radius

        # Compute the distance
        dist = get_distance_circle_circle(
            cx1=cx1, cy1=cy1, radius1=radius1, cx2=cx2, cy2=cy2, radius2=radius2
        )

        return dist

    def _get_distance_rectangle(self, rect: StaticEntity) -> float:
        """
        Compute the distance between the entity and a rectangular obstacle.
        """

        assert self.current_position is not None
        assert rect.current_position is not None

        # Extract the rectangle bounds and the circle geometry
        x_min, y_min, x_max, y_max = rect.get_bounds_at(pos=rect.current_position)
        cx, cy = self.current_position
        radius = self.radius

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

    def _collide_circle(
        self,
        circle: MovingEntity,
        pos_self: tuple[float, float],
        pos_other: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check whether two circular entities collide at the given positions.
        """

        # Extract the geometry of both circular entities
        cx1, cy1 = pos_self
        radius1 = self.radius
        cx2, cy2 = pos_other
        radius2 = circle.radius

        # Compute the distance
        dist = get_distance_circle_circle(
            cx1=cx1, cy1=cy1, radius1=radius1, cx2=cx2, cy2=cy2, radius2=radius2
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
        Check whether the entity collides with a rectangular obstacle at the given
        positions.
        """

        # Extract the rectangle bounds and the circle geometry
        x_min, y_min, x_max, y_max = rect.get_bounds_at(pos=pos_other)
        cx, cy = pos_self
        radius = self.radius

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

    def _get_next_pos(self, v: float, dt: float = 1) -> tuple[float, float]:
        """
        Compute the next position of the entity from its current state.
        """

        assert self.current_position is not None

        # Update the position according to the current heading.
        x, y = self.current_position
        x += v * np.cos(self.theta) * dt
        y += v * np.sin(self.theta) * dt

        return x, y
