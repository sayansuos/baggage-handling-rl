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

        super().__init__(num)

        self.start_position: tuple[float, float] | None = None
        self.target_positions: list[tuple[float, float]] = []

        self.radius: float = 0.5
        self.theta: float = 0
        self.v: float = 0
        self.omega: float = 0

        self.path: list[tuple[int, int] | tuple[float, float]] = []
        self.path_index: int = 0

    def step(self):
        """
        Advance the entity to the next position along its predefined path.
        """

        if self.path:
            if self.path_index < len(self.path) - 1:
                self.path_index += 1

            self.current_position = self.path[self.path_index]

    def render(
        self,
        ax,
        color: str | tuple[float, float, float, float] = "black",
    ):
        """
        Default render method for moving entities.
        """

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

    @property
    def id(self) -> str:
        """
        Advance the entity to the next position along its predefined path.
        """

        return f"moving_obstacle_{self.num}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the bounding box of the entity at its current position.
        """

        assert self.current_position is not None
        return self.bounds_at(self.current_position)

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
        return get_relative_position(pos, self.current_position)

    def get_relative_distance(self, pos: tuple[float, float]) -> float:
        """
        Return the Euclidean distance between the entity and a given point.
        """

        assert self.current_position is not None
        return get_relative_distance(self.current_position, pos)

    def get_distance(self, other: StaticEntity | MovingEntity) -> float:
        """
        Compute the distance between the entity and another entity.
        """

        return other._get_distance_circle(self)

    def collides_with(
        self,
        other: MovingEntity | StaticEntity,
        new_pos: tuple[float, float],
        min_dist: float,
    ) -> bool:
        """
        Check whether the entity would collide with another entity at the given positions.
        """

        assert other.current_position
        return bool(
            other._collide_circle(self, other.current_position, new_pos, min_dist)
        )

    def _get_distance_circle(self, circle: MovingEntity) -> float:
        """
        Compute the distance between two circular entities.
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
        Compute the distance between the entity and a rectangular obstacle.
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
        Check whether two circular entities collide at the given positions.
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
        Check whether the entity collides with a rectangular obstacle at the given positions.
        """

        x_min, y_min, x_max, y_max = rect.get_bounds_at(pos_other)
        cx, cy = pos_self
        radius = self.radius
        dist = get_distance_rectangle_circle(x_min, y_min, x_max, y_max, cx, cy, radius)
        return dist <= min_dist

    def _get_next_pos(self, v: float, dt: float = 1) -> tuple[float, float]:
        """
        Compute the next position of the entity from its current state.
        """

        assert self.current_position is not None
        x, y = self.current_position
        x += v * np.cos(self.theta) * dt
        y += v * np.sin(self.theta) * dt
        return x, y
