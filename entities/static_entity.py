import numpy as np
from entities.entity import Entity
from matplotlib.patches import Rectangle


class StaticEntity(Entity):
    """
    Represents a static entity in the environment.

    Static entities are immobile rectangular objects such as:
    walls, shelves, racks and fixed obstacles.

    Attributes
    ----------
    num : int
        Unique identifier of the object.
    current_position : np.ndarray
        Current center position of the object.
    width : float
        Width of the rectangular obstacle.
    height : float
        Height of the rectangular obstacle.
    """

    def __init__(self, width: float, height: float, num: int = None):
        """
        Builder
        """
        super().__init__(num)
        self.width = width
        self.height = height

    def __str__(self):
        return f"StaticEntity_{self.num}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the axis-aligned bounding box of the rectangular entity.
        """

        x, y = self.current_position

        return (
            x - self.width / 2,
            y - self.height / 2,
            x + self.width / 2,
            y + self.height / 2,
        )

    def bounds_at(self, position):
        """
        Return the axis-aligned bounding box of the rectangular entity at a given position.
        """
        x, y = position
        return (
            x - self.width / 2,
            y - self.height / 2,
            x + self.width / 2,
            y + self.height / 2,
        )

    def collides_with(
        self,
        other: Entity,
        new_pos: np.array,
        min_dist: float,
    ):
        """
        Check collision between this static entity and another entity.

        This method dispatches the collision logic to the appropriate
        shape-specific implementation of the other entity.
        """
        return other._collide_rectangle(self, other.current_position, new_pos, min_dist)

    def _collide_circle(
        self,
        circle: Entity,
        pos_self: np.array,
        pos_other: np.array,
        min_dist: float,
    ):
        """
        Check collision between this static entity and a circular entity.
        """
        x_min, y_min, x_max, y_max = self.bounds_at(pos_self)
        x, y = pos_other
        closest_x = np.clip(x, x_min, x_max)
        closest_y = np.clip(y, y_min, y_max)
        dist = np.linalg.norm([x - closest_x, y - closest_y])
        return dist < circle.radius + min_dist

    def _collide_rectangle(
        self,
        rect: StaticEntity,
        pos_self: np.array,
        pos_other: np.array,
        min_dist: float,
    ):
        """
        Check collision between this static entity and a rectangular entity.
        """
        x1_min, y1_min, x1_max, y1_max = self.bounds_at(pos_self)
        x2_min, y2_min, x2_max, y2_max = rect.bounds_at(pos_other)
        return not (
            x1_max + min_dist < x2_min
            or x1_min - min_dist > x2_max
            or y1_max + min_dist < y2_min
            or y1_min - min_dist > y2_max
        )

    def render(self, ax, color="black"):
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
