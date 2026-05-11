import numpy as np
from abc import ABC, abstractmethod


class Entity(ABC):
    """
    Base class for every object in the environment
    (robot, static obstacle, dynamic obstacle, etc.).

    Attributes
    ----------
    num : int
        Unique identifier of the object.
    current_position : np.ndarray
        Current center position of the object.
    """

    def __init__(self, num: int = None):
        """
        Builder
        """

        super().__init__()
        self.num = num
        self.current_position = None

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the object's bounds.
        """
        pass

    @abstractmethod
    def collides_with(
        self,
        other: Entity,
        pos_self: np.array,
        pos_other: np.array,
        margin: float = 0.0,
    ):
        """
        Check collision between this entity and another entity.

        This method dispatches the collision logic to the appropriate
        shape-specific implementation of the other entity.
        """
        pass

    def render(self, ax):
        """
        Default render method.
        """
        pass
