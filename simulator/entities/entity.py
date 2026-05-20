from abc import ABC, abstractmethod


class Entity(ABC):
    """
    Base class for every object in the environment
    (robot, static obstacle, dynamic obstacle, etc.).

    Attributes
    ----------
    num : int
        Unique identifier of the object.
    current_position : tuple
        Current center position of the object.
    """

    def __init__(self, num: int | None = None):
        """
        Constructor
        """

        super().__init__()
        self.num = num
        self.current_position: tuple[float, float] | None = None

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
        pos_self: tuple[float, float],
        pos_other: tuple[float, float],
        min_dist: float = 0.0,
    ) -> bool:
        """
        Check collision between this entity and another entity.

        This method dispatches the collision logic to the appropriate
        shape-specific implementation of the other entity.
        """
        pass

    @abstractmethod
    def render(self, ax) -> None:
        """
        Default render method.
        """
        pass
