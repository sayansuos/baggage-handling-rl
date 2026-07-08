from abc import ABC, abstractmethod


class Entity(ABC):
    """
    Base class for all entities in the environment.
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
        Return the bounding box of the entity.
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
        Check whether the entity collides with another entity.
        """

        pass

    @abstractmethod
    def render(self, ax) -> None:
        """
        Render the entity.
        """

        pass
