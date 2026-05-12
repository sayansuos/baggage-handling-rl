import numpy as np
from entities.agent import Agent
from entities.entity import Entity
from entities.static_entity import StaticEntity
from entities.moving_entity import MovingEntity


class EnvironmentManager:
    """
    Responsible for generating and initializing all entities
    in the environment (agents, static obstacles, moving obstacles).

    This class handles:
    - random environment generation
    - entity placement
    - collision-free spawning
    - target generation

    Attributes
    ----------
    env_width : float
        Width of the environment.
    env_height : float
        Height of the environment.
    nb_agents : int
        Number of agents to generate.
    nb_static_obstacles : int
        Number of static obstacles to generate.
    nb_moving_obstacles : int
        Number of moving obstacles to generate.
    agents : list[Agent]
        List of generated agents.
    static_obstacles : list[StaticEntity]
        List of generated static obstacles.
    moving_obstacles : list[MovingEntity]
        List of generated moving obstacles.
    """

    WIDTH_MIN, WIDTH_MAX = 5, 20
    HEIGHT_MIN, HEIGHT_MAX = 5, 20
    THICKNESS = 1
    MARGIN = 2 * Agent.RADIUS
    MAX_ATTEMPTS = 100

    def __init__(
        self,
        env_width: float,
        env_height: float,
        nb_agents: int,
        nb_static_obstacles: int,
        nb_moving_obstacles: int,
    ):
        """
        Builder
        """

        self.env_width = env_width
        self.env_height = env_height

        self.nb_agents = nb_agents
        self.nb_static_obstacles = nb_static_obstacles
        self.nb_moving_obstacles = nb_moving_obstacles

        self.agents = []
        self.static_obstacles = []
        self.moving_obstacles = []

    @property
    def entities(self):
        """
        Return all entities currently present in the environment.
        """

        return self.agents + self.static_obstacles + self.moving_obstacles

    def generate_static_obstacles(
        self,
        min_dist: int = MARGIN,
        mode: str = "random",
    ):
        """
        Generate static obstacles in the environment.

        Walls are automatically added before generating
        additional obstacles.
        """

        self._add_walls()

        if mode == "random":
            self._random_static_obstacles(min_dist)

        if mode == "setup1":
            self._setup1_static_obstacles()

        return self.static_obstacles

    def generate_moving_obstacles(
        self,
        nb_targets: int = 2,
        radius_min: float = Agent.RADIUS,
        radius_max: float = 2 * Agent.RADIUS,
        min_dist: float = MARGIN,
        mode: str = "random",
    ):
        """
        Generate moving obstacles in the environment.
        """

        if mode == "random":
            self._random_moving_obstacles(nb_targets, radius_min, radius_max, min_dist)

        return self.moving_obstacles

    def generate_agents(
        self,
        nb_targets: int = 2,
        min_dist: float = MARGIN,
        mode: str = "random",
    ):
        """
        Generate agents in the environment.
        """

        if mode == "random":
            self._random_agents(nb_targets, min_dist)

        return self.agents

    def _add_walls(self):
        """
        Generate the environment border walls.
        """

        top_wall = StaticEntity(
            width=self.env_width, height=self.THICKNESS, num=len(self.static_obstacles)
        )
        top_wall.current_position = (
            self.env_width / 2,
            self.env_height,
        )

        right_wall = StaticEntity(
            width=self.THICKNESS, height=self.env_height, num=len(self.static_obstacles)
        )
        right_wall.current_position = (
            self.env_width,
            self.env_height / 2,
        )

        bot_wall = StaticEntity(
            width=self.env_width, height=self.THICKNESS, num=len(self.static_obstacles)
        )
        bot_wall.current_position = (self.env_width / 2, 0)

        left_wall = StaticEntity(
            width=self.THICKNESS, height=self.env_height, num=len(self.static_obstacles)
        )
        left_wall.current_position = (0, self.env_height / 2)

        self.static_obstacles.extend([top_wall, right_wall, bot_wall, left_wall])

    def _random_static_obstacles(self, min_dist: float):
        """
        Randomly generate static rectangular obstacles.

        Obstacles are sampled with random dimensions and
        positions while ensuring collision-free placement.
        """
        pad = self.THICKNESS

        for _ in range(self.nb_static_obstacles):

            placed = False
            attempts = 0

            while not placed and attempts < self.MAX_ATTEMPTS:
                w = np.random.uniform(self.WIDTH_MIN, self.WIDTH_MAX)
                h = np.random.uniform(self.HEIGHT_MIN, self.HEIGHT_MAX)
                obstacle = StaticEntity(
                    width=w, height=h, num=len(self.static_obstacles)
                )
                pos = (
                    np.random.uniform(pad, self.env_width - pad),
                    np.random.uniform(pad, self.env_height - pad),
                )

                obstacle.current_position = pos

                if self._is_free(obstacle, pos, min_dist, False):
                    self.static_obstacles.append(obstacle)
                    placed = True

                attempts += 1

    def _setup1_static_obstacles(self):
        """
        Generate a predefined static obstacle configuration.
        """
        pass

    def _random_moving_obstacles(
        self, nb_targets: int, radius_min: int, radius_max: int, min_dist: int
    ):
        """
        Randomly generate moving obstacles.

        Each obstacle receives:
        - a random radius
        - a collision-free initial position
        - multiple target positions
        """

        rdm_radius = np.random.uniform(radius_min, radius_max, self.nb_moving_obstacles)

        for i in range(self.nb_moving_obstacles):
            moving_obstacle = MovingEntity(radius=rdm_radius[i], num=i + 1)
            pos = self._set_random_position(moving_obstacle, min_dist, False)
            moving_obstacle.start_position = pos
            moving_obstacle.current_position = pos

            for _ in range(nb_targets):
                moving_obstacle.target_positions.append(
                    self._set_random_position(moving_obstacle, min_dist, True)
                )

            self.moving_obstacles.append(moving_obstacle)

    def _random_agents(self, nb_targets: int, min_dist: float):
        """
        Randomly generate agents.

        Each agent receives:
        - a collision-free initial position
        - multiple target positions
        """

        for i in range(self.nb_agents):
            agent = Agent(i + 1)
            pos = self._set_random_position(agent, False)
            agent.start_position = pos
            agent.current_position = pos
            self.agents.append(agent)
            for _ in range(nb_targets):
                agent.target_positions.append(
                    self._set_random_position(agent, min_dist, True)
                )

    def _set_random_position(self, entity: Entity, min_dist: float, for_target=False):
        """
        Randomly generate agents.

        Each agent receives:
        - a collision-free initial position
        - multiple target positions
        """
        is_free = False
        i = 0
        while not is_free and i < self.MAX_ATTEMPTS:
            pos = (
                np.random.randint(self.THICKNESS, self.env_width - self.THICKNESS),
                np.random.randint(self.THICKNESS, self.env_height - self.THICKNESS),
            )
            is_free = self._is_free(entity, pos, min_dist, for_target)
            i += 1
        return pos

    def _is_free(self, new: Entity, pos: np.array, min_dist: float, for_target: bool):
        """
        Check whether an entity position is collision-free.
        """
        if for_target:
            entities = self.static_obstacles
        else:
            entities = self.entities

        for entity in entities:
            if entity is new:
                continue
            if entity.current_position is None:
                continue
            if new.collides_with(entity, pos, min_dist):
                return False

        return True
