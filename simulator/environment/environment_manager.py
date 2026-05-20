import numpy as np
from simulator.utils.config import EnvConfig, AgentConfig
from simulator.entities.agent import Agent
from simulator.entities.entity import Entity
from simulator.entities.static_entity import StaticEntity
from simulator.entities.moving_entity import MovingEntity


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

    # WIDTH_MIN, WIDTH_MAX = 5, 20
    # HEIGHT_MIN, HEIGHT_MAX = 5, 20
    # THICKNESS = 1
    # MARGIN = 2 * Agent.RADIUS
    # MAX_ATTEMPTS = 100

    def __init__(self, env_config: EnvConfig, agent_config: AgentConfig):
        """
        Builder
        """
        self.env_config = env_config
        self.agent_config = agent_config

        self.width = env_config.width
        self.height = env_config.height

        self.nb_agents = env_config.nb_agents
        self.nb_static_obstacles = env_config.nb_static_obstacles
        self.nb_moving_obstacles = env_config.nb_moving_obstacles

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
        min_dist: float = None,
        mode: str = "random",
    ):
        """
        Generate static obstacles in the environment.

        Walls are automatically added before generating
        additional obstacles.
        """

        self._add_walls()
        if not min_dist:
            min_dist = self.env_config.margin
        if mode == "random":
            self._random_static_obstacles(min_dist)
        if mode == "setup1":
            self._setup1_static_obstacles()
        return self.static_obstacles

    def generate_moving_obstacles(
        self,
        nb_targets: int = 2,
        radius_min: float = None,
        radius_max: float = None,
        min_dist: float = None,
        mode: str = "random",
    ):
        """
        Generate moving obstacles in the environment.
        """

        if not radius_min:
            radius_min = self.env_config.radius_min
        if not radius_max:
            radius_max = self.env_config.radius_max
        if not min_dist:
            min_dist = self.env_config.margin

        if mode == "random":
            self._random_moving_obstacles(nb_targets, radius_min, radius_max, min_dist)

        return self.moving_obstacles

    def generate_agents(
        self,
        nb_targets: int = None,
        min_dist: float = None,
        mode: str = "random",
    ):
        """
        Generate agents in the environment.
        """

        if not nb_targets:
            nb_targets = self.env_config.nb_targets
        if not min_dist:
            min_dist = self.env_config.margin

        if mode == "random":
            self._random_agents(nb_targets, min_dist)

        return self.agents

    def _add_walls(self, thickness: int = None):
        """
        Generate the environment border walls.
        """
        if not thickness:
            thickness = self.env_config.thickness

        top_wall = StaticEntity(
            width=self.width, height=thickness, num=len(self.static_obstacles)
        )
        top_wall.current_position = (
            self.width / 2,
            self.height,
        )

        right_wall = StaticEntity(
            width=thickness, height=self.height, num=len(self.static_obstacles)
        )
        right_wall.current_position = (
            self.width,
            self.height / 2,
        )

        bot_wall = StaticEntity(
            width=self.width, height=thickness, num=len(self.static_obstacles)
        )
        bot_wall.current_position = (self.width / 2, 0)

        left_wall = StaticEntity(
            width=thickness, height=self.height, num=len(self.static_obstacles)
        )
        left_wall.current_position = (0, self.height / 2)

        self.static_obstacles.extend([top_wall, right_wall, bot_wall, left_wall])

    def _random_static_obstacles(
        self,
        min_dist: float,
        thickness: int = None,
        max_attempts: int = None,
        width_min: int = None,
        width_max: int = None,
        height_min: int = None,
        height_max: int = None,
    ):
        """
        Randomly generate static rectangular obstacles.

        Obstacles are sampled with random dimensions and
        positions while ensuring collision-free placement.
        """

        pad = self.env_config.thickness if not thickness else thickness
        max_attempts = (
            self.env_config.max_attempts if not max_attempts else max_attempts
        )
        width_min = self.env_config.width_min if not width_min else width_min
        width_max = self.env_config.width_max if not width_max else width_max
        height_min = self.env_config.height_min if not height_min else height_min
        height_max = self.env_config.height_max if not height_max else height_max

        for _ in range(self.nb_static_obstacles):

            placed = False
            attempts = 0

            while not placed and attempts < max_attempts:
                w = np.random.randint(width_min, width_max - 1)
                h = np.random.randint(height_min, height_max - 1)
                w = w if w % 2 == 1 else w + 1
                h = h if h % 2 == 1 else h + 1
                obstacle = StaticEntity(
                    width=w, height=h, num=len(self.static_obstacles)
                )
                pos = (
                    np.random.randint(pad, self.width - pad),
                    np.random.randint(pad, self.height - pad),
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

        for i in range(self.nb_moving_obstacles):
            moving_obstacle = MovingEntity(num=i + 1)
            pos = self._set_random_position(moving_obstacle, min_dist, False)
            moving_obstacle.start_position = pos
            moving_obstacle.current_position = pos
            moving_obstacle.radius = np.random.uniform(radius_min, radius_max)

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
            agent = Agent(self.agent_config, i + 1)
            pos = self._set_random_position(agent, False)
            agent.start_position = pos
            agent.current_position = pos
            self.agents.append(agent)
            for _ in range(nb_targets):
                agent.target_positions.append(
                    self._set_random_position(agent, min_dist, True)
                )

    def _set_random_position(
        self,
        entity: Entity,
        min_dist: float,
        for_target=False,
        max_attempts: int = None,
    ):
        """
        Randomly generate positions.
        """
        pad = self.env_config.thickness
        if not max_attempts:
            max_attempts = self.env_config.max_attempts
        is_free = False
        i = 0
        while not is_free and i < max_attempts:
            pos = (
                np.random.randint(pad, self.width - pad),
                np.random.randint(pad, self.height - pad),
            )
            is_free = self._is_free(entity, pos, min_dist, for_target)
            i += 1
        return pos

    def _is_free(self, new: Entity, pos: np.ndarray, min_dist: float, for_target: bool):
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
