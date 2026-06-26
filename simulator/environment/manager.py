import numpy as np

from configs.config import AgentConfig, EnvConfig
from simulator.entities import agent, moving_entity, static_entity


class Manager:
    """
    Environment entity generation manager.

    This class is responsible for generating and initializing all
    entities present in the simulation environment, including:
    - agents
    - static obstacles
    - moving obstacles

    The manager ensures that generated entities respect spatial
    constraints and minimum safety distances.

    Attributes
    ----------
    env_config : EnvConfig
        Environment configuration parameters.
    agent_config : AgentConfig
        Agent configuration parameters.
    width : int
        Width of the environment.
    height : int
        Height of the environment.
    nb_agents : int
        Number of agents to generate.
    nb_static_obstacles : int
        Number of static obstacles to generate.
    nb_moving_obstacles : int
        Number of moving obstacles to generate.
    agents : list[Agent]
        Generated agents.
    static_obstacles : list[StaticEntity]
        Generated static obstacles.
    moving_obstacles : list[MovingEntity]
        Generated moving obstacles.
    """

    def __init__(self, env_config: EnvConfig, agent_config: AgentConfig):
        """
        Constructor
        """

        self.env_config = env_config
        self.agent_config = agent_config

        self.width: int = env_config.width
        self.height: int = env_config.height

        self.nb_agents: int = env_config.nb_agents
        self.nb_static_obstacles: int = env_config.nb_static_obstacles
        self.nb_moving_obstacles: int = env_config.nb_moving_obstacles

        self.agents: list[agent.Agent] = []
        self.static_obstacles: list[static_entity.StaticEntity] = []
        self.moving_obstacles: list[moving_entity.MovingEntity] = []

    # ---------------------------------------------------------------
    # GLOBAL GENERATE
    # ---------------------------------------------------------------

    def generate(
        self,
    ) -> tuple[
        list[static_entity.StaticEntity],
        list[moving_entity.MovingEntity],
        list[agent.Agent],
    ]:
        """"""

        static_obstacles = self.generate_static_obstacles()
        moving_obstacles = self.generate_moving_obstacles()
        agents = self.generate_agents()

        return static_obstacles, moving_obstacles, agents

    def reset(self) -> tuple[list[moving_entity.MovingEntity], list[agent.Agent]]:
        """"""

        self.moving_obstacles = []
        self.agents = []
        moving_obstacles = self.generate_moving_obstacles()
        agents = self.generate_agents()
        return moving_obstacles, agents

    def generate_static_obstacles(self) -> list[static_entity.StaticEntity]:
        """
        Generate static obstacles in the environment.

        Walls are automatically added before generating
        additional obstacles.
        """

        self.static_obstacles = []

        pad = self.env_config.thickness
        width_min = self.env_config.width_min
        width_max = self.env_config.width_max
        height_min = self.env_config.height_min
        height_max = self.env_config.height_max
        min_dist = self.env_config.margin
        max_attempts = self.env_config.max_attempts

        self._add_walls(pad)
        mode = self.env_config.env_mode

        if mode == "random":
            self._random_static_obstacles(
                pad,
                width_min,
                width_max,
                height_min,
                height_max,
                min_dist,
                max_attempts,
            )

        if mode == "fixed":
            self._fixed_static_obstacles()

        if mode == "fixed_2":
            self._fixed_2_static_obstacles()

        if mode == "fixed_3":
            self._fixed_3_static_obstacles()

        if mode == "fixed_random":
            self._fixed_random_static_obstacles()

        if mode == "warehouse":
            self._warehouse_static_obstacles()

        if mode == "crossing":
            self._crossing_static_obstacles()

        return self.static_obstacles

    def generate_moving_obstacles(self) -> list[moving_entity.MovingEntity]:
        """
        Generate moving obstacles in the environment.
        """

        nb_targets = self.env_config.nb_targets
        radius_min = self.env_config.radius_min
        radius_max = self.env_config.radius_max
        min_dist = self.env_config.margin
        max_attempts = self.env_config.max_attempts
        mode = self.env_config.env_mode

        if mode == "random" or mode == "warehouse":
            self._random_moving_obstacles(
                nb_targets, radius_min, radius_max, min_dist, max_attempts
            )

        if mode == "crossing":
            self._crossing_moving_obstacles(min_dist, max_attempts)

        return self.moving_obstacles

    def generate_agents(self) -> list[agent.Agent]:
        """
        Generate agents in the environment.
        """

        nb_targets = self.env_config.nb_targets
        min_dist = self.env_config.margin
        max_attempts = self.env_config.max_attempts
        mode = self.env_config.agent_mode

        if mode == "fixed":
            self._fixed_agents(nb_targets, min_dist)

        if mode == "fixed_2":
            self._fixed_2_agents(nb_targets, min_dist)

        if mode == "fixed_3":
            self._fixed_3_agents(nb_targets, min_dist)

        if mode == "fixed_random":
            self._fixed_random_agents(nb_targets, min_dist)

        if mode == "random":
            self._random_agents(nb_targets, min_dist, max_attempts)

        if mode == "crossing":
            self._crossing_agents(min_dist, max_attempts)

        return self.agents

    # ---------------------------------------------------------------
    # GENERATE STATIC OBSTACLES
    # ---------------------------------------------------------------

    def _add_walls(self, pad: int):
        """
        Generate the environment border walls.
        """

        top_wall = static_entity.StaticEntity(width=self.width, height=pad, num=1)
        top_wall.current_position = (
            self.width / 2,
            self.height,
        )
        right_wall = static_entity.StaticEntity(width=pad, height=self.height, num=2)
        right_wall.current_position = (
            self.width,
            self.height / 2,
        )
        bot_wall = static_entity.StaticEntity(width=self.width, height=pad, num=3)
        bot_wall.current_position = (self.width / 2, 0)
        left_wall = static_entity.StaticEntity(width=pad, height=self.height, num=4)
        left_wall.current_position = (0, self.height / 2)

        self.static_obstacles.extend([top_wall, right_wall, bot_wall, left_wall])

    def _fixed_static_obstacles(self):
        """ """

        n = self.env_config.nb_static_obstacles

        cx = self.env_config.width // 2

        w = self.env_config.width // 16
        h = self.env_config.height // 16

        y_positions = np.linspace(
            self.env_config.height // (n + 1),
            self.env_config.height * n // (n + 1),
            n,
        )

        for y in y_positions:
            obstacle = static_entity.StaticEntity(
                width=w,
                height=h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (cx, int(y))
            self.static_obstacles.append(obstacle)

    def _fixed_2_static_obstacles(self):
        """ """

        n = self.env_config.nb_static_obstacles

        cx = self.env_config.width // 3

        w = self.env_config.width // 16
        h = self.env_config.height // 16

        y_positions = np.linspace(
            self.env_config.height // (n + 1),
            self.env_config.height * n // (n + 1),
            n,
        )

        for y in y_positions:

            obstacle = static_entity.StaticEntity(
                width=w,
                height=h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (cx, int(y))
            self.static_obstacles.append(obstacle)

            obstacle = static_entity.StaticEntity(
                width=w,
                height=h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (2 * cx, int(y))
            self.static_obstacles.append(obstacle)

    def _fixed_3_static_obstacles(self):
        """ """

        n = self.env_config.nb_static_obstacles

        cx = self.env_config.width // 3

        w = self.env_config.width // 8
        h = self.env_config.height // 16

        y_positions = np.linspace(
            self.env_config.height // (n + 1),
            self.env_config.height * n // (n + 1),
            n,
        )

        for y in y_positions:

            obstacle = static_entity.StaticEntity(
                width=w,
                height=h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (cx, int(y))
            self.static_obstacles.append(obstacle)

            obstacle = static_entity.StaticEntity(
                width=w,
                height=h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (2 * cx, int(y))
            self.static_obstacles.append(obstacle)

    def _fixed_random_static_obstacles(self):
        """ """

        n = self.env_config.nb_static_obstacles

        cx = self.env_config.width // 3
        y_positions = np.linspace(
            self.env_config.height // (n + 1),
            self.env_config.height * n // (n + 1),
            n,
        )

        for cy in y_positions:

            w = np.random.randint(1, 8)
            h = np.random.randint(1, 5)
            x = np.random.randint(-3, 3)
            y = np.random.randint(-2, 2)
            obstacle = static_entity.StaticEntity(
                width=w,
                height=h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (cx + x, int(cy + y))
            self.static_obstacles.append(obstacle)

            w = np.random.randint(1, 8)
            h = np.random.randint(1, 5)
            x = np.random.randint(-3, 3)
            y = np.random.randint(-2, 2)
            obstacle = static_entity.StaticEntity(
                width=w,
                height=h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (2 * cx + x, int(cy + y))
            self.static_obstacles.append(obstacle)

    def _random_static_obstacles(
        self,
        pad: int,
        width_min: int,
        width_max: int,
        height_min: int,
        height_max: int,
        min_dist: float,
        max_attempts: int,
    ):
        """
        Randomly generate static rectangular obstacles.

        Obstacles are sampled with random dimensions and
        positions while ensuring collision-free placement.
        """

        for _ in range(self.nb_static_obstacles):
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                w = np.random.randint(width_min, width_max - 1)
                h = np.random.randint(height_min, height_max - 1)
                w = w if w % 2 == 1 else w + 1
                h = h if h % 2 == 1 else h + 1
                obstacle = static_entity.StaticEntity(
                    width=w, height=h, num=len(self.static_obstacles) + 1
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

    def _warehouse_static_obstacles(self):
        """
        Generate a predefined static obstacle configuration.
        """

        n_cols = self.width // 10

        w = 2
        h = self.height // 2 - 12

        x = [self.width * i for i in np.linspace(1 / n_cols, 1 - 1 / n_cols, n_cols)]
        y = [self.height * i for i in [0.25, 0.75]]

        for i in x:
            for j in y:
                obstacle = static_entity.StaticEntity(
                    width=w,
                    height=h,
                    num=len(self.static_obstacles) + 1,
                )
                obstacle.current_position = (i, j)
                self.static_obstacles.append(obstacle)

    def _crossing_static_obstacles(self):
        """ """

        n = self.nb_static_obstacles
        cx, cy = self.width / 2, self.height / 2
        r = min(self.width, self.height) // 4

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        positions = []
        for a in angles:
            x = cx + r * np.cos(a)
            y = cy + r * np.sin(a)
            positions.append((x, y))

        for pos in positions:
            obstacle = static_entity.StaticEntity(
                width=2,
                height=2,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = pos
            self.static_obstacles.append(obstacle)

    # ---------------------------------------------------------------
    # GENERATE MOVING OBSTACLES
    # ---------------------------------------------------------------

    def _random_moving_obstacles(
        self,
        nb_targets: int,
        radius_min: float,
        radius_max: float,
        min_dist: float,
        max_attempts: int,
    ):
        """
        Randomly generate moving obstacles.

        Each obstacle receives:
        - a random radius
        - a collision-free initial position
        - multiple target positions
        """

        for i in range(self.nb_moving_obstacles):
            moving_obstacle = moving_entity.MovingEntity(num=i + 1)
            pos = self._set_random_position(
                moving_obstacle, min_dist, max_attempts, False
            )
            moving_obstacle.start_position = pos
            moving_obstacle.current_position = pos
            moving_obstacle.old_position = pos
            moving_obstacle.radius = np.random.uniform(radius_min, radius_max)

            for _ in range(nb_targets):
                moving_obstacle.target_positions.append(
                    self._set_random_position(
                        moving_obstacle, min_dist, max_attempts, True
                    )
                )

            self.moving_obstacles.append(moving_obstacle)

    def _crossing_moving_obstacles(self, min_dist: float, max_attempts: int):
        """"""

        for i in range(self.nb_moving_obstacles):
            moving_obstacle = moving_entity.MovingEntity(num=i + 1)
            pos, target = self._set_circular_position(
                moving_obstacle, min_dist, max_attempts
            )
            moving_obstacle.start_position = pos
            moving_obstacle.current_position = pos
            moving_obstacle.old_position = pos
            moving_obstacle.radius = self.agent_config.radius
            moving_obstacle.target_positions.append(target)

            self.moving_obstacles.append(moving_obstacle)

    # ---------------------------------------------------------------
    # GENERATE AGENTS
    # ---------------------------------------------------------------

    def _fixed_agents(self, nb_targets: int, min_dist: float):
        """"""

        pad = int(self.env_config.thickness + min_dist)

        for i in range(self.nb_agents):
            a = agent.Agent(self.agent_config, i + 1)

            cx = self.width // 2
            w = self.width // 16

            pos = (
                np.random.randint(pad, cx - w - pad),
                np.random.randint(pad, self.height - pad),
            )
            a.start_position = pos
            a.current_position = pos
            a.old_position = pos

            for _ in range(nb_targets):
                pos = (
                    np.random.randint(cx + w + pad, self.width - pad),
                    np.random.randint(pad, self.height - pad),
                )
                a.target_positions.append(pos)

            self.agents.append(a)

    def _fixed_2_agents(self, nb_targets: int, min_dist: float):
        """"""

        pad = int(self.env_config.thickness + min_dist)

        for i in range(self.nb_agents):
            a = agent.Agent(self.agent_config, i + 1)

            cx = self.width // 3
            w = self.width // 16

            pos = (
                np.random.randint(pad, cx - w - pad),
                np.random.randint(pad, self.height - pad),
            )
            a.start_position = pos
            a.current_position = pos
            a.old_position = pos

            for j in range(nb_targets):
                if j % 2 == 1:
                    pos = (
                        np.random.randint(2 * cx + w + pad, self.width - pad),
                        np.random.randint(pad, self.height - pad),
                    )
                else:
                    pos = (
                        np.random.randint(cx + w + pad, 2 * cx - w - pad),
                        np.random.randint(pad, self.height - pad),
                    )
                a.target_positions.append(pos)

            self.agents.append(a)

    def _fixed_3_agents(self, nb_targets: int, min_dist: float):
        """"""

        pad = int(self.env_config.thickness + min_dist)

        for i in range(self.nb_agents):
            a = agent.Agent(self.agent_config, i + 1)

            cx = self.width // 3
            w = self.width // 10

            pos = (
                np.random.randint(pad, cx - w - pad),
                np.random.randint(pad, self.height - pad),
            )
            a.start_position = pos
            a.current_position = pos
            a.old_position = pos

            for j in range(nb_targets):
                if j % 2 == 1:
                    pos = (
                        np.random.randint(2 * cx + w + pad, self.width - pad),
                        np.random.randint(pad, self.height - pad),
                    )
                else:
                    pos = (
                        np.random.randint(cx + w + pad, 2 * cx - w - pad),
                        np.random.randint(pad, self.height - pad),
                    )
                a.target_positions.append(pos)

            self.agents.append(a)

    def _fixed_random_agents(self, nb_targets: int, min_dist: float):
        """ """

        pad = int(self.env_config.thickness + min_dist)

        for i in range(self.nb_agents):

            a = agent.Agent(self.agent_config, i + 1)
            cx = self.width // 3

            pos = (
                np.random.randint(pad, cx - pad),
                np.random.randint(pad, self.height - pad),
            )
            a.start_position = pos
            a.current_position = pos
            a.old_position = pos

            for j in range(nb_targets):
                if j % 2 == 1:
                    pos = (
                        np.random.randint(2 * cx + pad, self.width - pad),
                        np.random.randint(pad, self.height - pad),
                    )
                else:
                    pos = (
                        np.random.randint(cx + pad, 2 * cx - pad),
                        np.random.randint(pad, self.height - pad),
                    )
                a.target_positions.append(pos)

            self.agents.append(a)

    def _random_agents(self, nb_targets: int, min_dist: float, max_attempts: int):
        """
        Randomly generate agents.

        Each agent receives:
        - a collision-free initial position
        - multiple target positions
        """

        for i in range(self.nb_agents):
            a = agent.Agent(self.agent_config, i + 1)
            pos = self._set_random_position(a, min_dist, max_attempts, False)
            a.start_position = pos
            a.current_position = pos
            a.old_position = pos
            self.agents.append(a)
            for _ in range(nb_targets):
                a.target_positions.append(
                    self._set_random_position(a, min_dist, max_attempts, True)
                )

    def _crossing_agents(self, min_dist: float, max_attempts: int):
        """"""

        for i in range(self.nb_agents):
            a = agent.Agent(self.agent_config, i + 1)
            pos, target = self._set_circular_position(a, min_dist, max_attempts)
            a.start_position = pos
            a.current_position = pos
            a.old_position = pos
            a.radius = self.agent_config.radius
            a.target_positions.append(target)

            self.agents.append(a)

    # ---------------------------------------------------------------
    # SET POSITIONS
    # ---------------------------------------------------------------

    def _set_random_position(
        self,
        entity: static_entity.StaticEntity | moving_entity.MovingEntity | agent.Agent,
        min_dist: float,
        max_attempts: int,
        for_target=False,
    ) -> tuple[int, int]:
        """
        Randomly generate positions.
        """

        pad = self.env_config.thickness
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

    def _set_circular_position(
        self,
        entity: static_entity.StaticEntity | moving_entity.MovingEntity | agent.Agent,
        min_dist: float,
        max_attempts: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """ """

        n = self.nb_agents + self.nb_moving_obstacles
        cx, cy = self.width / 2, self.height / 2
        r = min(self.width, self.height) // 2 - 5
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

        is_free = False
        i = 0
        while not is_free and i < max_attempts:
            k = np.random.randint(len(positions))
            pos = positions[k]
            is_free = self._is_free(entity, pos, min_dist, False)
            i += 1

        opposite_angle = angles[k] + np.pi
        target = (
            cx + r * np.cos(opposite_angle),
            cy + r * np.sin(opposite_angle),
        )

        return pos, target

    def _is_free(
        self,
        new: static_entity.StaticEntity | moving_entity.MovingEntity | agent.Agent,
        pos: tuple[int, int],
        min_dist: float,
        for_target: bool,
    ) -> bool:
        """
        Check whether an entity position is collision-free.
        """

        if for_target:
            entities = self.static_obstacles
        else:
            entities = self.agents + self.static_obstacles + self.moving_obstacles

        for entity in entities:
            if entity is new:
                continue
            if entity.current_position is None:
                continue
            if new.collides_with(entity, pos, min_dist):
                return False
        return True
