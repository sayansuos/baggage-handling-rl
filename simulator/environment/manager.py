import numpy as np

from configs.config import AgentConfig, EnvConfig
from simulator.entities import agent, moving_entity, static_entity


class Manager:
    """
    Generate and initialize the entities of the environment.
    """

    def __init__(self, env_config: EnvConfig, agent_config: AgentConfig):
        """
        Constructor
        """

        self.env_config: EnvConfig = env_config
        self.agent_config: AgentConfig = agent_config

        self.width: int = env_config.width
        self.height: int = env_config.height

        self.agents: list[agent.Agent] = []
        self.static_obstacles: list[static_entity.StaticEntity] = []
        self.moving_obstacles: list[moving_entity.MovingEntity] = []

    # ---------------------------------------------------------------
    # GENERATE
    # ---------------------------------------------------------------

    def generate(
        self,
    ) -> tuple[
        list[static_entity.StaticEntity],
        list[moving_entity.MovingEntity],
        list[agent.Agent],
    ]:
        """
        Generate all entities of the environment.
        """

        static_obstacles = self.generate_static_obstacles()
        moving_obstacles = self.generate_moving_obstacles()
        agents = self.generate_agents()

        return static_obstacles, moving_obstacles, agents

    def reset(self) -> tuple[list[moving_entity.MovingEntity], list[agent.Agent]]:
        """
        Reset and regenerate the dynamic entities.
        """

        # Delete current dynamic entities
        self.moving_obstacles = []
        self.agents = []

        # Generate new ones
        moving_obstacles = self.generate_moving_obstacles()
        agents = self.generate_agents()

        return moving_obstacles, agents

    def generate_static_obstacles(self) -> list[static_entity.StaticEntity]:
        """
        Generate the static obstacles.
        """

        self.static_obstacles = []

        # Retrieve the generation parameters
        pad = self.env_config.thickness
        min_dist = self.env_config.margin
        max_attempts = self.env_config.max_attempts
        mode = self.env_config.env_mode

        # Add the environment border walls
        self._add_walls(pad)

        # Generate the obstacles according to the selected mode
        if "fixed" in mode:
            n_cols = int(mode.split("_")[-1])
            if "fixed_random" not in mode:
                self._fixed_static_obstacles(n_cols=n_cols)
            else:
                self._fixed_random_static_obstacles(n_cols=n_cols)
        if mode == "random":
            self._random_static_obstacles(min_dist=min_dist, max_attempts=max_attempts)
        if mode == "warehouse":
            self._warehouse_static_obstacles()
        if mode == "crossing":
            self._crossing_static_obstacles()

        return self.static_obstacles

    def generate_moving_obstacles(self) -> list[moving_entity.MovingEntity]:
        """
        Generate the moving obstacles.
        """

        # Retrieve the generation parameters
        nb_targets = self.env_config.nb_targets
        radius_min = self.env_config.radius_min
        radius_max = self.env_config.radius_max
        min_dist = self.env_config.margin
        max_attempts = self.env_config.max_attempts
        mode = self.env_config.agent_mode

        # Generate the obstacles according to the selected mode
        if "fixed" in mode:
            self._fixed_moving_obstacles(
                nb_targets=nb_targets,
                min_dist=min_dist,
                max_attempts=max_attempts,
            )
        if mode == "random" or mode == "warehouse":
            self._random_moving_obstacles(
                nb_targets=nb_targets,
                radius_min=radius_min,
                radius_max=radius_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
            )
        if mode == "crossing":
            self._crossing_moving_obstacles(
                min_dist=min_dist, max_attempts=max_attempts
            )

        return self.moving_obstacles

    def generate_agents(self) -> list[agent.Agent]:
        """
        Generate the agents.
        """

        # Retrieve the generation parameters
        nb_targets = self.env_config.nb_targets
        min_dist = self.env_config.margin // 2
        max_attempts = self.env_config.max_attempts
        mode = self.env_config.agent_mode

        # Generate the agents according to the selected mode
        if "fixed" in mode:
            n_cols = int(mode.split("_")[-1])
            self._fixed_agents(
                nb_targets=nb_targets,
                n_cols=n_cols,
                min_dist=min_dist,
                max_attempts=max_attempts,
            )
        if mode == "random":
            self._random_agents(
                nb_targets=nb_targets, min_dist=min_dist, max_attempts=max_attempts
            )
        if mode == "crossing":
            self._crossing_agents(min_dist=min_dist, max_attempts=max_attempts)

        return self.agents

    # ---------------------------------------------------------------
    # GENERATE STATIC OBSTACLES
    # ---------------------------------------------------------------

    def _add_walls(self, pad: int):
        """
        Generate the environment border walls.
        """

        # Create the four border walls
        top_wall = static_entity.StaticEntity(width=self.width, height=pad, num=1)
        right_wall = static_entity.StaticEntity(width=pad, height=self.height, num=2)
        bot_wall = static_entity.StaticEntity(width=self.width, height=pad, num=3)
        left_wall = static_entity.StaticEntity(width=pad, height=self.height, num=4)

        # Set the positions
        top_wall.current_position = (
            self.width / 2,
            self.height,
        )
        right_wall.current_position = (
            self.width,
            self.height / 2,
        )
        bot_wall.current_position = (self.width / 2, 0)
        left_wall.current_position = (0, self.height / 2)

        # Add the walls to the environment
        self.static_obstacles.extend([top_wall, right_wall, bot_wall, left_wall])

    def _fixed_static_obstacles(self, n_cols: int):
        """
        Generate fixed static obstacles distributed over several columns.
        """

        # Prevent the number of columns from exceeding the number of obstacles
        n = self.env_config.nb_static_obstacles
        if n < 1:
            return
        n_cols = min(n_cols, n)

        # Define obstacle dimensions relative to the environment dimensions
        w = max(1, self.width // 16)
        h = max(1, self.height // 16)

        # Compute evenly x positions for the columns
        x_positions = np.linspace(
            self.width / (n_cols + 1),
            self.width * n_cols / (n_cols + 1),
            n_cols,
        )
        base = n // n_cols  # Minimum obstacles per columns
        remainder = n % n_cols  # Remaining obstacles if there are

        # Iterate over the columns
        for col_idx, x in enumerate(x_positions):
            n_rows = base + (1 if col_idx < remainder else 0)

            # Compute evenly y positions within the column
            y_positions = np.linspace(
                self.height / (n_rows + 1),
                self.height * n_rows / (n_rows + 1),
                n_rows,
            )

            # Create the obstacles
            for y in y_positions:
                obstacle = static_entity.StaticEntity(
                    width=w,
                    height=h,
                    num=len(self.static_obstacles) + 1,
                )
                obstacle.current_position = (int(x), int(y))
                self.static_obstacles.append(obstacle)

    def _fixed_random_static_obstacles(self, n_cols: int):
        """
        Generate fixed columns of static obstacles with small random offsets.
        """

        # Prevent the number of columns from exceeding the number of obstacles
        n = self.env_config.nb_static_obstacles
        if n < 1:
            return
        n_cols = min(n_cols, n)

        # Prevent the number of columns from exceeding the number of obstacles
        x_positions = np.linspace(
            self.width / (n_cols + 1),
            self.width * n_cols / (n_cols + 1),
            n_cols,
        )

        # Compute evenly x positions for the columns
        base = n // n_cols
        remainder = n % n_cols

        # Iterate over the columns
        for i, x in enumerate(x_positions):
            n_rows = base + (1 if i < remainder else 0)

            # Compute evenly y positions within the column
            y_positions = np.linspace(
                self.height / (n_rows + 1),
                self.height * n_rows / (n_rows + 1),
                n_rows,
            )

            # Create the obstacles
            for y in y_positions:
                # Select a random width
                w = np.random.randint(1, self.width // 16)
                # Select a random height
                h = np.random.randint(1, self.height // 16)

                obstacle = static_entity.StaticEntity(
                    width=w,
                    height=h,
                    num=len(self.static_obstacles) + 1,
                )
                obstacle.current_position = (int(x), int(y))

                self.static_obstacles.append(obstacle)

    def _random_static_obstacles(self, min_dist: float, max_attempts: int):
        """
        Generate random static obstacles with sizes adapted to the environment.
        """

        n = self.env_config.nb_static_obstacles
        if n < 1:
            return

        # Compute the minimul distance from the environment borders
        pad = int(self.env_config.thickness + min_dist)

        # Estimate the available surface per obstacle and get a reference dimension
        area = (self.width * self.height) / n
        base = int(np.sqrt(area))

        # Compute the width and height ranges
        w_min = max(1, base // 6)
        w_max = max(w_min + 1, base // 4)
        h_min = max(1, base // 6)
        h_max = max(h_min + 1, base // 4)

        # Attempt to generate the obstacles
        for _ in range(n):
            placed = False
            attempts = 0

            # Continue until a valid position is found or attempts are exhausted.
            while not placed and attempts < max_attempts:
                # Compute random width and height
                w = np.random.randint(w_min, w_max + 1)
                h = np.random.randint(h_min, h_max + 1)

                # Create obstacle
                obstacle = static_entity.StaticEntity(
                    width=w,
                    height=h,
                    num=len(self.static_obstacles) + 1,
                )

                # Compute a random position
                pos = (
                    np.random.randint(pad, self.width - pad),
                    np.random.randint(pad, self.height - pad),
                )

                # If the position is valid, assign it
                if self._is_free(
                    new=obstacle, pos=pos, min_dist=min_dist, for_target=False
                ):
                    obstacle.current_position = pos
                    self.static_obstacles.append(obstacle)
                    placed = True

                # Increment attempts
                attempts += 1

    def _warehouse_static_obstacles(self):
        """
        Generate a predefined static obstacle configuration.
        """

        n = self.env_config.nb_static_obstacles
        if n < 1:
            return

        n_cols = self.width // 10  # Number of columns adapted to the env width
        h = self.height // 2 - 10  # Obstacle height adapted to the env height
        w = 2  # Fixed obstacle width

        # Compute evenly centered x positions with 2 vertical obstacles
        x = [self.width * i for i in np.linspace(1 / n_cols, 1 - 1 / n_cols, n_cols)]
        y = [self.height * i for i in [0.3, 0.7]]

        # Create 2 vertical obstacles over all x positions
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
        """
        Generate the static obstacles for the crossing scenario.
        """

        n = self.env_config.nb_static_obstacles
        if n < 1:
            return

        # Compute the center of the environment
        cx, cy = self.width / 2, self.height / 2

        # Define the radius of the circular layout
        r = min(self.width, self.height) // 4

        # Generate evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Convert angles into a list of cartesian positions
        positions = []
        for a in angles:
            x = cx + r * np.cos(a)
            y = cy + r * np.sin(a)
            positions.append((x, y))

        # Generate an obstacle at each position
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

    def _fixed_moving_obstacles(
        self, nb_targets: int, min_dist: float, max_attempts: int
    ):
        """
        Generate moving obstacles and targets on each side of the obstacle columns.
        """

        n = self.env_config.nb_moving_obstacles
        if n < 1:
            return

        # Define the boundaries between the upper and lower sides
        top_limit = int(self.height * 0.5)
        bot_limit = int(self.height * 0.5)

        # Generate all obstacles
        for i in range(n):
            # Randomly choose the starting side
            side = np.random.choice(["top", "bot"])

            # Start in the top side and go to the bot side
            if side == "top":
                h_min, h_max = top_limit, self.height
                h_min_t, h_max_t = 0, bot_limit

            # Start in the bot side and go to the top side
            else:
                h_min, h_max = 0, bot_limit
                h_min_t, h_max_t = top_limit, self.height

            entity = moving_entity.MovingEntity(i + 1)

            # Generate a valid position on the chosen side
            pos = self._set_random_position(
                entity=entity,
                w_min=0,
                w_max=self.width,
                h_min=h_min,
                h_max=h_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos

            # Generate valid target positions on the opposite side
            for _ in range(nb_targets):
                target = self._set_random_position(
                    entity=entity,
                    w_min=0,
                    w_max=self.width,
                    h_min=h_min_t,
                    h_max=h_max_t,
                    min_dist=min_dist,
                    max_attempts=max_attempts,
                    for_target=False,
                )
                entity.target_positions.append(target)

            self.moving_obstacles.append(entity)

    def _random_moving_obstacles(
        self,
        nb_targets: int,
        radius_min: float,
        radius_max: float,
        min_dist: float,
        max_attempts: int,
    ):
        """
        Generate random moving obstacles.
        """

        n = self.env_config.nb_moving_obstacles
        if n < 1:
            return

        # Generate all obstacles
        for i in range(n):
            # Create the obstacle
            entity = moving_entity.MovingEntity(num=i + 1)

            # Generate a valid position
            pos = self._set_random_position(
                entity=entity,
                w_min=0,
                w_max=self.width,
                h_min=0,
                h_max=self.height,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos

            # Randomly select the obstacle radius
            entity.radius = np.random.uniform(radius_min, radius_max)

            # Generate valid target positions
            for _ in range(nb_targets):
                target = self._set_random_position(
                    entity=entity,
                    w_min=0,
                    w_max=self.width,
                    h_min=0,
                    h_max=self.height,
                    min_dist=min_dist,
                    max_attempts=max_attempts,
                    for_target=False,
                )
                entity.target_positions.append(target)

            self.moving_obstacles.append(entity)

    def _crossing_moving_obstacles(self, min_dist: float, max_attempts: int):
        """
        Generate moving obstacles for the crossing scenario.
        """

        n = self.env_config.nb_moving_obstacles
        if n < 1:
            return

        # Generate all obstacles
        for i in range(n):
            moving_obstacle = moving_entity.MovingEntity(num=i + 1)

            # Generate a position and its opposite
            pos, target = self._set_circular_position(
                entity=moving_obstacle, min_dist=min_dist, max_attempts=max_attempts
            )
            moving_obstacle.start_position = pos
            moving_obstacle.current_position = pos
            moving_obstacle.target_positions.append(target)

            # Use the same radius as the agents
            moving_obstacle.radius = self.agent_config.radius

            self.moving_obstacles.append(moving_obstacle)

    # ---------------------------------------------------------------
    # GENERATE AGENTS
    # ---------------------------------------------------------------

    def _fixed_agents(
        self, nb_targets: int, n_cols: int, min_dist: float, max_attempts: int
    ):
        """
        Generate agents and targets on each side of the obstacle columns.
        """

        n = self.env_config.nb_agents
        if n < 1:
            return

        # Estimate the obstacle width
        w = self.width // 16
        obstacle_zone = n_cols * w

        # Compute the center of the environment
        cx = self.width // 2

        # Compute the left and right boundaries of the obstacle area
        left_limit = int(cx - obstacle_zone / 2)
        right_limit = int(cx + obstacle_zone / 2)

        # Generate all agents
        for i in range(n):
            entity = agent.Agent(self.agent_config, i + 1)

            # Generate a valid position on the left side
            pos = self._set_random_position(
                entity=entity,
                w_min=0,
                w_max=left_limit,
                h_min=0,
                h_max=self.height,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos
            entity.old_position = pos
            entity.path = [pos]

            # Generate valid target positions on the right side
            for _ in range(nb_targets):
                target = self._set_random_position(
                    entity=entity,
                    w_min=right_limit,
                    w_max=self.width,
                    h_min=0,
                    h_max=self.height,
                    min_dist=min_dist,
                    max_attempts=max_attempts,
                    for_target=True,
                )
                entity.target_positions.append(target)

            self.agents.append(entity)

    def _random_agents(self, nb_targets: int, min_dist: float, max_attempts: int):
        """
        Generate random agents.
        """

        n = self.env_config.nb_agents
        if n < 1:
            return

        # Generate all agents
        for i in range(n):
            entity = agent.Agent(self.agent_config, i + 1)

            # Generate a valid position
            pos = self._set_random_position(
                entity=entity,
                w_min=0,
                w_max=self.width,
                h_min=0,
                h_max=self.height,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos
            entity.old_position = pos
            entity.path = [pos]

            # Generate valid target positions
            for _ in range(nb_targets):
                target = self._set_random_position(
                    entity=entity,
                    w_min=0,
                    w_max=self.width,
                    h_min=0,
                    h_max=self.height,
                    min_dist=min_dist,
                    max_attempts=max_attempts,
                    for_target=True,
                )

                entity.target_positions.append(target)

            self.agents.append(entity)

    def _crossing_agents(self, min_dist: float, max_attempts: int):
        """
        Generate agents for the crossing scenario.
        """

        n = self.env_config.nb_agents
        if n < 1:
            return

        for i in range(n):
            entity = agent.Agent(self.agent_config, i + 1)

            # Generate a position and its opposite
            pos, target = self._set_circular_position(
                entity=entity, min_dist=min_dist, max_attempts=max_attempts
            )
            entity.start_position = pos
            entity.current_position = pos
            entity.old_position = pos
            entity.path = [pos]
            entity.target_positions.append(target)

            self.agents.append(entity)

    # ---------------------------------------------------------------
    # SET POSITIONS
    # ---------------------------------------------------------------

    def _set_random_position(
        self,
        entity: static_entity.StaticEntity | moving_entity.MovingEntity | agent.Agent,
        w_min: int,
        w_max: int,
        h_min: int,
        h_max: int,
        min_dist: float,
        max_attempts: int,
        for_target: bool = False,
    ) -> tuple[int, int]:
        """
        Generate a random collision-free position.
        """

        # Compute the minimum distance from the environment borders
        pad = int(self.env_config.thickness + min_dist)

        is_free = False
        i = 0

        # Continue until a valid position is found or the limit is reached
        while not is_free and i < max_attempts:
            # Generate a random position
            pos = (
                np.random.randint(w_min + pad, w_max - pad),
                np.random.randint(h_min, h_max - pad),
            )

            # Check whether the position is collision-free
            is_free = self._is_free(
                new=entity, pos=pos, min_dist=min_dist, for_target=for_target
            )

            # Increment the attempt counter
            i += 1

        return pos

    def _set_circular_position(
        self,
        entity: static_entity.StaticEntity | moving_entity.MovingEntity | agent.Agent,
        min_dist: float,
        max_attempts: int,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """
        Generate a random position on a circle and its opposite target.
        """

        # Compute the centers of the environment
        cx = self.width / 2
        cy = self.height / 2

        # Define a circle inside the environment boundaries
        radius = min(self.width, self.height) / 2 - 5

        # Try several random positions
        for _ in range(max_attempts):
            # Generate a random position around the circle
            angle = np.random.uniform(0, 2 * np.pi)
            pos = (
                cx + radius * np.cos(angle),
                cy + radius * np.sin(angle),
            )

            # Check whether the position is collision-free
            if self._is_free(
                new=entity,
                pos=pos,
                min_dist=min_dist,
                for_target=False,
            ):
                opposite_angle = angle + np.pi

                target = (
                    cx + radius * np.cos(opposite_angle),
                    cy + radius * np.sin(opposite_angle),
                )

                return pos, target

        return None

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

        # Target positions only need to avoid static entities
        if for_target:
            entities = self.static_obstacles

        # Start positions need to avoid all entities
        else:
            entities = self.agents + self.static_obstacles + self.moving_obstacles

        # Compare the position with entities
        for entity in entities:
            # Ignore itself
            if entity is new:
                continue

            # Ignore unplaced entities
            if entity.current_position is None:
                continue

            # Reject the position if the safety distance is not respected
            if new.collides_with(other=entity, new_pos=pos, min_dist=min_dist):
                return False

        return True
