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
        self._add_walls(pad=pad)

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
        if mode == "hospital":
            self._hospital_static_obstacles(pad=pad)
        if mode == "airport":
            self._airport_static_obstacles(
                pad=pad, min_dist=min_dist, max_attempts=max_attempts
            )

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
        if mode == "hospital":
            self._hospital_moving_obstacles(
                radius_min=radius_min,
                radius_max=radius_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
            )
        if mode == "airport":
            self._airport_moving_obstacles(
                nb_targets=nb_targets,
                radius_min=radius_min,
                radius_max=radius_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
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
        if mode == "hospital":
            self._hospital_agents(
                nb_targets=nb_targets, min_dist=min_dist, max_attempts=max_attempts
            )
        if mode == "airport":
            self._airport_agents(
                nb_targets=nb_targets, min_dist=min_dist, max_attempts=max_attempts
            )

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

    def _random_static_obstacles(
        self,
        min_dist: float,
        max_attempts: int,
        sizes: dict | None = None,
    ):
        """
        Generate random static obstacles with sizes adapted to the environment.
        """

        n = self.env_config.nb_static_obstacles
        if n < 1:
            return

        # Compute the minimul distance from the environment borders
        pad = int(self.env_config.thickness + min_dist)

        if sizes is not None:
            # Estimate the available surface per obstacle and get a reference dimension
            area = (self.width * self.height) / n
            base = int(np.sqrt(area))
            # Compute the width and height ranges
            w_min = max(1, base // 6)
            w_max = max(w_min + 1, base // 4)
            h_min = max(1, base // 6)
            h_max = max(h_min + 1, base // 4)
        else:
            # Get parameters from attributes
            area = sizes["area"]
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
        Generate warehouse shelves regularly distributed in the environment.
        """

        n = self.env_config.nb_static_obstacles
        if n < 1:
            return

        # Obstacles are arranged in two rows
        n_cols = int(np.ceil(n / 2))

        # Obstacle dimensions
        w = max(1, self.width // 32)
        h = max(2, self.height // 5)

        # Compute evenly centered x positions with 2 vertical obstacles
        x_positions = np.linspace(
            self.width / (n_cols + 1),
            self.width * n_cols / (n_cols + 1),
            n_cols,
        )

        # Compute upper and lower y positions
        y_positions = [
            self.height * 0.3,
            self.height * 0.7,
        ]

        count = 0

        # Create 2 vertical obstacles over all x positions
        for x in x_positions:
            for y in y_positions:
                # Stop if all obstacles has been created
                if count >= n:
                    return

                # Create obstacle
                obstacle = static_entity.StaticEntity(
                    width=w,
                    height=h,
                    num=len(self.static_obstacles) + 1,
                )
                obstacle.current_position = (x, y)
                self.static_obstacles.append(obstacle)

                # Increment obstacle
                count += 1

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

    def _hospital_static_obstacles(self, pad: float):
        """
        Generate a random hospital-like environment with a main corridor,
        an elevator area and rooms on both sides.
        """

        n = self.env_config.nb_static_obstacles
        if n < 1:
            return

        wall = pad / 2

        # Elevator area parameters
        elevator_w, elevator_h = 6, 6

        # Main corridor parameters
        corridor_h = 8

        # Create elevator horizontal walls
        elevator_bot_y = (self.height - elevator_h) / 2
        elevator_top_y = (self.height + elevator_h) / 2
        for y in [elevator_bot_y, elevator_top_y]:
            obstacle = static_entity.StaticEntity(
                width=elevator_w,
                height=wall,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (elevator_w / 2, y)
            self.static_obstacles.append(obstacle)

        # Create elevator vertical walls
        wall_bot_y = (self.height - corridor_h) / 4
        wall_top_y = self.height - (self.height - corridor_h) / 4
        wall_h = (self.height - min(elevator_h, corridor_h)) / 2 + 1
        for y in [wall_bot_y, wall_top_y]:
            obstacle = static_entity.StaticEntity(
                width=wall,
                height=wall_h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (elevator_w, y)
            self.static_obstacles.append(obstacle)

        # Create room walls
        room_walls_bot = []
        room_walls_top = []
        min_gap = self.width // 8
        wall_h = (self.height - max(elevator_h, corridor_h)) / 2
        wall_x_uniform = np.linspace(elevator_w + min_gap, self.width - min_gap, n - 2)
        for i, x in enumerate(wall_x_uniform):
            wall_x = x + np.random.uniform(-min_gap / 2, min_gap / 2)
            if i % 2 == 1:
                wall_y = wall_bot_y
                room_walls_bot.append(wall_x)
            else:
                wall_y = wall_top_y
                room_walls_top.append(wall_x)

            obstacle = static_entity.StaticEntity(
                width=wall,
                height=wall_h,
                num=len(self.static_obstacles) + 1,
            )
            obstacle.current_position = (wall_x, wall_y)
            self.static_obstacles.append(obstacle)

        # Create corridor walls
        door_w = 6 * self.agent_config.radius
        corridor_bot_y = (self.height - corridor_h) / 2
        corridor_top_y = (self.height + corridor_h) / 2
        self._hospital_corridor_wall(
            y=corridor_bot_y,
            room_walls=room_walls_bot,
            elevator_w=elevator_w,
            door_w=door_w,
            pad=wall,
        )
        self._hospital_corridor_wall(
            y=corridor_top_y,
            room_walls=room_walls_top,
            elevator_w=elevator_w,
            door_w=door_w,
            pad=wall,
        )

    def _hospital_corridor_wall(
        self,
        y: float,
        room_walls: list[float],
        elevator_w: float,
        door_w: float,
        pad: float,
    ):
        """
        Create one side of the main corridor with one door for each room.
        """

        # Add environment limits to define all rooms
        limits = [elevator_w] + sorted(room_walls) + [self.width]

        # Iterate over rooms
        for x_min, x_max in zip(limits[:-1], limits[1:]):
            # Center the door inside the room
            door_x = (x_min + x_max) / 2
            door_x_min = door_x - door_w / 2
            door_x_max = door_x + door_w / 2

            # Get walls parameters
            left_w = door_x_min - x_min
            right_w = x_max - door_x_max

            # Create the wall before the door
            if left_w > 0:
                obstacle = static_entity.StaticEntity(
                    width=left_w,
                    height=pad,
                    num=len(self.static_obstacles) + 1,
                )
                obstacle.current_position = (
                    x_min + left_w / 2,
                    y,
                )
                self.static_obstacles.append(obstacle)

            # Create the wall after the door
            if right_w > 0:
                obstacle = static_entity.StaticEntity(
                    width=right_w,
                    height=pad,
                    num=len(self.static_obstacles) + 1,
                )
                obstacle.current_position = (
                    door_x_max + right_w / 2,
                    y,
                )
                self.static_obstacles.append(obstacle)

    def _airport_static_obstacles(self, pad: float, min_dist: float, max_attempts: int):
        """
        Generate the pickup and delivery areas of the airport scenario,
        then add random static obstacles if required.
        """

        wall = pad / 2

        # Pickup and Delivery zone parameters
        w = self.width // 6
        h = self.height // 2
        pickup_x, delivery_x = w, self.width - w
        y = self.height / 2

        # Create pickup zone
        obstacle = static_entity.StaticEntity(
            width=wall, height=h, num=len(self.static_obstacles) + 1
        )
        obstacle.current_position = (pickup_x, y)
        self.static_obstacles.append(obstacle)

        # Create delivery zone
        obstacle = static_entity.StaticEntity(
            width=wall, height=h, num=len(self.static_obstacles) + 1
        )
        obstacle.current_position = (delivery_x, y)
        self.static_obstacles.append(obstacle)

        # Create fixed obstacles if required
        n = self.env_config.nb_static_obstacles
        if n > 1:
            for _ in range(n):
                return self._random_static_obstacles(
                    min_dist=min_dist, max_attempts=max_attempts
                )

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

    def _hospital_moving_obstacles(
        self,
        radius_min: float,
        radius_max: float,
        min_dist: float,
        max_attempts: int,
    ):
        """
        Generate moving obstacles traveling along the hospital corridor.
        """

        n = self.env_config.nb_moving_obstacles
        if n < 1:
            return

        # Elevator area parameters
        elevator_w, elevator_h = 6, 6

        # Corridor parameters
        corridor_h = 8
        corridor_w_min, corridor_w_max = elevator_w, self.width
        corridor_h_min = (self.height - corridor_h) / 2
        corridor_h_max = (self.height + corridor_h) / 2

        # Generate all obstacles
        for i in range(n):
            # Create the obstacle
            entity = moving_entity.MovingEntity(num=i + 1)

            # Randomly select one end of the corridor
            side = np.random.choice(["left", "right"])
            if side == "left":
                w_min = corridor_w_min
                w_max = corridor_w_min + self.width // 5
                target_w_min = 4 * self.width // 5
                target_w_max = corridor_w_max
            else:
                w_min = 4 * self.width // 5
                w_max = corridor_w_max
                target_w_min = corridor_w_min
                target_w_max = corridor_w_min + self.width // 5

            # Generate a valid position
            pos = self._set_random_position(
                entity=entity,
                w_min=w_min,
                w_max=w_max,
                h_min=corridor_h_min,
                h_max=corridor_h_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos

            # Randomly select the obstacle radius
            entity.radius = np.random.uniform(radius_min, radius_max)

            # Generate a valid target position
            target = self._set_random_position(
                entity=entity,
                w_min=target_w_min,
                w_max=target_w_max,
                h_min=corridor_h_min,
                h_max=corridor_h_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.target_positions.append(target)

            self.moving_obstacles.append(entity)

    def _airport_moving_obstacles(
        self,
        nb_targets: int,
        radius_min: float,
        radius_max: float,
        min_dist: float,
        max_attempts: int,
    ):
        """
        Generate moving obstacles between the pickup and delivery areas.
        """

        n = self.env_config.nb_moving_obstacles
        if n < 1:
            return

        # Pickup and Delivery zone parameters
        w = self.width // 6
        pickup_w_max, delivery_w_min = w, self.width - w
        h_min, h_max = 0, self.height

        # Generate all agents
        for i in range(n):
            # Create the obstacle
            entity = moving_entity.MovingEntity(num=i + 1)

            # Generate a valid position between the pickup and delivery areas
            pos = self._set_random_position(
                entity=entity,
                w_min=pickup_w_max,
                w_max=delivery_w_min,
                h_min=h_min,
                h_max=h_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos

            # Randomly select the obstacle radius
            entity.radius = np.random.uniform(radius_min, radius_max)

            # Generate valid target positions between the pickup and delivery areas
            for _ in range(nb_targets):
                target = self._set_random_position(
                    entity=entity,
                    w_min=pickup_w_max,
                    w_max=delivery_w_min,
                    h_min=h_min,
                    h_max=h_max,
                    min_dist=min_dist,
                    max_attempts=max_attempts,
                    for_target=False,
                )
                entity.target_positions.append(target)

            self.moving_obstacles.append(entity)

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

    def _hospital_agents(
        self,
        nb_targets: int,
        min_dist: float,
        max_attempts: int,
    ):
        """
        Generate agents starting from the elevator area, performing a mission
        on the hospital floor and returning to their starting position.
        """

        n = self.env_config.nb_agents
        if n < 1:
            return

        # Elevator area parameters
        elevator_w, elevator_h = 6, 6
        elevator_w_min, elevator_w_max = 0, elevator_w
        elevator_h_min = (self.height - elevator_h) / 2
        elevator_h_max = (self.height + elevator_h) / 2

        # Corridor parameters
        corridor_h = 8
        corridor_w_min, corridor_w_max = elevator_w_max, self.height
        corridor_h_min = (self.height - corridor_h) / 2
        corridor_h_max = (self.height + corridor_h) / 2

        # Generate all agents
        for i in range(n):
            entity = agent.Agent(
                self.agent_config,
                i + 1,
            )

            # Generate a valid position in the elevator area
            pos = self._set_random_position(
                entity=entity,
                w_min=elevator_w_min,
                w_max=elevator_w_max,
                h_min=elevator_h_min,
                h_max=elevator_h_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos
            entity.old_position = pos
            entity.path = [pos]

            # Generate valid target positions inside a room
            for _ in range(nb_targets - 1):
                side = np.random.choice(["top", "bot"])
                if side == "top":
                    h_min = corridor_h_max
                    h_max = self.height
                else:
                    h_min = 0
                    h_max = corridor_h_min

                target = self._set_random_position(
                    entity=entity,
                    w_min=corridor_w_min,
                    w_max=corridor_w_max,
                    h_min=h_min,
                    h_max=h_max,
                    min_dist=min_dist,
                    max_attempts=max_attempts,
                    for_target=True,
                )
                entity.target_positions.append(target)

            # Return to the elevator
            entity.target_positions.append(pos)

            self.agents.append(entity)

    def _airport_agents(self, nb_targets: int, min_dist: float, max_attempts: int):
        """
        Generate agents starting in the pickup area and alternating between
        delivery targets and returns to their initial position.
        """

        n = self.env_config.nb_agents
        if n < 1:
            return

        # Pickup and Delivery zone parameters
        w = self.width // 6
        h = self.height // 2
        pickup_w_min, pickup_w_max = 0, w
        delivery_w_min, delivery_w_max = self.width - w, self.width
        h_min, h_max = (self.height - h) // 2, (self.height + h) // 2

        # Generate all agents
        for i in range(n):
            entity = agent.Agent(self.agent_config, i + 1)

            # Generate a valid position in the pickup area
            pos = self._set_random_position(
                entity=entity,
                w_min=pickup_w_min,
                w_max=pickup_w_max,
                h_min=h_min,
                h_max=h_max,
                min_dist=min_dist,
                max_attempts=max_attempts,
                for_target=False,
            )
            entity.start_position = pos
            entity.current_position = pos
            entity.old_position = pos
            entity.path = [pos]

            # Generate a valid target positions in the delivery area
            for j in range(nb_targets):
                target = self._set_random_position(
                    entity=entity,
                    w_min=delivery_w_min,
                    w_max=delivery_w_max,
                    h_min=h_min,
                    h_max=h_max,
                    min_dist=min_dist,
                    max_attempts=max_attempts,
                    for_target=True,
                )
                entity.target_positions.append(target)

                if j < nb_targets - 1:
                    entity.target_positions.append(pos)

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
