import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer used by off-policy reinforcement learning algorithms.

    Stores transitions of the form: (state, action, reward, next_state, done)
    and allows random sampling of mini-batches during training.
    """

    def __init__(
        self,
        map_shape: tuple[int, int, int],
        n_actions: int,
        buffer_length: int = 100_000,
    ):
        """
        Constructor
        """

        self.buffer_length = buffer_length  # Number of transitions that can be stored
        self.mem_counter = 0  # Total number of transitions ever stored

        # State memory
        self.local_map_memory = np.zeros(
            (buffer_length, *map_shape),
            dtype=np.float32,
        )
        self.goal_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )
        self.motion_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )
        self.orientation_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # Next state memory
        self.next_local_map_memory = np.zeros(
            (buffer_length, *map_shape),
            dtype=np.float32,
        )
        self.next_goal_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )
        self.next_motion_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )
        self.next_orientation_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # Step memory
        self.action_memory = np.zeros(
            (self.buffer_length, n_actions),
            dtype=np.float32,
        )
        self.reward_memory = np.zeros(
            (self.buffer_length),
            dtype=np.float32,
        )
        self.terminal_memory = np.zeros(self.buffer_length, dtype=bool)

    def store_transition(
        self,
        state: dict,
        action: np.ndarray,
        reward: float,
        next_state: dict,
        done: bool,
    ):
        """
        Store a transition in the replay buffer.
        """

        # Circular buffer index
        index = self.mem_counter % self.buffer_length

        self.local_map_memory[index] = state["local_map"]
        self.goal_memory[index] = state["goal_relative_position"]
        self.motion_memory[index] = state["motion"]
        self.orientation_memory[index] = state["orientation"]

        self.next_local_map_memory[index] = next_state["local_map"]
        self.next_goal_memory[index] = next_state["goal_relative_position"]
        self.next_motion_memory[index] = next_state["motion"]
        self.next_orientation_memory[index] = next_state["orientation"]

        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random mini-batch of transitions.
        """

        mem_size = min(self.mem_counter, self.buffer_length)
        batch = np.random.choice(mem_size, batch_size, replace=False)

        states = {
            "local_map": self.local_map_memory[batch],
            "goal_relative_position": self.goal_memory[batch],
            "motion": self.motion_memory[batch],
            "orientation": self.orientation_memory[batch],
        }

        next_states = {
            "local_map": self.next_local_map_memory[batch],
            "goal_relative_position": self.next_goal_memory[batch],
            "motion": self.next_motion_memory[batch],
            "orientation": self.next_orientation_memory[batch],
        }

        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
