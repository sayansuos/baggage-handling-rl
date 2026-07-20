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
        buffer_length: int,
    ):
        """
        Constructor
        """

        # Maximum number ot transitions that can be stored
        self.buffer_length = buffer_length

        # Total number of transitions ever stored
        self.mem_counter = 0

        # ------------------------------------
        # Store the CURRENT STATE memory
        # ------------------------------------

        # Local occupancy maps of the current states
        self.local_map_memory = np.zeros(
            (buffer_length, *map_shape),
            dtype=np.float32,
        )

        # Normalized distances to the goals in the current states
        self.goal_memory = np.zeros(
            (buffer_length, 1),
            dtype=np.float32,
        )

        # Goal heading errors of the current states
        self.heading_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # Normalized motion components of the current states
        self.motion_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # Orientations of the current states
        self.orientation_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # ------------------------------------
        # Store the NEXT STATE memory
        # ------------------------------------

        # Local occupancy maps of the next states
        self.next_local_map_memory = np.zeros(
            (buffer_length, *map_shape),
            dtype=np.float32,
        )

        # Normalized distances to the goals in the next states
        self.next_goal_memory = np.zeros(
            (buffer_length, 1),
            dtype=np.float32,
        )

        # Goal heading errors of the next states
        self.next_heading_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # Normalized motion components of the next states
        self.next_motion_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # Orientations of the next states
        self.next_orientation_memory = np.zeros(
            (buffer_length, 2),
            dtype=np.float32,
        )

        # ------------------------------------
        # Store the TRANSITION memory
        # ------------------------------------

        # Actions applied at each transition
        self.action_memory = np.zeros(
            (self.buffer_length, n_actions),
            dtype=np.float32,
        )

        # Reward received after each transition
        self.reward_memory = np.zeros(
            (self.buffer_length),
            dtype=np.float32,
        )

        # Terminal flag
        self.terminal_memory = np.zeros(self.buffer_length, dtype=bool)

    def store_transition(
        self,
        state: dict,
        action: np.ndarray,
        reward: float,
        next_state: dict,
        done: bool,
    ) -> None:
        """
        Store a transition in the replay buffer.
        """

        # Compute the storage index using a circular buffer
        index = self.mem_counter % self.buffer_length

        # Store the current state components
        self.local_map_memory[index] = state["local_map"]
        self.goal_memory[index] = state["goal_relative_distance"]
        self.heading_memory[index] = state["heading_error"]
        self.motion_memory[index] = state["motion"]
        self.orientation_memory[index] = state["orientation"]

        # Store the next state components
        self.next_local_map_memory[index] = next_state["local_map"]
        self.next_goal_memory[index] = next_state["goal_relative_distance"]
        self.next_heading_memory[index] = next_state["heading_error"]
        self.next_motion_memory[index] = next_state["motion"]
        self.next_orientation_memory[index] = next_state["orientation"]

        # Store the transition state components (action, reward and terminal flag)
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        # Increment the total number of stored transitions
        self.mem_counter += 1

    def sample(
        self, batch_size: int
    ) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch of transitions.
        """

        # Compute the number of transitions currently available for sampling
        mem_size = min(self.mem_counter, self.buffer_length)

        # Randomly select unique transition indices
        batch = np.random.choice(mem_size, batch_size, replace=False)

        # Gather the current state components for the selected transitions
        states = {
            "local_map": self.local_map_memory[batch],
            "goal_relative_distance": self.goal_memory[batch],
            "heading_error": self.heading_memory[batch],
            "motion": self.motion_memory[batch],
            "orientation": self.orientation_memory[batch],
        }

        # Gather the next state components for the selected transitions
        next_states = {
            "local_map": self.next_local_map_memory[batch],
            "goal_relative_distance": self.next_goal_memory[batch],
            "heading_error": self.next_heading_memory[batch],
            "motion": self.next_motion_memory[batch],
            "orientation": self.next_orientation_memory[batch],
        }

        # Gather the actions, rewards and terminal flags for the selected transitions
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
