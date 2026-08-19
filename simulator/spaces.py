import numpy as np
from gymnasium import spaces

from configs.config import AgentConfig


def get_observation_space(agent_config: AgentConfig) -> spaces.Dict:
    """
    Create the observation space of a single agent.
    """

    return spaces.Dict(
        {
            # Stack of local occupancy maps
            "local_map": spaces.Box(
                low=0,
                high=1,
                shape=(
                    agent_config.n_maps,
                    agent_config.length_view,
                    agent_config.length_view,
                ),
                dtype=np.float64,
            ),
            # Normalized distance to the current goal
            "goal_relative_distance": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32,
            ),
            # Goal heading encoded as cosine and sine
            "heading_error": spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float32,
            ),
            # Normalized linear and angular velocities
            "motion": spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float64,
            ),
            #  Agent orientation encoded as cosine and sine
            "orientation": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
        }
    )


def get_action_space(agent_config: AgentConfig) -> spaces.Box:
    """
    Create the action space of a single agent.
    """

    return spaces.Box(
        # Continuous linear and angular velocity commands.
        low=np.array([agent_config.v_min, agent_config.omega_min]),
        high=np.array([agent_config.v_max, agent_config.omega_max]),
        dtype=np.float64,
    )
