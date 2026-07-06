import numpy as np
from gymnasium import spaces

from configs.config import AgentConfig


def get_multi_spaces(
    nb_agents: int, agent_config: AgentConfig
) -> tuple[spaces.Dict, spaces.Dict]:
    """
    Create the observation and action spaces for all agents.
    """

    observation_space = spaces.Dict(
        {
            f"agent_{i+1}": get_single_observation_space(agent_config)
            for i in range(nb_agents)
        }
    )
    action_space = spaces.Dict(
        {
            f"agent_{i+1}": get_single_action_space(agent_config)
            for i in range(nb_agents)
        }
    )

    return observation_space, action_space


def get_single_observation_space(agent_config: AgentConfig) -> spaces.Dict:
    """
    Create the observation space of a single agent.
    """

    return spaces.Dict(
        {
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
            "goal_relative_distance": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32,
            ),
            "heading_error": spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float32,
            ),
            "motion": spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float64,
            ),
            "orientation": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
        }
    )


def get_single_action_space(agent_config: AgentConfig) -> spaces.Box:
    """
    Create the action space of a single agent.
    """

    return spaces.Box(
        low=np.array([agent_config.v_min, agent_config.omega_min]),
        high=np.array([agent_config.v_max, agent_config.omega_max]),
        dtype=np.float64,
    )
