import numpy as np
from gymnasium import spaces
from simulator.utils.config import EnvConfig, AgentConfig


def get_single_observation_space(
    env_config: EnvConfig, agent_config: AgentConfig
) -> spaces.Dict:
    return spaces.Dict(
        {
            "local_map": spaces.Box(
                low=0,
                high=1,
                shape=(1, agent_config.length_view, AgentConfig.length_view),
                dtype=np.float64,
            ),
            "goal_relative_position": spaces.Box(
                low=-max(env_config.width, env_config.height),
                high=max(env_config.width, env_config.height),
                shape=(2,),
                dtype=np.float64,
            ),
            "motion": spaces.Box(
                low=np.array([agent_config.v_min, agent_config.omega_min]),
                high=np.array([agent_config.v_max, agent_config.omega_max]),
                dtype=np.float64,
            ),
            "orientation": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
        }
    )


def get_single_action_space(env_config: EnvConfig) -> spaces.Box:
    return spaces.Box(
        low=np.array([env_config.v_min_allowed, env_config.omega_min_allowed]),
        high=np.array([env_config.v_max_allowed, env_config.omega_min_allowed]),
        dtype=np.float64,
    )
