import numpy as np
from gymnasium import spaces
from simulator.entities.agent import Agent


def get_single_observation_space(width: int, height: int):
    return spaces.Dict(
        {
            "local_map": spaces.Box(
                low=0,
                high=1,
                shape=(1, Agent.LENGTH_VIEW, Agent.LENGTH_VIEW),
                dtype=np.float64,
            ),
            "goal_relative_position": spaces.Box(
                low=-max(width, height),
                high=max(width, height),
                shape=(2,),
                dtype=np.float64,
            ),
            "motion": spaces.Box(
                low=np.array([Agent.V_MIN, Agent.OMEGA_MIN]),
                high=np.array([Agent.V_MAX, Agent.OMEGA_MAX]),
                dtype=np.float64,
            ),
            "orientation": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
        }
    )


def get_single_action_space(
    allowed_v_min: float,
    allowed_v_max: float,
    allowed_omega_min: float,
    allowed_omega_max: float,
):
    return spaces.Box(
        low=np.array([allowed_v_min, allowed_omega_min]),
        high=np.array([allowed_v_max, allowed_omega_max]),
        dtype=np.float64,
    )
