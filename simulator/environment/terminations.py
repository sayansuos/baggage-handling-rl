import numpy as np

from simulator.entities import agent, moving_entity, static_entity


def compute_closest(
    static_obstacles: list[static_entity.StaticEntity],
    moving_obstacles: list[moving_entity.MovingEntity],
    agents: list[agent.Agent],
) -> dict:
    """
    Compute the closest surrounding entity for each agent.
    """

    metrics = {}
    entities = static_obstacles + moving_obstacles + agents

    for a in agents:

        min_distance = np.inf
        closest_entity = None

        for other in entities:

            if other == a or other.current_position is None:
                continue

            dist = a.get_distance(other)
            if dist < min_distance:
                min_distance = dist
                closest_entity = other

            if dist < 0.5:
                a.state = "collided"

        metrics[a.id] = {
            "closest_distance": min_distance,
            "closest_entity": closest_entity,
        }

        a._closest_dist = min_distance

    return metrics


def compute_dones(agents: list[agent.Agent], timeout: bool) -> tuple[dict, dict, dict]:
    """
    Compute the termination status of all agents.
    """

    terminated = compute_terminated(agents=agents)
    truncated = compute_truncated(agents=agents)
    dones = {a.id: terminated[a.id] or truncated[a.id] or timeout for a in agents}

    return terminated, truncated, dones


def compute_terminated(agents: list[agent.Agent]) -> dict:
    """
    Compute the termination state of all agents.
    """

    for agent in agents:

        if agent.state == "reached":

            if agent.target_index < len(agent.target_positions) - 1:
                agent.target_index += 1
                agent.state = "active"

            else:
                agent.state = "terminated"

    return {agent.id: agent.state == "terminated" for agent in agents}


def compute_truncated(agents: list[agent.Agent]) -> dict:
    """
    Compute the truncation state of all agents.
    """

    for agent in agents:

        if agent.state == "collided":
            agent.state = "truncated"

    return {agent.id: agent.state == "truncated" for agent in agents}
