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

    # Get all entity
    entities = static_obstacles + moving_obstacles + agents

    for ag in agents:
        # Do not recompute collisions for inactive agents
        if ag.state in ["terminated", "truncated"]:
            metrics[ag.id] = {
                "closest_distance": ag._closest_dist,
                "closest_entity": None,
            }
            continue

        min_distance = np.inf
        closest_entity = None

        # Search for the closest entity
        for other in entities:
            if other == ag or other.current_position is None:
                continue

            # Compute the distance
            dist = ag.get_distance(other=other)

            # Update the closest entity if needed
            if dist < min_distance:
                min_distance = dist
                closest_entity = other

            # Update the agent state if needed
            if dist < ag.collision_threshold:
                ag.state = "collided"

        # Store the closest entity metrics
        metrics[ag.id] = {
            "closest_distance": min_distance,
            "closest_entity": closest_entity,
        }

        # Update the agent closest distance
        ag._closest_dist = min_distance

    return metrics


def compute_dones(agents: list[agent.Agent], timeout: bool) -> tuple[dict, dict, dict]:
    """
    Compute the termination status of all agents.
    """

    # Compute terminal states
    terminated = compute_terminated(agents=agents)

    # Compute truncated states
    truncated = compute_truncated(agents=agents)

    # Mark an agent as done after termination, truncation, or timeout
    dones = {ag.id: terminated[ag.id] or truncated[ag.id] or timeout for ag in agents}

    return terminated, truncated, dones


def compute_terminated(agents: list[agent.Agent]) -> dict:
    """
    Compute the termination state of all agents.
    """

    for ag in agents:
        if ag.state == "reached":
            # If a target remains, continue
            if ag.target_index < len(ag.target_positions) - 1:
                ag.target_index += 1
                ag.state = "active"

            # Else, convert reached state into terminated state
            else:
                ag.state = "terminated"

    return {ag.id: ag.state == "terminated" for ag in agents}


def compute_truncated(agents: list[agent.Agent]) -> dict:
    """
    Compute the truncation state of all agents.
    """

    for ag in agents:
        # Convert collision state into truncated state
        if ag.state == "collided":
            ag.state = "truncated"

    return {ag.id: ag.state == "truncated" for ag in agents}
