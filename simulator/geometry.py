import numpy as np


def get_heading_error(
    goal_relative_position: tuple[float, float],
    theta: float,
) -> float:
    """
    Return the angle between the agent heading and the goal direction.
    """

    dx, dy = goal_relative_position
    goal_angle = np.arctan2(dy, dx)
    error = goal_angle - theta
    error = np.arctan2(np.sin(error), np.cos(error))

    return float(error)


def get_normalized_heading_error(
    goal_relative_position: tuple[float, float],
    theta: float,
) -> np.ndarray:
    """
    Return heading error encoded as cos/sin.
    """

    error = get_heading_error(goal_relative_position, theta)

    return np.array(
        [np.cos(error), np.sin(error)],
        dtype=np.float32,
    )


def get_normalized_relative_distance(
    distance: float,
    env_width: int,
    env_height: int,
) -> np.ndarray:
    """
    Normalize a relative distance between 0 and 1.
    """

    max_distance = np.hypot(env_width, env_height)

    return np.array(
        [distance / max_distance],
        dtype=np.float32,
    )


def get_normalized_position(
    pos: tuple[float, float], env_width: int, env_height: int
) -> np.ndarray:
    """
    Normalize a relative position to [-1, 1].
    """

    x, y = pos
    norm_x, norm_y = x / env_width, y / env_height

    return np.array([norm_x, norm_y], dtype=np.float32)


def get_normalized_motion(
    motion: tuple[float, float], v_max: float, omega_max: float
) -> np.ndarray:
    """
    Normalize a motion.
    """

    v, omega = motion
    norm_v, norm_omega = v / v_max, omega / omega_max

    return np.array([norm_v, norm_omega], dtype=np.float32)


def get_relative_position(
    pos1: tuple[float, float], pos2: tuple[float, float]
) -> tuple[float, float]:
    """
    Return the relative position of two positions.
    """

    x1, y1 = pos1
    x2, y2 = pos2

    return (x1 - x2, y1 - y2)


def get_relative_distance(
    pos1: tuple[float, float], pos2: tuple[float, float]
) -> float:
    """
    Compute the Euclidean distance between two positions.
    """

    return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))


def get_distance_rectangle_rectangle(
    x1_min: float,
    y1_min: float,
    x1_max: float,
    y1_max: float,
    x2_min: float,
    y2_min: float,
    x2_max: float,
    y2_max: float,
) -> float:
    """
    Compute the minimum Euclidean distance between two
    axis-aligned rectangles.
    """

    dx = max(x1_min - x2_max, x2_min - x1_max, 0)
    dy = max(y1_min - y2_max, y2_min - y1_max, 0)

    return np.hypot(dx, dy)


def get_distance_circle_circle(
    cx1: float, cy1: float, radius1: float, cx2: float, cy2: float, radius2: float
) -> float:
    """
    Compute the minimum Euclidean distance between two
    axis-aligned circles.
    """

    d = np.linalg.norm(np.array((cx1, cy1)) - np.array((cx2, cy2)))

    return float(d - radius1 - radius2)


def get_distance_rectangle_circle(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    cx: float,
    cy: float,
    radius,
) -> float:
    """
    Compute the minimum Euclidean distance between an
    axis-aligned rectangle an a circle.
    """

    closest_x = np.clip(cx, x_min, x_max)
    closest_y = np.clip(cy, y_min, y_max)

    return max(0, np.linalg.norm([cx - closest_x, cy - closest_y]) - radius)
