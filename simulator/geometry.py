import numpy as np


def get_relative_position(
    pos1: tuple[float, float], pos2: tuple[float, float]
) -> tuple[float, float]:
    """
    Return the relative position of two positions.

    The goal position is expressed in the world reference frame
    relative to the agent's current position.
    """

    x1, y1 = pos1
    x2, y2 = pos2
    return (x1 - x2, y1 - y2)


def get_relative_distance(
    pos1: tuple[float, float], pos2: tuple[float, float]
) -> float:
    """
    Compute the Euclidean distance between two
    positions.
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

    The returned distance is:
    - 0 if the rectangles overlap or touch
    - positive otherwise
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

    The returned distance is:
    - 0 if the circles overlap or touch
    - positive otherwise
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

    The returned distance is:
    - 0 if they overlap or touch
    - positive otherwise
    """

    closest_x = np.clip(cx, x_min, x_max)
    closest_y = np.clip(cy, y_min, y_max)
    return max(0, np.linalg.norm([cx - closest_x, cy - closest_y]) - radius)
