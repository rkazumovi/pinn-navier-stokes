"""
NACA 4-digit airfoil geometry: shape generation, rotation by angle of
attack, and inside/outside test for rejection sampling.
"""

import numpy as np
from matplotlib.path import Path

from config import (
    NACA_CODE, CHORD, AIRFOIL_CENTER_X, AIRFOIL_CENTER_Y, ANGLE_OF_ATTACK_DEG,
)


def _naca4_thickness(x_over_c, t):
    """
    NACA 4-digit symmetric thickness distribution.
    x_over_c: array of x/c values in [0, 1]
    t: thickness ratio (e.g. 0.12 for NACA 0012)
    """
    return 5 * t * (
        0.2969 * np.sqrt(x_over_c)
        - 0.1260 * x_over_c
        - 0.3516 * x_over_c ** 2
        + 0.2843 * x_over_c ** 3
        - 0.1015 * x_over_c ** 4
    )


def _parse_naca4(code):
    """
    Parse NACA 4-digit code, e.g. '0012' -> (m=0.00, p=0.0, t=0.12).
    m: max camber (% of chord), p: location of max camber (tenths of chord),
    t: max thickness (% of chord). We only handle symmetric airfoils (m=0)
    for now, which covers '0012' etc.
    """
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    return m, p, t


def _cosine_spacing(n_points):
    beta = np.linspace(0, np.pi, n_points)
    return (1 - np.cos(beta)) / 2  # x/c values in [0, 1], denser at both ends


def naca4_local_coordinates(n_points_per_surface=100):
    """
    Generate the (x, y) coordinates of the airfoil surface in the
    UNROTATED, chord-aligned local frame (leading edge at x=0, trailing
    edge at x=CHORD). Returns the closed polygon: upper surface from LE
    to TE, then lower surface from TE back to LE.
    """
    m, p, t = _parse_naca4(NACA_CODE)
    x_over_c = _cosine_spacing(n_points_per_surface)
    yt = _naca4_thickness(x_over_c, t)

    x = x_over_c * CHORD
    y_upper = yt * CHORD
    y_lower = -yt * CHORD

    x_poly = np.concatenate([x, x[::-1]])
    y_poly = np.concatenate([y_upper, y_lower[::-1]])

    return x_poly, y_poly


def _rotate_and_translate(x_local, y_local):
    """
    Rotate local airfoil coordinates by -AoA, then translate so the
    leading edge sits at (AIRFOIL_CENTER_X, AIRFOIL_CENTER_Y).
    """
    theta = -np.radians(ANGLE_OF_ATTACK_DEG)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x_rot = x_local * cos_t - y_local * sin_t
    y_rot = x_local * sin_t + y_local * cos_t

    x_world = x_rot + AIRFOIL_CENTER_X
    y_world = y_rot + AIRFOIL_CENTER_Y

    return x_world, y_world


def get_airfoil_polygon(n_points_per_surface=100):
    """
    Returns the closed airfoil polygon in world coordinates (rotated by
    AoA, translated to its position in the channel).
    """
    x_local, y_local = naca4_local_coordinates(n_points_per_surface)
    return _rotate_and_translate(x_local, y_local)


def get_airfoil_surface_points(n_points_per_surface=100):
    """
    Same as get_airfoil_polygon, but intended for BC sampling.
    """
    return get_airfoil_polygon(n_points_per_surface)


_airfoil_path_cache = None


def _get_airfoil_path():
    """Cached matplotlib.path.Path object for fast inside/outside tests."""
    global _airfoil_path_cache
    if _airfoil_path_cache is None:
        x_poly, y_poly = get_airfoil_polygon(n_points_per_surface=150)
        vertices = np.column_stack([x_poly, y_poly])
        _airfoil_path_cache = Path(vertices)
    return _airfoil_path_cache


def inside_airfoil(x, y):
    """
    Vectorized inside/outside test. x, y: 1D numpy arrays.
    Returns boolean mask, True where the point is inside the airfoil.
    """
    path = _get_airfoil_path()
    points = np.column_stack([np.asarray(x).flatten(), np.asarray(y).flatten()])
    return path.contains_points(points)


if __name__ == "__main__":
    x_poly, y_poly = get_airfoil_polygon()
    print("polygon shape:", x_poly.shape, y_poly.shape)
    print("x range:", x_poly.min(), x_poly.max())
    print("y range:", y_poly.min(), y_poly.max())

    print(f"\nexpected leading edge near ({AIRFOIL_CENTER_X}, {AIRFOIL_CENTER_Y})")
    print("actual first point (LE):", x_poly[0], y_poly[0])

    test_x = np.array([AIRFOIL_CENTER_X + 0.5 * np.cos(np.radians(-5)),
                        AIRFOIL_CENTER_X - 100])
    test_y = np.array([AIRFOIL_CENTER_Y + 0.5 * np.sin(np.radians(-5)),
                        AIRFOIL_CENTER_Y])
    mask = inside_airfoil(test_x, test_y)
    print("\ninside test (expect [True, False]):", mask)

    xg = np.linspace(AIRFOIL_CENTER_X - 1, AIRFOIL_CENTER_X + 2, 200)
    yg = np.linspace(AIRFOIL_CENTER_Y - 1, AIRFOIL_CENTER_Y + 1, 200)
    Xg, Yg = np.meshgrid(xg, yg)
    mask_grid = inside_airfoil(Xg.flatten(), Yg.flatten())
    print(f"grid points inside airfoil: {mask_grid.sum()} / {mask_grid.size}")