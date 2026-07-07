"""
Point sampling for the 2D lid-driven cavity PINN.

Generates:
  - interior (collocation) points for PDE residual loss
  - boundary points for BC loss (4 walls)
  - a single reference point to pin pressure

All returned as torch tensors with requires_grad=True where needed
(interior points need grad for computing derivatives; boundary points
don't need grad since we only compare network output to fixed BC values).
"""

import torch
from config import X_MIN, X_MAX, Y_MIN, Y_MAX


def sample_interior_points(n_points, device="cpu"):
    """
    Randomly sample n_points inside the open domain (x_min, x_max) x (y_min, y_max).
    These points require grad because we need du/dx, du/dy, d2u/dx2, etc.
    """
    x = torch.rand(n_points, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n_points, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    x.requires_grad_(True)
    y.requires_grad_(True)
    return x, y


def sample_boundary_points(n_per_wall, device="cpu"):
    """
    Sample points on the 4 walls of the unit square, plus the BC values
    (u, v) that should hold there.

    Returns a dict with keys 'top', 'bottom', 'left', 'right', each mapping
    to (x, y, u_bc, v_bc) tensors.
    """
    boundaries = {}

    # Top wall (lid): y = Y_MAX, u = 1, v = 0
    x_top = torch.rand(n_per_wall, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y_top = torch.full((n_per_wall, 1), Y_MAX, device=device)
    u_top = torch.ones(n_per_wall, 1, device=device)
    v_top = torch.zeros(n_per_wall, 1, device=device)
    boundaries["top"] = (x_top, y_top, u_top, v_top)

    # Bottom wall: y = Y_MIN, u = 0, v = 0
    x_bot = torch.rand(n_per_wall, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y_bot = torch.full((n_per_wall, 1), Y_MIN, device=device)
    u_bot = torch.zeros(n_per_wall, 1, device=device)
    v_bot = torch.zeros(n_per_wall, 1, device=device)
    boundaries["bottom"] = (x_bot, y_bot, u_bot, v_bot)

    # Left wall: x = X_MIN, u = 0, v = 0
    y_left = torch.rand(n_per_wall, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    x_left = torch.full((n_per_wall, 1), X_MIN, device=device)
    u_left = torch.zeros(n_per_wall, 1, device=device)
    v_left = torch.zeros(n_per_wall, 1, device=device)
    boundaries["left"] = (x_left, y_left, u_left, v_left)

    # Right wall: x = X_MAX, u = 0, v = 0
    y_right = torch.rand(n_per_wall, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    x_right = torch.full((n_per_wall, 1), X_MAX, device=device)
    u_right = torch.zeros(n_per_wall, 1, device=device)
    v_right = torch.zeros(n_per_wall, 1, device=device)
    boundaries["right"] = (x_right, y_right, u_right, v_right)

    return boundaries


def get_pressure_reference_point(device="cpu"):
    """
    Single point where we pin p = 0, since pressure in incompressible
    N-S is only defined up to an additive constant.
    Using the bottom-left corner (0,0).
    """
    x_ref = torch.tensor([[X_MIN]], device=device)
    y_ref = torch.tensor([[Y_MIN]], device=device)
    p_ref = torch.tensor([[0.0]], device=device)
    return x_ref, y_ref, p_ref


if __name__ == "__main__":
    # quick sanity check
    x_int, y_int = sample_interior_points(10)
    print("interior x:", x_int.shape, "requires_grad:", x_int.requires_grad)

    bounds = sample_boundary_points(5)
    for name, (x, y, u, v) in bounds.items():
        print(f"{name}: x{x.shape} y{y.shape} u{u.shape} v{v.shape}")

    x_ref, y_ref, p_ref = get_pressure_reference_point()
    print("pressure ref:", x_ref.item(), y_ref.item(), p_ref.item())