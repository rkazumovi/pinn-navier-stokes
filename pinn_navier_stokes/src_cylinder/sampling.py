"""
Point sampling for 2D unsteady flow past a cylinder.

All interior/boundary points now include time t, since this is an
unsteady (vortex-shedding) problem: inputs are (x, y, t) triples.

Point types:
  - interior (x,y,t) collocation points, EXCLUDING the cylinder interior
  - cylinder surface points (parametrized by angle), no-slip BC
  - inlet points, uniform inflow BC
  - top/bottom far-field points, approximated as uniform free-stream
  - outlet points, soft pressure reference (no strong velocity BC)
  - initial condition points at t=0, uniform flow everywhere outside cylinder
"""

import torch
import numpy as np
from config import (
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    CYLINDER_CENTER_X, CYLINDER_CENTER_Y, CYLINDER_RADIUS,
    T_MIN, T_MAX, INLET_U, INLET_V,
)


def _inside_cylinder(x, y):
    """Boolean mask: True where (x,y) falls inside the cylinder."""
    dx = x - CYLINDER_CENTER_X
    dy = y - CYLINDER_CENTER_Y
    return (dx ** 2 + dy ** 2) < (CYLINDER_RADIUS ** 2)


def sample_interior_points(n_points, device="cpu"):
    """
    Rejection-sample (x, y, t) triples inside the channel but OUTSIDE
    the cylinder. We oversample, drop points inside the cylinder, and
    repeat until we have exactly n_points.
    """
    collected_x, collected_y, collected_t = [], [], []
    n_needed = n_points

    while n_needed > 0:
        batch = int(n_needed * 1.3) + 10
        x = np.random.uniform(X_MIN, X_MAX, batch)
        y = np.random.uniform(Y_MIN, Y_MAX, batch)
        t = np.random.uniform(T_MIN, T_MAX, batch)

        mask = ~_inside_cylinder(x, y)
        x, y, t = x[mask], y[mask], t[mask]

        take = min(len(x), n_needed)
        collected_x.append(x[:take])
        collected_y.append(y[:take])
        collected_t.append(t[:take])
        n_needed -= take

    x_all = np.concatenate(collected_x).reshape(-1, 1)
    y_all = np.concatenate(collected_y).reshape(-1, 1)
    t_all = np.concatenate(collected_t).reshape(-1, 1)

    x_t = torch.tensor(x_all, dtype=torch.float32, device=device, requires_grad=True)
    y_t = torch.tensor(y_all, dtype=torch.float32, device=device, requires_grad=True)
    t_t = torch.tensor(t_all, dtype=torch.float32, device=device, requires_grad=True)

    return x_t, y_t, t_t


def sample_cylinder_boundary(n_points, device="cpu"):
    """
    Points exactly on the cylinder surface, parametrized by angle theta.
    No-slip: u = 0, v = 0.
    """
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = CYLINDER_CENTER_X + CYLINDER_RADIUS * np.cos(theta)
    y = CYLINDER_CENTER_Y + CYLINDER_RADIUS * np.sin(theta)
    t = np.random.uniform(T_MIN, T_MAX, n_points)

    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    u_bc = torch.zeros(n_points, 1, device=device)
    v_bc = torch.zeros(n_points, 1, device=device)

    return x_t, y_t, t_t, u_bc, v_bc


def sample_inlet_points(n_points, device="cpu"):
    """Inlet (x = X_MIN): uniform inflow u=INLET_U, v=INLET_V."""
    x = np.full(n_points, X_MIN)
    y = np.random.uniform(Y_MIN, Y_MAX, n_points)
    t = np.random.uniform(T_MIN, T_MAX, n_points)

    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    u_bc = torch.full((n_points, 1), INLET_U, device=device)
    v_bc = torch.full((n_points, 1), INLET_V, device=device)

    return x_t, y_t, t_t, u_bc, v_bc


def sample_farfield_points(n_points, device="cpu"):
    """
    Top + bottom walls, approximated as free-stream (u=INLET_U, v=0).
    Simplification: valid since 8D is reasonably far from the wake region.
    """
    half = n_points // 2

    y_top = np.full(half, Y_MAX)
    y_bot = np.full(n_points - half, Y_MIN)
    y = np.concatenate([y_top, y_bot])
    x = np.random.uniform(X_MIN, X_MAX, n_points)
    t = np.random.uniform(T_MIN, T_MAX, n_points)

    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    u_bc = torch.full((n_points, 1), INLET_U, device=device)
    v_bc = torch.zeros(n_points, 1, device=device)

    return x_t, y_t, t_t, u_bc, v_bc


def sample_outlet_points(n_points, device="cpu"):
    """
    Outlet (x = X_MAX): NO strong velocity BC (would over-constrain the
    flow). Instead we softly reference pressure ~ 0 here as a simplified
    'do-nothing' outflow approximation.
    """
    x = np.full(n_points, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX, n_points)
    t = np.random.uniform(T_MIN, T_MAX, n_points)

    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    p_bc = torch.zeros(n_points, 1, device=device)

    return x_t, y_t, t_t, p_bc


def sample_initial_condition_points(n_points, device="cpu"):
    """
    At t=0: assume uniform flow everywhere outside the cylinder
    (u=INLET_U, v=0). The network must then evolve this correctly
    forward in time via the PDE loss.
    """
    collected_x, collected_y = [], []
    n_needed = n_points

    while n_needed > 0:
        batch = int(n_needed * 1.3) + 10
        x = np.random.uniform(X_MIN, X_MAX, batch)
        y = np.random.uniform(Y_MIN, Y_MAX, batch)
        mask = ~_inside_cylinder(x, y)
        x, y = x[mask], y[mask]
        take = min(len(x), n_needed)
        collected_x.append(x[:take])
        collected_y.append(y[:take])
        n_needed -= take

    x_all = np.concatenate(collected_x).reshape(-1, 1)
    y_all = np.concatenate(collected_y).reshape(-1, 1)
    t_all = np.full_like(x_all, T_MIN)

    x_t = torch.tensor(x_all, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_all, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_all, dtype=torch.float32, device=device)
    u_ic = torch.full((n_points, 1), INLET_U, device=device)
    v_ic = torch.zeros(n_points, 1, device=device)

    return x_t, y_t, t_t, u_ic, v_ic


if __name__ == "__main__":
    x, y, t = sample_interior_points(2000)
    print("interior:", x.shape, "requires_grad:", x.requires_grad)

    dx = x.detach().numpy() - CYLINDER_CENTER_X
    dy = y.detach().numpy() - CYLINDER_CENTER_Y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    n_inside = (dist < CYLINDER_RADIUS).sum()
    print(f"points incorrectly inside cylinder: {n_inside} (should be 0)")

    x_c, y_c, t_c, u_c, v_c = sample_cylinder_boundary(100)
    dist_c = np.sqrt((x_c.numpy() - CYLINDER_CENTER_X) ** 2 + (y_c.numpy() - CYLINDER_CENTER_Y) ** 2)
    print(f"cylinder boundary points, distance from center (should all be ~{CYLINDER_RADIUS}):",
          dist_c.min(), dist_c.max())

    x_in, y_in, t_in, u_in, v_in = sample_inlet_points(50)
    print("inlet x (should all be X_MIN=0):", x_in.min().item(), x_in.max().item())

    x_ff, y_ff, t_ff, u_ff, v_ff = sample_farfield_points(50)
    print("farfield y (should be 0 or 8):", torch.unique(y_ff))

    x_out, y_out, t_out, p_out = sample_outlet_points(50)
    print("outlet x (should all be X_MAX=20):", x_out.min().item(), x_out.max().item())

    x_ic, y_ic, t_ic, u_ic, v_ic = sample_initial_condition_points(500)
    print("IC t (should all be 0):", t_ic.min().item(), t_ic.max().item())
    dist_ic = np.sqrt((x_ic.numpy() - CYLINDER_CENTER_X) ** 2 + (y_ic.numpy() - CYLINDER_CENTER_Y) ** 2)
    print(f"IC points incorrectly inside cylinder: {(dist_ic < CYLINDER_RADIUS).sum()} (should be 0)")