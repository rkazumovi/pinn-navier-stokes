"""
Point sampling for 2D unsteady flow past a NACA airfoil.

Same structure as the cylinder-flow sampling: interior (x,y,t) points
excluding the airfoil interior, airfoil surface points (no-slip),
inlet, far-field, outlet, and initial condition points. The only
difference from the cylinder case is that "inside the obstacle" and
"on the obstacle surface" now use the airfoil polygon (geometry.py)
instead of a simple circle distance check.
"""

import torch
import numpy as np
from config import X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, INLET_U, INLET_V
from geometry import inside_airfoil, get_airfoil_surface_points


def sample_interior_points(n_points, device="cpu"):
    """
    Rejection-sample (x, y, t) triples inside the channel but OUTSIDE
    the airfoil polygon.
    """
    collected_x, collected_y, collected_t = [], [], []
    n_needed = n_points

    while n_needed > 0:
        batch = int(n_needed * 1.3) + 10
        x = np.random.uniform(X_MIN, X_MAX, batch)
        y = np.random.uniform(Y_MIN, Y_MAX, batch)
        t = np.random.uniform(T_MIN, T_MAX, batch)

        mask = ~inside_airfoil(x, y)
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


def sample_airfoil_boundary(n_points, device="cpu"):
    """
    Points on the airfoil surface (parametrized via cosine spacing along
    the chord in geometry.py). No-slip: u = 0, v = 0.
    """
    x_poly, y_poly = get_airfoil_surface_points(n_points_per_surface=n_points // 2)
    t = np.random.uniform(T_MIN, T_MAX, len(x_poly))

    x_t = torch.tensor(x_poly.reshape(-1, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_poly.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    u_bc = torch.zeros(len(x_poly), 1, device=device)
    v_bc = torch.zeros(len(x_poly), 1, device=device)

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
    """Top + bottom walls, approximated as free-stream (u=INLET_U, v=0)."""
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
    """Outlet (x = X_MAX): soft pressure reference p ~ 0."""
    x = np.full(n_points, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX, n_points)
    t = np.random.uniform(T_MIN, T_MAX, n_points)

    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    p_bc = torch.zeros(n_points, 1, device=device)

    return x_t, y_t, t_t, p_bc


def sample_initial_condition_points(n_points, device="cpu"):
    """At t=0: uniform flow everywhere outside the airfoil."""
    collected_x, collected_y = [], []
    n_needed = n_points

    while n_needed > 0:
        batch = int(n_needed * 1.3) + 10
        x = np.random.uniform(X_MIN, X_MAX, batch)
        y = np.random.uniform(Y_MIN, Y_MAX, batch)
        mask = ~inside_airfoil(x, y)
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

    mask_inside = inside_airfoil(x.detach().numpy(), y.detach().numpy())
    print(f"points incorrectly inside airfoil: {mask_inside.sum()} (should be 0)")

    x_a, y_a, t_a, u_a, v_a = sample_airfoil_boundary(100)
    print("airfoil boundary points:", x_a.shape)

    x_in, y_in, t_in, u_in, v_in = sample_inlet_points(50)
    print("inlet x (should all be X_MIN=0):", x_in.min().item(), x_in.max().item())

    x_out, y_out, t_out, p_out = sample_outlet_points(50)
    print("outlet x (should all be X_MAX=20):", x_out.min().item(), x_out.max().item())

    x_ic, y_ic, t_ic, u_ic, v_ic = sample_initial_condition_points(500)
    mask_ic = inside_airfoil(x_ic.numpy(), y_ic.numpy())
    print(f"IC points incorrectly inside airfoil: {mask_ic.sum()} (should be 0)")