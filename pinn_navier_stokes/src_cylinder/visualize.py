"""
Visualization + physical validation for the cylinder-flow PINN.

Produces:
  1. Vorticity field snapshots over time
  2. Lift and drag coefficient time series (computed by integrating
     pressure + viscous shear stress around the cylinder surface)
  3. FFT of the lift coefficient -> dominant shedding frequency ->
     Strouhal number, compared against the known Re=100 benchmark
     range (St ~ 0.16-0.17) from the literature.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import CylinderPINN
from config import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX,
    CYLINDER_CENTER_X, CYLINDER_CENTER_Y, CYLINDER_RADIUS, RE, INLET_U,
)

MODEL_PATH = "../outputs/pinn_cylinder.pt"
N_GRID = 150
N_TIME_FRAMES = 8
N_SURFACE_PTS = 200
N_TIME_SAMPLES_FOR_FFT = 200


def load_model():
    model = CylinderPINN(hidden_layers=8, hidden_units=100)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def _inside_cylinder(X, Y):
    return (X - CYLINDER_CENTER_X) ** 2 + (Y - CYLINDER_CENTER_Y) ** 2 < CYLINDER_RADIUS ** 2


def evaluate_snapshot(model, t_value, n=N_GRID):
    x = np.linspace(X_MIN, X_MAX, n)
    y = np.linspace(Y_MIN, Y_MAX, n)
    X, Y = np.meshgrid(x, y)

    x_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    t_t = torch.tensor(np.full_like(X.reshape(-1, 1), t_value), dtype=torch.float32)

    u, v, p = model(x_t, y_t, t_t)

    # vorticity = dv/dx - du/dy
    v_x = torch.autograd.grad(v, x_t, torch.ones_like(v), create_graph=False, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y_t, torch.ones_like(u), create_graph=False, retain_graph=True)[0]
    vorticity = (v_x - u_y).detach().numpy().reshape(n, n)

    U = u.detach().numpy().reshape(n, n)
    V = v.detach().numpy().reshape(n, n)
    P = p.detach().numpy().reshape(n, n)

    # mask out the cylinder interior for cleaner plots
    mask = _inside_cylinder(X, Y)
    vorticity[mask] = np.nan
    P[mask] = np.nan

    return X, Y, U, V, P, vorticity


def plot_time_snapshots(model):
    """Grid of vorticity snapshots at increasing times."""
    t_values = np.linspace(T_MIN + 2.0, T_MAX, N_TIME_FRAMES)  # skip early transient

    fig, axes = plt.subplots(2, N_TIME_FRAMES // 2, figsize=(20, 8))
    axes = axes.flatten()

    for i, t_val in enumerate(t_values):
        X, Y, U, V, P, vort = evaluate_snapshot(model, t_val)
        im = axes[i].contourf(X, Y, vort, levels=60, cmap="RdBu_r")
        circle = plt.Circle((CYLINDER_CENTER_X, CYLINDER_CENTER_Y), CYLINDER_RADIUS, color="black")
        axes[i].add_patch(circle)
        axes[i].set_title(f"t = {t_val:.1f}")
        axes[i].set_xlim(X_MIN, X_MAX)
        axes[i].set_ylim(Y_MIN, Y_MAX)
        axes[i].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("../outputs/cylinder_vorticity_snapshots.png", dpi=120)
    print("Saved plot to ../outputs/cylinder_vorticity_snapshots.png")
    plt.show()


def compute_lift_drag(model, t_value, n_surface=N_SURFACE_PTS):
    """
    Integrate pressure + viscous shear stress around the cylinder surface
    to get drag (Fx) and lift (Fy) coefficients at a given time.

    Traction on the surface: t_i = -p*n_i + (1/Re)*(du_i/dx_j + du_j/dx_i)*n_j
    Simplified 2D form used here (standard for cylinder-flow PINN validation).
    """
    theta = np.linspace(0, 2 * np.pi, n_surface, endpoint=False)
    nx = np.cos(theta)  # outward normal components
    ny = np.sin(theta)

    x = CYLINDER_CENTER_X + CYLINDER_RADIUS * np.cos(theta)
    y = CYLINDER_CENTER_Y + CYLINDER_RADIUS * np.sin(theta)

    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    t_t = torch.tensor(np.full((n_surface, 1), t_value), dtype=torch.float32)

    u, v, p = model(x_t, y_t, t_t)

    u_x = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y_t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x_t, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y_t, torch.ones_like(v), create_graph=True, retain_graph=True)[0]

    p_np = p.detach().numpy().flatten()
    u_x_np = u_x.detach().numpy().flatten()
    u_y_np = u_y.detach().numpy().flatten()
    v_x_np = v_x.detach().numpy().flatten()
    v_y_np = v_y.detach().numpy().flatten()

    tau_xx = (1.0 / RE) * 2 * u_x_np
    tau_yy = (1.0 / RE) * 2 * v_y_np
    tau_xy = (1.0 / RE) * (u_y_np + v_x_np)

    fx = -p_np * nx + tau_xx * nx + tau_xy * ny
    fy = -p_np * ny + tau_xy * nx + tau_yy * ny

    ds = CYLINDER_RADIUS * (2 * np.pi / n_surface)
    drag = np.sum(fx) * ds
    lift = np.sum(fy) * ds

    denom = 0.5 * INLET_U ** 2 * (2 * CYLINDER_RADIUS)
    c_d = drag / denom
    c_l = lift / denom

    return c_d, c_l


def compute_strouhal(model):
    """
    Sample lift coefficient over time, FFT to find the dominant shedding
    frequency, convert to Strouhal number St = f*D/U.
    Known Re=100 benchmark: St ~ 0.16-0.17 (see e.g. Williamson 1996 review).
    """
    t_values = np.linspace(T_MIN + 10.0, T_MAX, N_TIME_SAMPLES_FOR_FFT)
    dt = t_values[1] - t_values[0]

    c_l_series = []
    for t_val in t_values:
        _, c_l = compute_lift_drag(model, t_val)
        c_l_series.append(c_l)
    c_l_series = np.array(c_l_series)

    c_l_centered = c_l_series - np.mean(c_l_series)
    fft_vals = np.fft.rfft(c_l_centered)
    freqs = np.fft.rfftfreq(len(c_l_centered), d=dt)

    dominant_idx = np.argmax(np.abs(fft_vals[1:])) + 1
    dominant_freq = freqs[dominant_idx]

    D = 2 * CYLINDER_RADIUS
    strouhal = dominant_freq * D / INLET_U

    plt.figure(figsize=(10, 4))
    plt.plot(t_values, c_l_series)
    plt.xlabel("t")
    plt.ylabel("Lift coefficient C_L")
    plt.title(f"Lift coefficient over time (estimated St = {strouhal:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig("../outputs/cylinder_lift_timeseries.png", dpi=120)
    print("Saved plot to ../outputs/cylinder_lift_timeseries.png")
    plt.show()

    print(f"\nDominant shedding frequency: {dominant_freq:.4f}")
    print(f"Estimated Strouhal number: {strouhal:.4f}")
    print("Reference (literature, Re=100): St ~ 0.16-0.17")

    return strouhal, c_l_series, t_values


if __name__ == "__main__":
    model = load_model()

    print("Plotting vorticity snapshots over time...")
    plot_time_snapshots(model)

    print("\nComputing lift/drag and Strouhal number...")
    strouhal, c_l_series, t_values = compute_strouhal(model)