"""
Visualization + aerodynamic validation for the airfoil-flow PINN.

Produces:
  1. Velocity/pressure field snapshots around the airfoil
  2. Surface pressure distribution (Cp) along the airfoil chord
  3. Lift and drag coefficients, time-averaged after the initial
     transient, compared against thin-airfoil-theory's prediction:
     C_L ~ 2*pi*sin(AoA) for a symmetric airfoil at small angles.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import AirfoilPINN
from config import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, RE, INLET_U,
    AIRFOIL_CENTER_X, AIRFOIL_CENTER_Y, ANGLE_OF_ATTACK_DEG, CHORD,
)
from geometry import inside_airfoil, get_airfoil_polygon

MODEL_PATH = "../outputs/pinn_airfoil.pt"
N_GRID = 150
N_SURFACE_PTS = 300


def load_model():
    model = AirfoilPINN(hidden_layers=8, hidden_units=100)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def evaluate_snapshot(model, t_value, n=N_GRID):
    x = np.linspace(AIRFOIL_CENTER_X - 2, AIRFOIL_CENTER_X + 4, n)
    y = np.linspace(AIRFOIL_CENTER_Y - 2, AIRFOIL_CENTER_Y + 2, n)
    X, Y = np.meshgrid(x, y)

    x_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    t_t = torch.tensor(np.full_like(X.reshape(-1, 1), t_value), dtype=torch.float32)

    u, v, p = model(x_t, y_t, t_t)

    v_x = torch.autograd.grad(v, x_t, torch.ones_like(v), retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y_t, torch.ones_like(u), retain_graph=True)[0]
    vorticity = (v_x - u_y).detach().numpy().reshape(n, n)

    U = u.detach().numpy().reshape(n, n)
    V = v.detach().numpy().reshape(n, n)
    P = p.detach().numpy().reshape(n, n)

    mask = inside_airfoil(X.flatten(), Y.flatten()).reshape(n, n)
    U[mask] = np.nan
    V[mask] = np.nan
    P[mask] = np.nan
    vorticity[mask] = np.nan

    return X, Y, U, V, P, vorticity


def plot_field(model, t_value=15.0):
    X, Y, U, V, P, vort = evaluate_snapshot(model, t_value)
    speed = np.sqrt(U ** 2 + V ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im0 = axes[0].contourf(X, Y, speed, levels=60, cmap="viridis")
    axes[0].streamplot(X, Y, U, V, color="white", linewidth=0.6, density=1.5)
    x_poly, y_poly = get_airfoil_polygon()
    axes[0].fill(x_poly, y_poly, color="black")
    axes[0].set_title(f"Velocity magnitude + streamlines (t={t_value})")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, P, levels=60, cmap="coolwarm")
    axes[1].fill(x_poly, y_poly, color="black")
    axes[1].set_title(f"Pressure field (t={t_value})")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig("../outputs/airfoil_field.png", dpi=140)
    print("Saved plot to ../outputs/airfoil_field.png")
    plt.show()


def compute_lift_drag(model, t_value, n_surface=N_SURFACE_PTS):
    """
    Integrate pressure + viscous shear stress around the airfoil surface
    to get drag (Fx, aligned with inflow) and lift (Fy, perpendicular to
    inflow) coefficients at a given time.
    """
    x_poly, y_poly = get_airfoil_polygon(n_points_per_surface=n_surface // 2)

    dx = np.gradient(x_poly)
    dy = np.gradient(y_poly)
    tangent_len = np.sqrt(dx ** 2 + dy ** 2)
    nx = dy / tangent_len
    ny = -dx / tangent_len

    x_t = torch.tensor(x_poly.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_poly.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    t_t = torch.tensor(np.full((len(x_poly), 1), t_value), dtype=torch.float32)

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

    ds = tangent_len
    drag_x = np.sum(fx * ds)
    drag_y = np.sum(fy * ds)

    aoa_rad = np.radians(ANGLE_OF_ATTACK_DEG)
    drag = drag_x * np.cos(-aoa_rad) - drag_y * np.sin(-aoa_rad)
    lift = drag_x * np.sin(-aoa_rad) + drag_y * np.cos(-aoa_rad)

    denom = 0.5 * INLET_U ** 2 * CHORD
    c_d = drag / denom
    c_l = lift / denom

    return c_d, c_l


def validate_lift_coefficient(model, n_time_samples=30):
    """
    Average C_L over several time samples (after initial transient) and
    compare against thin-airfoil-theory: C_L ~ 2*pi*sin(AoA).
    """
    t_values = np.linspace(T_MIN + 10.0, T_MAX, n_time_samples)

    c_l_series = []
    c_d_series = []
    for t_val in t_values:
        c_d, c_l = compute_lift_drag(model, t_val)
        c_l_series.append(c_l)
        c_d_series.append(c_d)

    c_l_series = np.array(c_l_series)
    c_d_series = np.array(c_d_series)

    c_l_mean = c_l_series.mean()
    c_d_mean = c_d_series.mean()

    theory_c_l = 2 * np.pi * np.sin(np.radians(ANGLE_OF_ATTACK_DEG))

    print(f"\nPINN mean C_L (t={T_MIN+10:.0f}-{T_MAX:.0f}): {c_l_mean:.4f}")
    print(f"PINN mean C_D: {c_d_mean:.4f}")
    print(f"Thin-airfoil-theory C_L (2*pi*sin({ANGLE_OF_ATTACK_DEG}deg)): {theory_c_l:.4f}")
    print(f"Relative difference: {abs(c_l_mean - theory_c_l) / theory_c_l * 100:.1f}%")

    plt.figure(figsize=(10, 4))
    plt.plot(t_values, c_l_series, label="PINN C_L(t)")
    plt.axhline(theory_c_l, color="red", linestyle="--", label=f"Thin-airfoil theory ({theory_c_l:.3f})")
    plt.xlabel("t")
    plt.ylabel("Lift coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Lift coefficient over time vs. thin-airfoil-theory prediction")
    plt.savefig("../outputs/airfoil_lift_validation.png", dpi=140)
    print("Saved plot to ../outputs/airfoil_lift_validation.png")
    plt.show()

    return c_l_mean, c_d_mean, theory_c_l


if __name__ == "__main__":
    model = load_model()

    print("Plotting velocity/pressure field...")
    plot_field(model, t_value=15.0)

    print("\nValidating lift coefficient against thin-airfoil theory...")
    c_l_mean, c_d_mean, theory_c_l = validate_lift_coefficient(model)