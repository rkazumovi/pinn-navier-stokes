"""
Visualize the trained PINN solution: velocity magnitude, streamlines,
and pressure field over the unit square.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import NavierStokesPINN
from config import X_MIN, X_MAX, Y_MIN, Y_MAX

MODEL_PATH = "../outputs/pinn_cavity.pt"
N_GRID = 100


def load_model():
    model = NavierStokesPINN(hidden_layers=8, hidden_units=50)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def evaluate_on_grid(model, n=N_GRID):
    x = np.linspace(X_MIN, X_MAX, n)
    y = np.linspace(Y_MIN, Y_MAX, n)
    X, Y = np.meshgrid(x, y)

    x_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)

    with torch.no_grad():
        u, v, p = model(x_t, y_t)

    U = u.numpy().reshape(n, n)
    V = v.numpy().reshape(n, n)
    P = p.numpy().reshape(n, n)

    return X, Y, U, V, P


def plot_fields(X, Y, U, V, P):
    speed = np.sqrt(U ** 2 + V ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].contourf(X, Y, speed, levels=50, cmap="viridis")
    axes[0].streamplot(X, Y, U, V, color="white", linewidth=0.5, density=1.2)
    axes[0].set_title("Velocity magnitude + streamlines")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, U, levels=50, cmap="RdBu_r")
    axes[1].set_title("u (x-velocity)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, Y, P, levels=50, cmap="coolwarm")
    axes[2].set_title("Pressure")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("../outputs/cavity_fields.png", dpi=150)
    print("Saved plot to ../outputs/cavity_fields.png")
    plt.show()


def plot_centerline_u(X, Y, U):
    """
    u-velocity along the vertical centerline (x=0.5), compared against
    Ghia et al. (1982) benchmark values for Re=100.
    """
    # Ghia et al. 1982, Re=100, u along vertical centerline x=0.5
    ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                        0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                        0.9688, 0.9766, 1.0000])
    ghia_u = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                        -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                        0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

    # extract PINN u along x=0.5 column (nearest grid column)
    x_vals = X[0, :]
    col_idx = np.argmin(np.abs(x_vals - 0.5))
    y_col = Y[:, col_idx]
    u_col = U[:, col_idx]

    plt.figure(figsize=(6, 6))
    plt.plot(u_col, y_col, "b-", label="PINN prediction")
    plt.plot(ghia_u, ghia_y, "ro", label="Ghia et al. 1982 (Re=100)")
    plt.xlabel("u")
    plt.ylabel("y")
    plt.title("u-velocity along vertical centerline (x=0.5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("../outputs/centerline_comparison.png", dpi=150)
    print("Saved plot to ../outputs/centerline_comparison.png")
    plt.show()


if __name__ == "__main__":
    model = load_model()
    X, Y, U, V, P = evaluate_on_grid(model)
    plot_fields(X, Y, U, V, P)
    plot_centerline_u(X, Y, U)