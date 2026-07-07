"""
Training loop for the 2D lid-driven cavity Navier-Stokes PINN.

Total loss = PDE residual loss + boundary condition loss + pressure reference loss

  L_pde  = MSE(continuity) + MSE(momentum_x) + MSE(momentum_y)
  L_bc   = MSE(u_pred - u_bc) + MSE(v_pred - v_bc)   summed over 4 walls
  L_p0   = MSE(p_pred_at_ref - 0)

  L_total = w_pde * L_pde + w_bc * L_bc + w_p0 * L_p0

Fixed weights to start (w_pde=1, w_bc=1, w_p0=1). If training stalls with
BC loss staying high while PDE loss drops fast (or vice versa), that's the
signal to increase the weight on the lagging term -- very common in PINNs,
not a sign of a bug.
"""

import torch
import torch.optim as optim

from model import NavierStokesPINN
from sampling import sample_interior_points, sample_boundary_points, get_pressure_reference_point
from physics import navier_stokes_residuals

# ---- hyperparameters ----
N_INTERIOR = 4000
N_BOUNDARY_PER_WALL = 200
EPOCHS = 5000
LR = 1e-3
RESAMPLE_EVERY = 200  # resample collocation points periodically to avoid overfitting to a fixed set

W_PDE = 1.0
W_BC = 1.0
W_P0 = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_bc_loss(model, boundaries):
    """Sum of MSE(u,v) mismatch over all 4 walls."""
    loss = 0.0
    for name, (x, y, u_bc, v_bc) in boundaries.items():
        u_pred, v_pred, _ = model(x, y)
        loss = loss + torch.mean((u_pred - u_bc) ** 2) + torch.mean((v_pred - v_bc) ** 2)
    return loss


def compute_pde_loss(model, x_int, y_int):
    continuity, mom_x, mom_y = navier_stokes_residuals(model, x_int, y_int)
    loss = torch.mean(continuity ** 2) + torch.mean(mom_x ** 2) + torch.mean(mom_y ** 2)
    return loss


def compute_pressure_ref_loss(model):
    x_ref, y_ref, p_ref = get_pressure_reference_point(device=DEVICE)
    _, _, p_pred = model(x_ref, y_ref)
    return torch.mean((p_pred - p_ref) ** 2)


def train():
    print(f"Training on device: {DEVICE}")

    model = NavierStokesPINN(hidden_layers=8, hidden_units=50).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    x_int, y_int = sample_interior_points(N_INTERIOR, device=DEVICE)
    boundaries = sample_boundary_points(N_BOUNDARY_PER_WALL, device=DEVICE)

    history = {"total": [], "pde": [], "bc": [], "p0": []}

    for epoch in range(1, EPOCHS + 1):
        if epoch % RESAMPLE_EVERY == 0:
            x_int, y_int = sample_interior_points(N_INTERIOR, device=DEVICE)
            boundaries = sample_boundary_points(N_BOUNDARY_PER_WALL, device=DEVICE)

        optimizer.zero_grad()

        loss_pde = compute_pde_loss(model, x_int, y_int)
        loss_bc = compute_bc_loss(model, boundaries)
        loss_p0 = compute_pressure_ref_loss(model)

        loss_total = W_PDE * loss_pde + W_BC * loss_bc + W_P0 * loss_p0

        loss_total.backward()
        optimizer.step()

        history["total"].append(loss_total.item())
        history["pde"].append(loss_pde.item())
        history["bc"].append(loss_bc.item())
        history["p0"].append(loss_p0.item())

        if epoch % 500 == 0 or epoch == 1:
            print(
                f"epoch {epoch:5d} | total {loss_total.item():.6f} "
                f"| pde {loss_pde.item():.6f} | bc {loss_bc.item():.6f} "
                f"| p0 {loss_p0.item():.6f}"
            )

    return model, history


def finetune_lbfgs(model, max_iter=500):
    """
    Polish the Adam-trained model with L-BFGS. Uses a single fixed batch
    of points (no resampling) since L-BFGS expects a stable loss surface
    across its internal line-search evaluations.
    """
    print("\nStarting L-BFGS fine-tuning...")

    x_int, y_int = sample_interior_points(N_INTERIOR, device=DEVICE)
    boundaries = sample_boundary_points(N_BOUNDARY_PER_WALL, device=DEVICE)

    optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=max_iter,
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    iteration_counter = {"n": 0}

    def closure():
        optimizer.zero_grad()
        loss_pde = compute_pde_loss(model, x_int, y_int)
        loss_bc = compute_bc_loss(model, boundaries)
        loss_p0 = compute_pressure_ref_loss(model)
        loss_total = W_PDE * loss_pde + W_BC * loss_bc + W_P0 * loss_p0
        loss_total.backward()

        iteration_counter["n"] += 1
        if iteration_counter["n"] % 50 == 0:
            print(
                f"L-BFGS iter {iteration_counter['n']:4d} | total {loss_total.item():.6f} "
                f"| pde {loss_pde.item():.6f} | bc {loss_bc.item():.6f} "
                f"| p0 {loss_p0.item():.6f}"
            )
        return loss_total

    optimizer.step(closure)

    final_pde = compute_pde_loss(model, x_int, y_int)
    final_bc = compute_bc_loss(model, boundaries)
    final_p0 = compute_pressure_ref_loss(model)
    print(
        f"\nL-BFGS finished | total {(final_pde + final_bc + final_p0).item():.6f} "
        f"| pde {final_pde.item():.6f} | bc {final_bc.item():.6f} | p0 {final_p0.item():.6f}"
    )

    return model


if __name__ == "__main__":
    model, history = train()

    model = finetune_lbfgs(model, max_iter=2000)

    import os
    os.makedirs("../outputs", exist_ok=True)
    torch.save(model.state_dict(), "../outputs/pinn_cavity.pt")
    print("\nModel saved to ../outputs/pinn_cavity.pt")