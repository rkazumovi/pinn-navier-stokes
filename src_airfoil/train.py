"""
Training loop for 2D unsteady flow past a NACA airfoil.

Total loss = PDE residual + initial condition + airfoil no-slip
             + inlet + far-field + outlet pressure reference

  L_pde     = MSE(continuity) + MSE(momentum_x) + MSE(momentum_y)
  L_ic      = MSE(u_pred - u_ic) + MSE(v_pred - v_ic)         at t=0
  L_airfoil = MSE(u_pred) + MSE(v_pred)                       on airfoil surface (no-slip)
  L_inlet   = MSE(u_pred - INLET_U) + MSE(v_pred - INLET_V)   at x=X_MIN
  L_farfield= MSE(u_pred - INLET_U) + MSE(v_pred)             at y=Y_MIN/Y_MAX
  L_outlet  = MSE(p_pred - 0)                                 at x=X_MAX (soft reference)

This is a substantially harder optimization than cavity flow: 3D input
space, larger space-time volume, and vortex shedding is a delicate
pattern. Expect this to need many more epochs and likely loss-weight
tuning to converge well -- that's normal for this class of problem, not
a sign the code is broken.
"""

import torch
import torch.optim as optim

from model import AirfoilPINN
from sampling import (
    sample_interior_points,
    sample_airfoil_boundary,
    sample_inlet_points,
    sample_farfield_points,
    sample_outlet_points,
    sample_initial_condition_points,
)
from physics import navier_stokes_residuals

# ---- hyperparameters ----
N_INTERIOR = 4000
N_AIRFOIL = 200
N_INLET = 200
N_FARFIELD = 200
N_OUTLET = 200
N_IC = 2000

EPOCHS = 5000
LR = 1e-3
RESAMPLE_EVERY = 200

# loss weights -- starting point, likely needs tuning based on observed
# training behavior (see note above)
W_PDE = 1.0
W_IC = 5.0        # weighted higher: getting t=0 right anchors the whole time evolution
W_AIRFOIL = 5.0       # weighted higher: no-slip on the obstacle is critical for wake formation
W_INLET = 1.0
W_FARFIELD = 1.0
W_OUTLET = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_pde_loss(model, x_int, y_int, t_int):
    continuity, mom_x, mom_y = navier_stokes_residuals(model, x_int, y_int, t_int)
    return torch.mean(continuity ** 2) + torch.mean(mom_x ** 2) + torch.mean(mom_y ** 2)


def compute_ic_loss(model, x_ic, y_ic, t_ic, u_ic, v_ic):
    u_pred, v_pred, _ = model(x_ic, y_ic, t_ic)
    return torch.mean((u_pred - u_ic) ** 2) + torch.mean((v_pred - v_ic) ** 2)


def compute_airfoil_loss(model, x_c, y_c, t_c, u_c, v_c):
    u_pred, v_pred, _ = model(x_c, y_c, t_c)
    return torch.mean((u_pred - u_c) ** 2) + torch.mean((v_pred - v_c) ** 2)


def compute_inlet_loss(model, x_in, y_in, t_in, u_in, v_in):
    u_pred, v_pred, _ = model(x_in, y_in, t_in)
    return torch.mean((u_pred - u_in) ** 2) + torch.mean((v_pred - v_in) ** 2)


def compute_farfield_loss(model, x_ff, y_ff, t_ff, u_ff, v_ff):
    u_pred, v_pred, _ = model(x_ff, y_ff, t_ff)
    return torch.mean((u_pred - u_ff) ** 2) + torch.mean((v_pred - v_ff) ** 2)


def compute_outlet_loss(model, x_out, y_out, t_out, p_out):
    _, _, p_pred = model(x_out, y_out, t_out)
    return torch.mean((p_pred - p_out) ** 2)


def resample_all():
    x_int, y_int, t_int = sample_interior_points(N_INTERIOR, device=DEVICE)
    x_a, y_a, t_a, u_a, v_a = sample_airfoil_boundary(N_AIRFOIL, device=DEVICE)
    x_in, y_in, t_in, u_in, v_in = sample_inlet_points(N_INLET, device=DEVICE)
    x_ff, y_ff, t_ff, u_ff, v_ff = sample_farfield_points(N_FARFIELD, device=DEVICE)
    x_out, y_out, t_out, p_out = sample_outlet_points(N_OUTLET, device=DEVICE)
    x_ic, y_ic, t_ic, u_ic, v_ic = sample_initial_condition_points(N_IC, device=DEVICE)

    return {
        "interior": (x_int, y_int, t_int),
        "airfoil": (x_a, y_a, t_a, u_a, v_a),
        "inlet": (x_in, y_in, t_in, u_in, v_in),
        "farfield": (x_ff, y_ff, t_ff, u_ff, v_ff),
        "outlet": (x_out, y_out, t_out, p_out),
        "ic": (x_ic, y_ic, t_ic, u_ic, v_ic),
    }


def train():
    print(f"Training on device: {DEVICE}")

    model = AirfoilPINN(hidden_layers=8, hidden_units=100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pts = resample_all()

    for epoch in range(1, EPOCHS + 1):
        if epoch % RESAMPLE_EVERY == 0:
            pts = resample_all()

        optimizer.zero_grad()

        loss_pde = compute_pde_loss(model, *pts["interior"])
        loss_ic = compute_ic_loss(model, *pts["ic"])
        loss_airfoil = compute_airfoil_loss(model, *pts["airfoil"])
        loss_inlet = compute_inlet_loss(model, *pts["inlet"])
        loss_ff = compute_farfield_loss(model, *pts["farfield"])
        loss_outlet = compute_outlet_loss(model, *pts["outlet"])

        loss_total = (
            W_PDE * loss_pde
            + W_IC * loss_ic
            + W_AIRFOIL * loss_airfoil
            + W_INLET * loss_inlet
            + W_FARFIELD * loss_ff
            + W_OUTLET * loss_outlet
        )

        loss_total.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(
                f"epoch {epoch:5d} | total {loss_total.item():.6f} "
                f"| pde {loss_pde.item():.6f} | ic {loss_ic.item():.6f} "
                f"| airfoil {loss_airfoil.item():.6f} | inlet {loss_inlet.item():.6f} "
                f"| ff {loss_ff.item():.6f} | outlet {loss_outlet.item():.6f}"
            )

    return model


def finetune_lbfgs(model, max_iter=1000):
    """
    Polish the Adam-trained model with L-BFGS on a single fixed batch
    of points (all 6 loss terms, no resampling during L-BFGS).
    """
    print("\nStarting L-BFGS fine-tuning...")

    pts = resample_all()

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

        loss_pde = compute_pde_loss(model, *pts["interior"])
        loss_ic = compute_ic_loss(model, *pts["ic"])
        loss_airfoil = compute_airfoil_loss(model, *pts["airfoil"])
        loss_inlet = compute_inlet_loss(model, *pts["inlet"])
        loss_ff = compute_farfield_loss(model, *pts["farfield"])
        loss_outlet = compute_outlet_loss(model, *pts["outlet"])

        loss_total = (
            W_PDE * loss_pde
            + W_IC * loss_ic
            + W_AIRFOIL * loss_airfoil
            + W_INLET * loss_inlet
            + W_FARFIELD * loss_ff
            + W_OUTLET * loss_outlet
        )
        loss_total.backward()

        iteration_counter["n"] += 1
        if iteration_counter["n"] % 100 == 0:
            print(
                f"L-BFGS iter {iteration_counter['n']:4d} | total {loss_total.item():.6f} "
                f"| pde {loss_pde.item():.6f} | ic {loss_ic.item():.6f} "
                f"| airfoil {loss_airfoil.item():.6f} | inlet {loss_inlet.item():.6f} "
                f"| ff {loss_ff.item():.6f} | outlet {loss_outlet.item():.6f}"
            )
        return loss_total

    optimizer.step(closure)

    loss_pde = compute_pde_loss(model, *pts["interior"])
    loss_ic = compute_ic_loss(model, *pts["ic"])
    loss_airfoil = compute_airfoil_loss(model, *pts["airfoil"])
    loss_inlet = compute_inlet_loss(model, *pts["inlet"])
    loss_ff = compute_farfield_loss(model, *pts["farfield"])
    loss_outlet = compute_outlet_loss(model, *pts["outlet"])
    total = (
        W_PDE * loss_pde + W_IC * loss_ic + W_AIRFOIL * loss_airfoil
        + W_INLET * loss_inlet + W_FARFIELD * loss_ff + W_OUTLET * loss_outlet
    )
    print(
        f"\nL-BFGS finished | total {total.item():.6f} | pde {loss_pde.item():.6f} "
        f"| ic {loss_ic.item():.6f} | airfoil {loss_airfoil.item():.6f} "
        f"| inlet {loss_inlet.item():.6f} | ff {loss_ff.item():.6f} "
        f"| outlet {loss_outlet.item():.6f}"
    )

    return model


if __name__ == "__main__":
    model = train()

    model = finetune_lbfgs(model, max_iter=1000)

    import os
    os.makedirs("../outputs", exist_ok=True)
    torch.save(model.state_dict(), "../outputs/pinn_airfoil.pt")
    print("\nModel saved to ../outputs/pinn_airfoil.pt")