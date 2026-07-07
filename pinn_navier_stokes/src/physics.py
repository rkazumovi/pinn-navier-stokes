"""
Physics residuals for steady 2D incompressible Navier-Stokes.

Governing equations (non-dimensional):
  continuity:  u_x + v_y = 0
  momentum-x:  u*u_x + v*u_y = -p_x + (1/Re)*(u_xx + u_yy)
  momentum-y:  u*v_x + v*v_y = -p_y + (1/Re)*(v_xx + v_yy)

All derivatives are computed via torch.autograd.grad through the network,
NOT via finite differences. create_graph=True is required on every grad
call whose output will itself be differentiated again (i.e. all first
derivatives here, since we need second derivatives from them).
"""

import torch
from config import RE


def _grad(output, input_tensor):
    """
    Helper: d(output)/d(input_tensor), keeping the computation graph so
    we can differentiate again (needed for 2nd order derivatives).
    """
    return torch.autograd.grad(
        outputs=output,
        inputs=input_tensor,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def navier_stokes_residuals(model, x, y):
    """
    Compute the PDE residuals at points (x, y).
    x, y must have requires_grad=True (from sample_interior_points).

    Returns:
        continuity_residual, momentum_x_residual, momentum_y_residual
        (each shape (N, 1), should be driven to ~0 during training)
    """
    u, v, p = model(x, y)

    # first derivatives
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    v_x = _grad(v, x)
    v_y = _grad(v, y)
    p_x = _grad(p, x)
    p_y = _grad(p, y)

    # second derivatives
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x)
    v_yy = _grad(v_y, y)

    continuity = u_x + v_y

    momentum_x = (u * u_x + v * u_y) + p_x - (1.0 / RE) * (u_xx + u_yy)
    momentum_y = (u * v_x + v * v_y) + p_y - (1.0 / RE) * (v_xx + v_yy)

    return continuity, momentum_x, momentum_y


if __name__ == "__main__":
    # --- sanity check: compare autograd derivatives to finite differences ---
    torch.manual_seed(0)
    x = torch.tensor([[0.7]], requires_grad=True)
    y = torch.tensor([[0.3]], requires_grad=True)

    u = x**2 * y + y**3

    u_x = _grad(u, x)
    u_y = _grad(u, y)
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)

    # analytical derivatives of x^2*y + y^3:
    # u_x = 2xy, u_y = x^2 + 3y^2, u_xx = 2y, u_yy = 6y
    x_val, y_val = 0.7, 0.3
    print("u_x   autograd:", u_x.item(), " analytical:", 2 * x_val * y_val)
    print("u_y   autograd:", u_y.item(), " analytical:", x_val**2 + 3 * y_val**2)
    print("u_xx  autograd:", u_xx.item(), " analytical:", 2 * y_val)
    print("u_yy  autograd:", u_yy.item(), " analytical:", 6 * y_val)

    # --- now test with the actual model ---
    from model import NavierStokesPINN
    from sampling import sample_interior_points

    model = NavierStokesPINN(hidden_layers=4, hidden_units=20)
    x_int, y_int = sample_interior_points(5)
    cont, mom_x, mom_y = navier_stokes_residuals(model, x_int, y_int)
    print("\ncontinuity residual shape:", cont.shape)
    print("momentum_x residual shape:", mom_x.shape)
    print("momentum_y residual shape:", mom_y.shape)
    print("(values are meaningless before training, this just checks shapes/no errors)")