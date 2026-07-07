"""
Physics residuals for unsteady 2D incompressible Navier-Stokes
(flow past a NACA airfoil).

Governing equations (non-dimensional):
  continuity:  u_x + v_y = 0
  momentum-x:  u_t + u*u_x + v*u_y = -p_x + (1/Re)*(u_xx + u_yy)
  momentum-y:  v_t + u*v_x + v*v_y = -p_y + (1/Re)*(v_xx + v_yy)

Identical to the cylinder-flow physics -- the PDE doesn't change based
on obstacle shape, only where we evaluate it (handled in sampling.py).
"""

import torch
from config import RE


def _grad(output, input_tensor):
    """d(output)/d(input_tensor), graph kept for further differentiation."""
    return torch.autograd.grad(
        outputs=output,
        inputs=input_tensor,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def navier_stokes_residuals(model, x, y, t):
    """
    Compute the unsteady PDE residuals at points (x, y, t).
    x, y, t must have requires_grad=True.

    Returns:
        continuity_residual, momentum_x_residual, momentum_y_residual
    """
    u, v, p = model(x, y, t)

    u_x = _grad(u, x)
    u_y = _grad(u, y)
    u_t = _grad(u, t)
    v_x = _grad(v, x)
    v_y = _grad(v, y)
    v_t = _grad(v, t)
    p_x = _grad(p, x)
    p_y = _grad(p, y)

    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x)
    v_yy = _grad(v_y, y)

    continuity = u_x + v_y

    momentum_x = u_t + (u * u_x + v * u_y) + p_x - (1.0 / RE) * (u_xx + u_yy)
    momentum_y = v_t + (u * v_x + v * v_y) + p_y - (1.0 / RE) * (v_xx + v_yy)

    return continuity, momentum_x, momentum_y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.tensor([[0.7]], requires_grad=True)
    y = torch.tensor([[0.3]], requires_grad=True)
    t = torch.tensor([[0.4]], requires_grad=True)

    u = x**2 * y * t + y**3 + t**2

    u_x = _grad(u, x)
    u_y = _grad(u, y)
    u_t = _grad(u, t)
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)

    x_val, y_val, t_val = 0.7, 0.3, 0.4
    print("u_x   autograd:", u_x.item(), " analytical:", 2 * x_val * y_val * t_val)
    print("u_y   autograd:", u_y.item(), " analytical:", x_val**2 * t_val + 3 * y_val**2)
    print("u_t   autograd:", u_t.item(), " analytical:", x_val**2 * y_val + 2 * t_val)
    print("u_xx  autograd:", u_xx.item(), " analytical:", 2 * y_val * t_val)
    print("u_yy  autograd:", u_yy.item(), " analytical:", 6 * y_val)

    from model import AirfoilPINN
    from sampling import sample_interior_points

    model = AirfoilPINN(hidden_layers=4, hidden_units=20)
    x_int, y_int, t_int = sample_interior_points(5)
    cont, mom_x, mom_y = navier_stokes_residuals(model, x_int, y_int, t_int)
    print("\ncontinuity residual shape:", cont.shape)
    print("momentum_x residual shape:", mom_x.shape)
    print("momentum_y residual shape:", mom_y.shape)
    print("(values meaningless before training, this just checks shapes/no errors)")