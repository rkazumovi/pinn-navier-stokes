"""
Network architecture for the 2D unsteady airfoil-flow PINN.

Input:  (x, y, t)       -- 3 features (space + time)
Output: (u, v, p)       -- 3 features (velocity components + pressure)

Identical architecture to the cylinder-flow model -- the network doesn't
"know" the obstacle shape directly; geometry only enters through where
we sample points and enforce the no-slip boundary condition.
"""

import torch
import torch.nn as nn


class AirfoilPINN(nn.Module):
    def __init__(self, hidden_layers=8, hidden_units=100):
        super().__init__()

        layers = []
        in_features = 3  # x, y, t

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.Tanh())
            in_features = hidden_units

        layers.append(nn.Linear(in_features, 3))  # u, v, p

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        """
        x, y, t: tensors of shape (N, 1)
        returns u, v, p each of shape (N, 1)
        """
        inputs = torch.cat([x, y, t], dim=1)
        out = self.net(inputs)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p


if __name__ == "__main__":
    model = AirfoilPINN(hidden_layers=8, hidden_units=100)
    x = torch.rand(10, 1, requires_grad=True)
    y = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    u, v, p = model(x, y, t)
    print("u:", u.shape, "v:", v.shape, "p:", p.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print("total trainable params:", n_params)