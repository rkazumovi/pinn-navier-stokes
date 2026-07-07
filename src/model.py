"""
Network architecture for the 2D Navier-Stokes PINN.

Input:  (x, y)          -- 2 features
Output: (u, v, p)       -- 3 features (velocity components + pressure)

A single MLP with shared hidden layers produces all three outputs from
its final linear layer. This is standard for coupled-PDE PINNs: the
outputs share learned features, which helps the network respect the
physical coupling between velocity and pressure enforced by the loss.
"""

import torch
import torch.nn as nn


class NavierStokesPINN(nn.Module):
    def __init__(self, hidden_layers=8, hidden_units=50):
        super().__init__()

        layers = []
        in_features = 2  # x, y

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.Tanh())
            in_features = hidden_units

        # final layer: 3 outputs, no activation (u, v, p unbounded)
        layers.append(nn.Linear(in_features, 3))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot init pairs well with tanh activations."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        """
        x, y: tensors of shape (N, 1)
        returns u, v, p each of shape (N, 1)
        """
        inputs = torch.cat([x, y], dim=1)
        out = self.net(inputs)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p


if __name__ == "__main__":
    # quick sanity check
    model = NavierStokesPINN(hidden_layers=8, hidden_units=50)
    x = torch.rand(10, 1, requires_grad=True)
    y = torch.rand(10, 1, requires_grad=True)
    u, v, p = model(x, y)
    print("u:", u.shape, "v:", v.shape, "p:", p.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print("total trainable params:", n_params)