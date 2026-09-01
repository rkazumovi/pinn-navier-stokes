# ✈️ Physics-Informed Neural Networks for 2D Navier-Stokes Aerodynamics

Three progressively harder Physics-Informed Neural Networks (PINNs) solving the incompressible Navier-Stokes equations from scratch in PyTorch — culminating in predicting lift generation around a NACA 0012 airfoil.

📂 **Repository:** [github.com/rkazumovi/pinn-navier-stokes](https://github.com/rkazumovi/pinn-navier-stokes)

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)

---

## Overview

| Stage | Physics | Domain | Validation |
|---|---|---|---|
| `src/` | Steady 2D lid-driven cavity flow | Unit square, Re=100 | Centerline velocity vs. Ghia, Ghia & Shin (1982) |
| `src_cylinder/` | Unsteady 2D flow past a cylinder | 20D × 8D channel, Re=100 | Strouhal number vs. literature (~0.16–0.17) |
| `src_airfoil/` | Unsteady 2D flow past a NACA 0012 airfoil | 20c × 8c channel, Re=100, AoA=5° | Lift coefficient vs. thin-airfoil theory ($C_L \approx 2\pi\sin(\text{AoA})$) |

Each stage is a self-contained folder with the same six-file structure:

```
config.py       -- physical/geometric constants
geometry.py     -- (airfoil only) NACA 4-digit shape + inside/outside test
sampling.py     -- collocation + boundary point generation
model.py        -- MLP: (x,y[,t]) -> (u,v,p)
physics.py      -- PDE residuals via autograd, verified against analytical derivatives
train.py        -- Adam + L-BFGS two-stage training loop
visualize.py    -- field plots + physical validation against a known benchmark
```

## Method

**Governing equations** (non-dimensional, incompressible Navier-Stokes):

```
continuity:   u_x + v_y = 0
momentum-x:   (u_t +) u*u_x + v*u_y = -p_x + (1/Re)*(u_xx + u_yy)
momentum-y:   (v_t +) u*v_x + v*v_y = -p_y + (1/Re)*(v_xx + v_yy)
```

(The `u_t`, `v_t` terms only appear in the unsteady cylinder/airfoil cases.)

**Network:** shared-trunk MLP with tanh activations (required for smooth 2nd derivatives through autograd) and Xavier initialization.
- Cavity: 8 hidden layers × 50 units, 2 inputs (x, y)
- Cylinder / Airfoil: 8 hidden layers × 100 units, 3 inputs (x, y, t)

All three output a shared final layer of (u, v, p) — velocity components and pressure.

**Loss:** weighted sum of PDE residual (interior collocation points) + boundary condition terms (no-slip on the solid object, inlet, far-field, outlet) + initial condition (unsteady cases only). Interior and boundary points are randomly resampled every 200 epochs during Adam training to avoid overfitting to a fixed point set.

For the unsteady cases, the initial-condition loss and the no-slip loss on the obstacle are weighted 5× higher than the other terms — getting $t=0$ right anchors the whole time evolution, and no-slip on the obstacle is critical for correct wake formation.

**Training:** two-stage optimization, 4,000 interior points and 200 points per boundary segment.
1. **Adam** (lr=1e-3, 5,000 epochs) for global descent from random initialization.
2. **L-BFGS** (strong-Wolfe line search, up to 2,000 iterations) fine-tuning on a fixed point batch, for precise final convergence. Adam alone consistently plateaued in testing; L-BFGS reliably dropped total loss by another 5–10x on top.

**Derivative verification:** before trusting any autograd-computed PDE residual, each `physics.py` checks the derivative machinery against a known test function with hand-derived analytical derivatives (first and second partials in x, y, and t). This caught would-be silent bugs (e.g. incorrect `create_graph` settings) before they could corrupt training.

## Results

- **Cavity flow (Re=100):** final loss ≈ 0.007 after Adam + L-BFGS; centerline u-velocity profile matches Ghia et al. (1982) reference points.
- **Cylinder flow (Re=100):** final loss ≈ 0.007; vorticity field shows alternating shed vortices (von Kármán street); measured Strouhal number compared against the ~0.16–0.17 literature range.
- **Airfoil flow (NACA 0012, Re=100, AoA=5°):** final loss ≈ 0.0007 (the best convergence of the three); measured lift coefficient compared against thin-airfoil theory's prediction of $C_L \approx 0.548$.

See `outputs/*.png` for field plots, lift/drag time series, and centerline/Strouhal validation plots.

## 🛠️ Built With

**Language**
- Python 3

**Core Libraries**
- [PyTorch](https://pytorch.org/) — model definition, autograd-based PDE differentiation, Adam and L-BFGS optimizers
- [NumPy](https://numpy.org/) — numerical operations
- [SciPy](https://scipy.org/) — scientific computing utilities
- [Matplotlib](https://matplotlib.org/) — field plots, validation plots, and time-series visualizations

**Concepts**
- Physics-Informed Machine Learning
- Computational Fluid Dynamics (CFD)
- Incompressible Navier-Stokes Equations
- Deep Learning
- Numerical Analysis
- Automatic Differentiation
- Scientific Computing

## Known Limitations / What a v2 Would Improve

- **Fixed loss weights.** All three stages use hand-picked weights (e.g. `W_IC=5`, `W_CYL`/`W_AIRFOIL=5`). Adaptive weighting schemes (e.g. gradient-normalized loss balancing) would likely improve convergence further, especially for the unsteady cases.
- **Outlet BC is a simplification.** True "do-nothing" outflow boundary conditions are more involved than the soft $p=0$ reference used here; fine for this domain size but worth revisiting for shorter channels.
- **Single Reynolds number per case.** No generalization across Re was attempted; each model is trained for one fixed Re.
- **2D only.** Real aerodynamic analysis is 3D; this project is scoped to 2D flow as the appropriate stepping stone before that jump.

## Requirements

```
torch
numpy
matplotlib
scipy
```

## Running

Each stage is run independently, in its own folder:

```bash
cd src            # or src_cylinder, or src_airfoil
python model.py       # sanity check: architecture + shapes
python physics.py     # sanity check: derivatives vs. analytical values
python sampling.py    # sanity check: point generation, geometry exclusion
python train.py       # full training run (Adam + L-BFGS), saves model to ../outputs
python visualize.py   # field plots + physical validation, reads from ../outputs
```
