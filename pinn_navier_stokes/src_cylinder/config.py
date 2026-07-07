"""
Configuration for 2D unsteady flow past a cylinder (von Karman vortex street).

Length scale: cylinder diameter D = 1.0
Domain: 20D long x 8D tall channel, cylinder centered vertically,
        placed 1/4 of the way from the inlet (5D from left wall).
Time domain: T = 20 convective time units (long enough to capture
             several vortex shedding cycles after initial transient).
"""

# --- Reynolds number ---
RE = 100.0

# --- spatial domain (channel) ---
X_MIN, X_MAX = 0.0, 20.0
Y_MIN, Y_MAX = 0.0, 8.0

# --- cylinder geometry ---
CYLINDER_CENTER_X = 5.0        # 1/4 of the way from inlet (5 / 20)
CYLINDER_CENTER_Y = 4.0        # vertically centered (8 / 2)
CYLINDER_RADIUS = 0.5          # diameter D = 1.0

# --- time domain ---
T_MIN, T_MAX = 0.0, 20.0

# --- inlet / far-field conditions ---
INLET_U = 1.0
INLET_V = 0.0