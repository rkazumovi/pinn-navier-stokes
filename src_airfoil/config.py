"""
Configuration for 2D unsteady flow past a NACA 0012 airfoil.

Chord length c = 1.0 (length scale).
Domain: 20c long x 8c tall channel, airfoil placed 1/4 from inlet,
        rotated by the angle of attack (AoA) relative to inflow.
Time domain: T = 20 convective time units (same as cylinder case).
"""

# --- Reynolds number ---
RE = 100.0

# --- spatial domain (channel), in chord lengths ---
X_MIN, X_MAX = 0.0, 20.0
Y_MIN, Y_MAX = 0.0, 8.0

# --- airfoil geometry ---
NACA_CODE = "0012"          # NACA 4-digit: 00 = symmetric, 12 = 12% thickness
CHORD = 1.0                 # chord length (length scale)
AIRFOIL_CENTER_X = 5.0      # leading edge reference x-position, 1/4 from inlet
AIRFOIL_CENTER_Y = 4.0      # vertically centered
ANGLE_OF_ATTACK_DEG = 5.0   # AoA in degrees

# --- time domain ---
T_MIN, T_MAX = 0.0, 20.0

# --- inlet / far-field conditions ---
INLET_U = 1.0
INLET_V = 0.0