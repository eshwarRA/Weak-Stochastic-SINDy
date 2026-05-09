# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Global hyperparameters.  Every number here matches the paper exactly.
# Import this module everywhere rather than hard-coding constants.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# ── Simulation ────────────────────────────────────────────────────────────────
DT   = 0.002          # Euler–Maruyama time step
T    = 100.0          # trajectory horizon
N    = int(T / DT)   # steps per trajectory (50 000)
R    = 120            # independent trajectories per system

# ── Polynomial library ────────────────────────────────────────────────────────
DEG        = 4
K_DIM      = DEG + 1
FEAT_NAMES = ["1", "x", "x²", "x³", "x⁴"]
x_eval     = np.linspace(-3.0, 3.0, 401)   # evaluation grid for plots

# ── Spatial Gaussian kernels ──────────────────────────────────────────────────
M_CENTRES = 50      # number of kernel centres
H_OU_DW   = 0.22   # bandwidth for OU and double-well  (overlap ratio ≈ 2.2)
H_MULT    = 0.27   # wider bandwidth for multiplicative system (heavier tails)

# ── Sparse regression ─────────────────────────────────────────────────────────
ALPHA_GRID     = np.logspace(-8, -0.5, 60)   # LassoCV regularisation grid
CV_SPLITS      = 5                            # grouped K-fold folds
STLS_THR_OU_DW = 0.25   # STLSQ relative threshold (OU and double-well)
STLS_THR_MULT  = 0.30   # slightly higher for multiplicative (more collinearity)

# ── True SDE parameters ───────────────────────────────────────────────────────
THETA_OU  = 1.0   # OU mean-reversion rate
SIGMA0_OU = 0.7   # OU diffusion  → a(x) = 0.490
SIGMA0_DW = 0.5   # double-well diffusion  → a(x) = 0.250

# ── Random seeds (fixed for reproducibility) ─────────────────────────────────
SEED_OU   = 42
SEED_DW   = 123
SEED_MULT = 7

# ── Publication plot style ────────────────────────────────────────────────────
BLUE = "#2166ac"
RED  = "#d6604d"

PUB_RC = {
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":           12,
    "axes.titlesize":      12,
    "axes.labelsize":      12,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "legend.fontsize":     10,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.08,
    "axes.linewidth":      0.9,
    "lines.linewidth":     1.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "text.usetex":         False,
}
