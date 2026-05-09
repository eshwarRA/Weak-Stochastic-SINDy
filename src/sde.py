# src/sde.py
# ─────────────────────────────────────────────────────────────────────────────
# True SDE definitions for the three benchmark systems and the scalar
# Euler–Maruyama integrator.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from src.config import DT, N, THETA_OU, SIGMA0_OU, SIGMA0_DW


# ── Ornstein–Uhlenbeck ────────────────────────────────────────────────────────
# dX = -θ X dt + σ₀ dW,   θ=1.0,  σ₀=0.7  →  a(x)=0.490

def b_ou(x: float) -> float:
    return -THETA_OU * x

def s_ou(x: float) -> float:        # noqa: ARG001
    return SIGMA0_OU


# ── Double-Well Langevin ──────────────────────────────────────────────────────
# dX = (X - X³) dt + σ₀ dW,   σ₀=0.5  →  a(x)=0.250

def b_dw(x: float) -> float:
    return x - x ** 3

def s_dw(x: float) -> float:        # noqa: ARG001
    return SIGMA0_DW


# ── Multiplicative Diffusion ──────────────────────────────────────────────────
# dX = -2X dt + 0.5√(1+X²) dW   →  a(x) = 0.25(1+x²)

def b_mult(x: float) -> float:
    return -2.0 * x

def s_mult(x: float) -> float:
    return 0.5 * np.sqrt(1.0 + x ** 2)


# ── Euler–Maruyama integrator ─────────────────────────────────────────────────

def euler_maruyama(
    b,
    sigma,
    x0: float = 0.0,
    dt: float = DT,
    n_steps: int = N,
) -> np.ndarray:
    """Scalar Euler–Maruyama integrator.

    Parameters
    ----------
    b, sigma : callables  drift and diffusion functions
    x0       : float      initial condition
    dt       : float      time step
    n_steps  : int        number of steps

    Returns
    -------
    x : ndarray of shape (n_steps + 1,)
    """
    x = np.empty(n_steps + 1)
    x[0] = float(x0)
    sqdt = np.sqrt(dt)
    xi   = np.random.randn(n_steps)
    for n in range(n_steps):
        xn     = x[n]
        x[n+1] = xn + float(b(xn)) * dt + float(sigma(xn)) * sqdt * xi[n]
    return x


def simulate_long(
    b,
    sigma,
    n_steps: int = 200_000,
    burn: int = 10_000,
    x0: float = 0.0,
    x_clip: float = 8.0,
    dt: float = DT,
) -> np.ndarray:
    """Long ergodic simulation with optional burn-in and state clipping.

    Used for autocorrelation estimation in Figure 4.
    """
    x = np.empty(n_steps + 1)
    x[0] = float(x0)
    sqdt = np.sqrt(dt)
    xi   = np.random.randn(n_steps)
    for n in range(n_steps):
        xn     = float(np.clip(x[n], -x_clip, x_clip))
        x[n+1] = xn + float(b(xn)) * dt + float(sigma(xn)) * sqdt * xi[n]
    return x[burn:]
