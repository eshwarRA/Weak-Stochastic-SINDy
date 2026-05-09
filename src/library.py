# src/library.py
# ─────────────────────────────────────────────────────────────────────────────
# Polynomial feature library and spatial Gaussian test functions (eq. 7).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from src.config import DEG


def poly_library(x: np.ndarray, deg: int = DEG) -> np.ndarray:
    """Return the polynomial feature matrix [1, x, x², …, x^deg].

    Parameters
    ----------
    x   : array-like  state values, shape (n,)
    deg : int         maximum polynomial degree

    Returns
    -------
    Theta : ndarray of shape (n, deg+1)
    """
    x = np.asarray(x).ravel()
    return np.column_stack([x ** k for k in range(deg + 1)])


def reconstruct(coef: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial with given coefficients on x_grid."""
    return poly_library(x_grid) @ coef


def spatial_kernel(
    xL: np.ndarray,
    centres: np.ndarray,
    h: float,
) -> np.ndarray:
    """Spatial Gaussian kernel matrix (eq. 7).

    K[j, n] = exp( -|xL_n - c_j|² / 2h² )

    Parameters
    ----------
    xL      : ndarray (n,)   left-endpoint states
    centres : ndarray (M,)   kernel centres
    h       : float          bandwidth

    Returns
    -------
    K : ndarray of shape (M, n)
    """
    return np.exp(-0.5 * ((xL[None, :] - centres[:, None]) / h) ** 2)
