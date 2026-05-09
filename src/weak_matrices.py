# src/weak_matrices.py
# ─────────────────────────────────────────────────────────────────────────────
# Weak-form matrix construction — Algorithm 1, lines 1–6 of the paper.
#
# For each trajectory r:
#   A_jk = Σ_n  K_j(X_n) f_k(X_n) Δt          (design matrix, eq. 11)
#   B_j  = Σ_n  K_j(X_n) ΔX_n                  (drift response,  eq. 12)
#   Q_j  = Σ_n  K_j(X_n) (ΔX_n)²               (diffusion response, eq. 13)
#
# All trajectories are stacked to form A_stack, B_stack, Q_stack.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from src.config import DT
from src.library import poly_library, spatial_kernel


def build_weak_matrices(
    trajectories: list,
    centres: np.ndarray,
    h: float,
    dt: float = DT,
) -> tuple:
    """Build stacked weak-form matrices from a list of trajectories.

    Parameters
    ----------
    trajectories : list of ndarray  each of shape (N+1,)
    centres      : ndarray (M,)     kernel centres
    h            : float            kernel bandwidth
    dt           : float            time step

    Returns
    -------
    A_stack  : ndarray (M*R, K)   stacked design matrix
    B_stack  : ndarray (M*R,)     stacked drift response
    Q_stack  : ndarray (M*R,)     stacked quadratic-variation response
    K_list   : list of (M, N) ndarrays   per-trajectory kernel matrices
    Th_list  : list of (N, K) ndarrays   per-trajectory feature matrices
    traj_id  : ndarray (M*R,)   trajectory index label (for grouped CV)
    """
    A_list, B_list, Q_list = [], [], []
    K_list, Th_list = [], []

    for x in trajectories:
        xL = x[:-1]           # left endpoints  X_{t_n}
        dx = np.diff(x)       # increments      ΔX_n

        Th = poly_library(xL)                   # (N, K)
        Km = spatial_kernel(xL, centres, h)     # (M, N)

        A_list.append((Km @ Th) * dt)           # (M, K)
        B_list.append(Km @ dx)                  # (M,)
        Q_list.append(Km @ (dx ** 2))           # (M,)
        K_list.append(Km)
        Th_list.append(Th)

    A_stack = np.vstack(A_list)
    B_stack = np.concatenate(B_list)
    Q_stack = np.concatenate(Q_list)
    traj_id = np.repeat(np.arange(len(trajectories)), len(centres))

    return A_stack, B_stack, Q_stack, K_list, Th_list, traj_id
