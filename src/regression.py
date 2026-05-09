# src/regression.py
# ─────────────────────────────────────────────────────────────────────────────
# Sparse regression pipeline (Algorithm 1, lines 7–10) and the high-level
# run_experiment() convenience function.
#
# Pipeline:  LassoCV (grouped K-fold by trajectory)
#            → OLS debias on selected support
#            → STLSQ to prune near-zero residuals
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold

from src.config import (
    ALPHA_GRID, CV_SPLITS, DT, M_CENTRES, R,
    STLS_THR_OU_DW, STLS_THR_MULT,
    SEED_OU, SEED_DW, SEED_MULT,
    H_OU_DW, H_MULT,
)
from src.sde import euler_maruyama
from src.weak_matrices import build_weak_matrices


# ── Core sparse regression ────────────────────────────────────────────────────

def fit_sparse_weak(
    A: np.ndarray,
    y: np.ndarray,
    traj_id: np.ndarray,
    alpha_grid: np.ndarray = ALPHA_GRID,
    n_splits: int = CV_SPLITS,
    stls_thr: float = 0.25,
    max_stls_iter: int = 20,
) -> tuple:
    """LassoCV + OLS debias + STLSQ sparse regression.

    Solves:  min_c  ‖Ac − y‖²  +  λ‖c‖₁
    with λ chosen by grouped K-fold CV (folds partitioned by trajectory index),
    followed by OLS debiasing on the selected support and iterated STLSQ.

    Parameters
    ----------
    A         : ndarray (m, K)   design matrix (columns are normalised internally)
    y         : ndarray (m,)     response vector
    traj_id   : ndarray (m,)     trajectory index for each row (for grouped CV)
    alpha_grid: ndarray          regularisation values to search
    n_splits  : int              number of CV folds
    stls_thr  : float            STLSQ relative threshold
    max_stls_iter : int          maximum STLSQ iterations

    Returns
    -------
    coef   : ndarray (K,)   estimated sparse coefficient vector
    lasso  : LassoCV        fitted LassoCV object (for path plots)
    supp   : ndarray        indices of the final active support
    """
    K = A.shape[1]

    # Column-normalise to prevent scale bias in LASSO
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    A_norm    = A / col_norms[None, :]

    # Grouped K-fold cross-validation by trajectory
    n_grp  = len(np.unique(traj_id))
    gkf    = GroupKFold(n_splits=min(n_splits, n_grp))
    cv_sp  = list(gkf.split(A_norm, y, traj_id))

    lasso = LassoCV(
        alphas=alpha_grid, cv=cv_sp, fit_intercept=False,
        max_iter=100_000, tol=1e-6, n_jobs=-1,
    ).fit(A_norm, y)

    # OLS debias on full support, then STLSQ
    coef, *_ = lstsq(A, y, rcond=None)
    supp     = np.arange(K)
    prev     = None

    for _ in range(max_stls_iter):
        dom = np.abs(coef).max()
        if dom == 0:
            break
        new_supp = np.where(np.abs(coef) >= stls_thr * dom)[0]
        if len(new_supp) == 0:
            new_supp = np.array([np.abs(coef).argmax()])
        if prev is not None and set(new_supp) == set(prev):
            break
        prev = set(new_supp)
        coef = np.zeros(K)
        sol, *_ = lstsq(A[:, new_supp], y, rcond=None)
        coef[new_supp] = sol
        supp = new_supp

    return coef, lasso, supp


# ── Drift-bias correction (sec. 3.5) ─────────────────────────────────────────

def bias_correct_Q(
    Q_stack: np.ndarray,
    K_list: list,
    Th_list: list,
    coef_b: np.ndarray,
    dt: float = DT,
) -> np.ndarray:
    """Subtract the O(Δt²) drift-squared bias from Q (eq. 17).

    E[(ΔX_n)²] = a(X_n)Δt  +  b(X_n)² Δt²  + O(Δt^{5/2})

    The second term is estimated using the already-fitted drift coef_b and
    subtracted before solving the diffusion regression.
    """
    corr = []
    for Km, Th in zip(K_list, Th_list):
        b_pw = Th @ coef_b                          # b̂(X_{t_n}) at each step
        corr.append((Km @ (b_pw ** 2)) * dt * dt)  # K_j · b̂² · Δt²
    return Q_stack - np.concatenate(corr)


# ── High-level experiment runner ──────────────────────────────────────────────

def run_experiment(
    b_func,
    sigma_func,
    seed: int,
    x_centres_range: tuple,
    h: float,
    stls_thr: float,
    const_diffusion: bool = True,
) -> dict:
    """Simulate R trajectories and identify drift + diffusion.

    Parameters
    ----------
    b_func, sigma_func : callables   true drift and diffusion
    seed               : int         RNG seed for trajectory generation
    x_centres_range    : (lo, hi)    range for M kernel centres
    h                  : float       kernel bandwidth
    stls_thr           : float       STLSQ threshold
    const_diffusion    : bool        if True fit only the constant term of a(x)

    Returns
    -------
    dict with keys:
        coef_b, coef_a   — estimated coefficient vectors
        lasso_b, lasso_a — LassoCV objects (for path plots)
    """
    np.random.seed(seed)
    trajs   = [
        euler_maruyama(b_func, sigma_func, x0=np.random.uniform(-3.0, 3.0))
        for _ in range(R)
    ]
    centres = np.linspace(*x_centres_range, M_CENTRES)
    A, B, Q, K_list, Th_list, traj_id = build_weak_matrices(trajs, centres, h)

    # Step 1: drift
    coef_b, lasso_b, _ = fit_sparse_weak(A, B, traj_id, stls_thr=stls_thr)

    # Step 2: diffusion (with drift-bias correction, eq. 17)
    Q_corr = bias_correct_Q(Q, K_list, Th_list, coef_b)
    A_fit  = A[:, :1] if const_diffusion else A   # scalar const vs full poly
    coef_a, lasso_a, _ = fit_sparse_weak(A_fit, Q_corr, traj_id, stls_thr=stls_thr)

    return {
        "coef_b": coef_b, "coef_a": coef_a,
        "lasso_b": lasso_b, "lasso_a": lasso_a,
    }


def run_all_experiments() -> dict:
    """Run all three benchmark experiments and return a results dict."""
    from src.sde import b_ou, s_ou, b_dw, s_dw, b_mult, s_mult
    from src.config import SIGMA0_OU, SIGMA0_DW, K_DIM

    print("Running Experiment 1: Ornstein–Uhlenbeck  (seed=%d) …" % SEED_OU)
    ou = run_experiment(b_ou, s_ou, seed=SEED_OU,
                        x_centres_range=(-2.5, 2.5), h=H_OU_DW,
                        stls_thr=STLS_THR_OU_DW, const_diffusion=True)

    print("Running Experiment 2: Double-Well  (seed=%d) …" % SEED_DW)
    dw = run_experiment(b_dw, s_dw, seed=SEED_DW,
                        x_centres_range=(-2.5, 2.5), h=H_OU_DW,
                        stls_thr=STLS_THR_OU_DW, const_diffusion=True)

    print("Running Experiment 3: Multiplicative Diffusion  (seed=%d) …" % SEED_MULT)
    mult = run_experiment(b_mult, s_mult, seed=SEED_MULT,
                          x_centres_range=(-2.8, 2.8), h=H_MULT,
                          stls_thr=STLS_THR_MULT, const_diffusion=False)

    # Pad constant-diffusion coef vectors to full length K for uniform use
    coef_a_ou_full  = np.zeros(K_DIM); coef_a_ou_full[0]  = ou["coef_a"][0]
    coef_a_dw_full  = np.zeros(K_DIM); coef_a_dw_full[0]  = dw["coef_a"][0]

    return {
        "ou":   {**ou,  "coef_a_full": coef_a_ou_full},
        "dw":   {**dw,  "coef_a_full": coef_a_dw_full},
        "mult": {**mult, "coef_a_full": mult["coef_a"]},
    }
