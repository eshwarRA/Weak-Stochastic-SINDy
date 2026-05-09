# scripts/fig3_stationary_density.py
# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Analytical stationary densities (Fokker–Planck formula) for the
# true SDE and the recovered model, for all three benchmark systems.
#
# Densities are computed analytically — no Monte Carlo variance.
#   π(x) ∝ a(x)⁻¹ exp( 2 ∫₀ˣ b(y)/a(y) dy )
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.config import PUB_RC, BLUE, RED, SIGMA0_OU, SIGMA0_DW
from src.library import reconstruct
from src.sde import b_ou, b_dw, b_mult

CACHE  = "results/run_cache.pkl"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

xs = np.linspace(-4.5, 4.5, 800)   # evaluation grid for stationary densities


def load_results() -> dict:
    if not os.path.exists(CACHE):
        raise FileNotFoundError(
            f"Cache not found at {CACHE}.\n"
            "Run `python reproduce_all.py` first."
        )
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def fp_stationary(b_fn, a_fn, x_grid: np.ndarray) -> np.ndarray:
    """Fokker–Planck stationary density via numerical integration (eq. 3).

    π(x) ∝ a(x)⁻¹ exp( 2 ∫₀ˣ b(y)/a(y) dy )
    """
    dx    = x_grid[1] - x_grid[0]
    bvals = b_fn(x_grid); avals = a_fn(x_grid)
    ratio = bvals / (avals + 1e-12)

    # Trapezoid integration of b/a from 0 to x
    integ = np.zeros_like(x_grid)
    for i in range(1, len(x_grid)):
        integ[i] = integ[i - 1] + 0.5 * (ratio[i - 1] + ratio[i]) * dx

    log_pi  = 2.0 * integ - np.log(avals + 1e-12)
    log_pi -= log_pi.max()
    pi      = np.exp(log_pi)
    pi     /= np.trapezoid(pi, x_grid)
    return pi


def tv_distance(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * np.trapezoid(np.abs(p - q), x))


def make_fig3(results: dict) -> None:
    ou   = results["ou"]
    dw   = results["dw"]
    mult = results["mult"]

    # True densities
    pi_ou_true = fp_stationary(b_ou,   lambda x: SIGMA0_OU**2 * np.ones_like(x), xs)
    pi_dw_true = fp_stationary(b_dw,   lambda x: SIGMA0_DW**2 * np.ones_like(x), xs)
    pi_m_true  = fp_stationary(b_mult, lambda x: 0.25 * (1.0 + x**2),            xs)

    # Recovered densities
    pi_ou_hat = fp_stationary(
        lambda x: reconstruct(ou["coef_b"],   np.atleast_1d(x)),
        lambda x: np.full_like(np.atleast_1d(x), float(ou["coef_a"][0])), xs,
    )
    pi_dw_hat = fp_stationary(
        lambda x: reconstruct(dw["coef_b"],   np.atleast_1d(x)),
        lambda x: np.full_like(np.atleast_1d(x), float(dw["coef_a"][0])), xs,
    )
    pi_m_hat = fp_stationary(
        lambda x: reconstruct(mult["coef_b"], np.atleast_1d(x)),
        lambda x: np.clip(reconstruct(mult["coef_a"], np.atleast_1d(x)), 1e-6, None), xs,
    )

    tv_ou = tv_distance(pi_ou_true, pi_ou_hat, xs)
    tv_dw = tv_distance(pi_dw_true, pi_dw_hat, xs)
    tv_m  = tv_distance(pi_m_true,  pi_m_hat,  xs)
    print(f"TV distances — OU: {tv_ou:.4f}  DW: {tv_dw:.4f}  Mult: {tv_m:.4f}")

    panels = [
        (pi_ou_true, pi_ou_hat, "OU",             tv_ou),
        (pi_dw_true, pi_dw_hat, "Double-well",    tv_dw),
        (pi_m_true,  pi_m_hat,  "Multiplicative", tv_m),
    ]

    with matplotlib.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            "Stationary density: true SDE vs recovered model\n"
            r"Fokker--Planck: $\pi(x) \propto a(x)^{-1}"
            r"\exp\!\left(2\int_0^x \frac{b(y)}{a(y)}\,dy\right)$",
            fontsize=13, fontweight="bold", y=1.05,
        )
        for ax, pi_t, pi_h, label, tv in zip(axes, *zip(*panels)):
            ax.plot(xs, pi_t, lw=2.0, color=BLUE, label="True SDE")
            ax.plot(xs, pi_h, lw=1.8, ls="--", color=RED,
                    label=f"Recovered  (TV = {tv:.4f})")
            ax.fill_between(xs, pi_t, pi_h, alpha=0.15, color=RED, label="Discrepancy")
            ax.set_title(label, fontsize=13, pad=6)
            ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\pi(x)$")
            ax.set_xlim(-4.5, 4.5)
            ax.legend(framealpha=0.75, fontsize=9)
            ax.grid(True, alpha=0.25, lw=0.5)

        plt.tight_layout(w_pad=3.0)
        path = os.path.join(OUTDIR, "fig3_stationary_density.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    make_fig3(load_results())
