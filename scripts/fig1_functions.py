# scripts/fig1_functions.py
# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Recovered vs. true drift b(x) and diffusion a(x) functions
# for all three benchmark systems.
#
# Usage (after reproduce_all.py has been run once):
#   python scripts/fig1_functions.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.config import PUB_RC, BLUE, RED, x_eval, SIGMA0_OU, SIGMA0_DW
from src.library import poly_library, reconstruct
from src.sde import b_ou, b_dw, b_mult, s_mult

CACHE  = "results/run_cache.pkl"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)


def load_results() -> dict:
    if not os.path.exists(CACHE):
        raise FileNotFoundError(
            f"Cache not found at {CACHE}.\n"
            "Run `python reproduce_all.py` first to generate it."
        )
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def mre(ytrue: np.ndarray, yhat: np.ndarray) -> float:
    """Mean relative error (%)."""
    return float(
        np.mean(np.abs(ytrue - yhat)) / (np.mean(np.abs(ytrue)) + 1e-9) * 100
    )


def ylim_pad(ytrue: np.ndarray, yhat: np.ndarray, frac: float = 0.3) -> list:
    lo   = min(ytrue.min(), yhat.min())
    hi   = max(ytrue.max(), yhat.max())
    span = max(hi - lo, 1e-6)
    return [lo - frac * span, hi + frac * span]


def make_fig1(results: dict) -> None:
    ou   = results["ou"]
    dw   = results["dw"]
    mult = results["mult"]

    # True function curves
    b_true_ou = b_ou(x_eval);   a_true_ou = SIGMA0_OU**2 * np.ones_like(x_eval)
    b_true_dw = b_dw(x_eval);   a_true_dw = SIGMA0_DW**2 * np.ones_like(x_eval)
    b_true_m  = b_mult(x_eval); a_true_m  = 0.25 * (1.0 + x_eval**2)

    # Recovered function curves
    b_hat_ou = reconstruct(ou["coef_b"],   x_eval); a_hat_ou = ou["coef_a"][0]  * np.ones_like(x_eval)
    b_hat_dw = reconstruct(dw["coef_b"],   x_eval); a_hat_dw = dw["coef_a"][0]  * np.ones_like(x_eval)
    b_hat_m  = reconstruct(mult["coef_b"], x_eval); a_hat_m  = reconstruct(mult["coef_a"], x_eval)

    panels = [
        # (ax_pos, ytrue, yhat, title, ylabel)
        ((0, 0), b_true_ou, b_hat_ou, r"OU drift  $b(x)=-\theta x$",               r"$b(x)$"),
        ((1, 0), a_true_ou, a_hat_ou, r"OU diffusion  $a(x)=\sigma_0^2$",          r"$a(x)$"),
        ((0, 1), b_true_dw, b_hat_dw, r"Double-well drift  $b(x)=x-x^3$",          r"$b(x)$"),
        ((1, 1), a_true_dw, a_hat_dw, r"Double-well diffusion  $a(x)=\sigma_0^2$", r"$a(x)$"),
        ((0, 2), b_true_m,  b_hat_m,  r"Multiplicative drift  $b(x)=-2x$",         r"$b(x)$"),
        ((1, 2), a_true_m,  a_hat_m,  r"Multiplicative diffusion  $a(x)=0.25(1+x^2)$", r"$a(x)$"),
    ]

    with matplotlib.rc_context(PUB_RC):
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(
            "Sparse weak-form discovery of stochastic generators\n"
            r"Test functions: $K_j(x) = \exp\!\left(-|x-x_j|^2/2h^2\right)$",
            fontsize=14, fontweight="bold", y=1.01,
        )
        for (r_idx, c_idx), ytrue, yhat, title, ylabel in panels:
            ax  = axes[r_idx, c_idx]
            err = mre(ytrue, yhat)
            ax.plot(x_eval, ytrue, lw=2.0, color=BLUE, label="True")
            ax.plot(x_eval, yhat,  lw=1.8, ls="--", color=RED, label="Recovered")
            ax.set_title(f"{title}\n(mean rel. err. {err:.1f}%)", fontsize=11, pad=6)
            ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel)
            ax.set_ylim(ylim_pad(ytrue, yhat))
            ax.legend(handlelength=1.8, framealpha=0.75)
            ax.grid(True, alpha=0.25, lw=0.5)

        plt.tight_layout(h_pad=3.5, w_pad=2.5)
        path = os.path.join(OUTDIR, "fig1_recovered_vs_true.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    make_fig1(load_results())
