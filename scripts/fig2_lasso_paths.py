# scripts/fig2_lasso_paths.py
# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: LassoCV regularisation paths for all six sub-problems
# (three drifts and three diffusions).
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import matplotlib
import matplotlib.pyplot as plt

from src.config import PUB_RC, BLUE, RED

CACHE  = "results/run_cache.pkl"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)


def load_results() -> dict:
    if not os.path.exists(CACHE):
        raise FileNotFoundError(
            f"Cache not found at {CACHE}.\n"
            "Run `python reproduce_all.py` first."
        )
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def plot_lasso_path(ax, lasso, title: str) -> None:
    mean_mse = lasso.mse_path_.mean(axis=1)
    ax.semilogx(lasso.alphas_, mean_mse, "o-", ms=3, lw=1.4, color=BLUE)
    ax.axvline(
        lasso.alpha_, color=RED, ls="--", lw=1.4,
        label=r"$\alpha^* = $" + f"{lasso.alpha_:.1e}",
    )
    ax.invert_xaxis()
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("CV MSE")
    ax.set_title(title, pad=6)
    ax.legend(framealpha=0.75)
    ax.grid(True, alpha=0.25, lw=0.5)


def make_fig2(results: dict) -> None:
    ou   = results["ou"]
    dw   = results["dw"]
    mult = results["mult"]

    panels = [
        # (ax position, lasso object, title)
        ((0, 0), ou["lasso_b"],   "OU drift"),
        ((1, 0), ou["lasso_a"],   "OU diffusion"),
        ((0, 1), dw["lasso_b"],   "Double-well drift"),
        ((1, 1), dw["lasso_a"],   "Double-well diffusion"),
        ((0, 2), mult["lasso_b"], "Multiplicative drift"),
        ((1, 2), mult["lasso_a"], "Multiplicative diffusion"),
    ]

    with matplotlib.rc_context(PUB_RC):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(
            "Weak-form LassoCV regularisation paths",
            fontsize=14, fontweight="bold", y=1.01,
        )
        for (r_idx, c_idx), lasso, title in panels:
            plot_lasso_path(axes[r_idx, c_idx], lasso, title)

        plt.tight_layout(h_pad=3.5, w_pad=2.5)
        path = os.path.join(OUTDIR, "fig2_lasso_paths.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    make_fig2(load_results())
