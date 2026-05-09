# scripts/fig4_autocorr.py
# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Empirical autocorrelation functions from long simulations of
# both the true and recovered SDEs, for all three benchmark systems.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.config import PUB_RC, BLUE, RED, DT, THETA_OU
from src.library import reconstruct
from src.sde import b_ou, s_ou, b_dw, s_dw, b_mult, s_mult, simulate_long

CACHE  = "results/run_cache.pkl"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

MAX_LAG = 600   # number of lags to display


def load_results() -> dict:
    if not os.path.exists(CACHE):
        raise FileNotFoundError(
            f"Cache not found at {CACHE}.\n"
            "Run `python reproduce_all.py` first."
        )
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def autocorr(x: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    x   = x - x.mean()
    ac  = np.correlate(x, x, mode="full")
    ac  = ac[len(ac) // 2:][:max_lag + 1]
    return ac / ac[0]


def make_fig4(results: dict) -> None:
    ou   = results["ou"]
    dw   = results["dw"]
    mult = results["mult"]

    np.random.seed(99)
    print("  Simulating long OU trajectories …")
    x_ou_true = simulate_long(b_ou, s_ou)
    x_ou_rec  = simulate_long(
        lambda x: float(reconstruct(ou["coef_b"],   np.atleast_1d(x))[0]),
        lambda x: float(np.sqrt(max(float(ou["coef_a"][0]), 1e-8))),
    )

    print("  Simulating long double-well trajectories …")
    x_dw_true = simulate_long(b_dw, s_dw)
    x_dw_rec  = simulate_long(
        lambda x: float(reconstruct(dw["coef_b"],   np.atleast_1d(x))[0]),
        lambda x: float(np.sqrt(max(float(dw["coef_a"][0]), 1e-8))),
    )

    print("  Simulating long multiplicative trajectories (500k steps) …")
    x_m_true  = simulate_long(b_mult, s_mult, n_steps=500_000, burn=20_000)
    x_m_rec   = simulate_long(
        lambda x: float(reconstruct(mult["coef_b"], np.atleast_1d(x))[0]),
        lambda x: float(np.sqrt(max(float(reconstruct(mult["coef_a"], np.atleast_1d(x))[0]), 1e-8))),
        n_steps=500_000, burn=20_000,
    )

    ac_ou_true = autocorr(x_ou_true);  ac_ou_rec = autocorr(x_ou_rec)
    ac_dw_true = autocorr(x_dw_true);  ac_dw_rec = autocorr(x_dw_rec)
    ac_m_true  = autocorr(x_m_true);   ac_m_rec  = autocorr(x_m_rec)
    lags       = np.arange(MAX_LAG + 1) * DT

    theta_sindy = float(-ou["coef_b"][1])
    err_pct     = abs(theta_sindy - THETA_OU) / THETA_OU * 100
    print(f"  OU: θ_true={THETA_OU:.3f}  θ_recovered={theta_sindy:.3f}  (err {err_pct:.1f}%)")

    panel_specs = [
        (ac_ou_true, ac_ou_rec,
         rf"OU  ($\hat{{\theta}}={theta_sindy:.3f}$, err {err_pct:.1f}\%)", True),
        (ac_dw_true, ac_dw_rec, "Double-well  (no closed-form AC)", False),
        (ac_m_true,  ac_m_rec,  "Multiplicative  (no closed-form AC)", False),
    ]

    with matplotlib.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            "Autocorrelation check: true SDE vs recovered model\n"
            "Long simulation from both true and recovered dynamics",
            fontsize=13, fontweight="bold", y=1.05,
        )
        for ax, (ac_t, ac_r, title, show_ana) in zip(axes, panel_specs):
            ax.plot(lags, ac_t, lw=1.4, alpha=0.75, color="#92c5de",
                    label="Empirical AC (true SDE)")
            ax.plot(lags, ac_r, lw=1.8, ls="--", color=RED,
                    label="Empirical AC (recovered)")
            if show_ana:
                ax.plot(lags, np.exp(-THETA_OU * lags), "k:", lw=1.6,
                        label=rf"Analytical $e^{{-\theta\tau}},\ \theta={THETA_OU:.2f}$")
            ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
            ax.set_xlim(0, lags[-1]); ax.set_ylim(-0.25, 1.05)
            ax.set_xlabel(r"Lag $\tau$"); ax.set_ylabel("Autocorrelation")
            ax.set_title(title, fontsize=11, pad=6)
            ax.legend(framealpha=0.75, fontsize=9)
            ax.grid(True, alpha=0.25, lw=0.5)

        plt.tight_layout(w_pad=3.0)
        path = os.path.join(OUTDIR, "fig4_autocorr.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    make_fig4(load_results())
