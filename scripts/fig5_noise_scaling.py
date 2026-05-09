# scripts/fig5_noise_scaling.py
# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Theoretical noise scaling — Weak Form vs Kramers–Moyal.
#
# PURELY ANALYTICAL — no regression, no simulation.
# Runtime: < 1 second.  No cache dependency.
#
# Derived directly from the variance expressions in Theorem 6:
#   KM noise:  σ_obs / Δt           (diverges as Δt → 0)
#   WF noise:  σ_obs / √(N h_eff)   (grows only as √Δt, remains bounded)
#   Advantage: (KM noise) / (WF noise) = √(N h_eff) / Δt  ∝  Δt^{-3/2}
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.config import PUB_RC, DT, T, H_OU_DW, SIGMA0_OU

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Effective kernel integration width  h_eff = √(π/2) · h
h_eff = np.sqrt(np.pi / 2) * H_OU_DW

dt_vals = np.logspace(-3, -0.5, 300)
N_vals  = (T / dt_vals).astype(int)

snr_levels = [
    (r"$\sigma_{\rm obs} = \sigma_{\rm signal}/5$  (SNR=5)",
     SIGMA0_OU / 5,  "#d6604d", "#f4a582"),
    (r"$\sigma_{\rm obs} = \sigma_{\rm signal}/10$  (SNR=10)",
     SIGMA0_OU / 10, "#2166ac", "#92c5de"),
    (r"$\sigma_{\rm obs} = \sigma_{\rm signal}/20$  (SNR=20)",
     SIGMA0_OU / 20, "#1a9641", "#a6d96a"),
]


def make_fig5() -> None:
    with matplotlib.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            "Theoretical noise scaling: Weak Form vs Kramers--Moyal\n"
            r"KM noise $\propto \sigma_{\rm obs}/\Delta t$ (diverges as $\Delta t\to 0$);"
            r"  WF noise $\propto \sigma_{\rm obs}/\sqrt{N h_{\rm eff}}$ ($\sqrt{\Delta t}$ growth)",
            fontsize=12, fontweight="bold", y=1.05,
        )

        # ── Panel 1: KM noise ─────────────────────────────────────────────────
        ax = axes[0]
        for label, sig, col_km, _ in snr_levels:
            ax.loglog(dt_vals, sig / dt_vals, color=col_km, lw=2.0, label=label)
        ax.axvline(DT, color="#555555", ls=":", lw=1.2)
        ax.set_xlabel(r"Time step $\Delta t$")
        ax.set_ylabel(r"KM noise magnitude  $\left(\sigma_{\rm obs}/\Delta t\right)$")
        ax.set_title(r"KM: noise diverges as $\Delta t \to 0$", pad=6)
        ax.legend(framealpha=0.8, fontsize=9)
        ax.grid(True, alpha=0.2, which="both")

        # ── Panel 2: WF noise ─────────────────────────────────────────────────
        ax = axes[1]
        for label, sig, _, col_wf in snr_levels:
            ax.loglog(dt_vals, sig / np.sqrt(N_vals * h_eff),
                      color=col_wf, lw=2.0, label=label)
        ax.axvline(DT, color="#555555", ls=":", lw=1.2)
        ax.set_xlabel(r"Time step $\Delta t$")
        ax.set_ylabel(
            r"WF effective noise  $\left(\sigma_{\rm obs}/\sqrt{N h_{\rm eff}}\right)$"
        )
        ax.set_title(r"WF: noise grows only as $\sqrt{\Delta t}$", pad=6)
        ax.legend(framealpha=0.8, fontsize=9)
        ax.grid(True, alpha=0.2, which="both")

        # ── Panel 3: SNR advantage ratio ──────────────────────────────────────
        ax        = axes[2]
        advantage = np.sqrt(N_vals * h_eff) / dt_vals   # ∝ Δt^{-3/2}

        ax.loglog(dt_vals, advantage, color="#4dac26", lw=2.5,
                  label="SNR advantage  (KM noise / WF noise)")
        ax.axhline(1, color="#888888", ls="--", lw=1.0, label="No advantage (ratio = 1)")
        ax.axvline(DT, color="#555555", ls=":", lw=1.2)

        # Annotate value at the paper's Δt = 0.002
        idx = np.argmin(np.abs(dt_vals - DT))
        adv = float(advantage[idx])
        ax.annotate(
            f"×{adv:.0f} at $\\Delta t={DT}$",
            xy=(DT, adv), xytext=(DT * 4, adv * 0.4),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
            fontsize=9, color="#333333",
        )

        # Reference Δt^{-3/2} slope line
        ref = float(advantage[150])
        ax.loglog(dt_vals, ref * (dt_vals / dt_vals[150]) ** (-1.5),
                  "k:", lw=1.0, alpha=0.5, label=r"$\propto \Delta t^{-3/2}$ (theory)")

        ax.set_xlabel(r"Time step $\Delta t$")
        ax.set_ylabel("SNR ratio  (KM noise / WF noise)")
        ax.set_title(r"WF advantage grows as $\Delta t^{-3/2}$ relative to KM", pad=6)
        ax.legend(framealpha=0.8, fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.2, which="both")

        plt.tight_layout(w_pad=2.5)
        path = os.path.join(OUTDIR, "fig5_noise_scaling.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")

    print(f"\nAt Δt={DT}, T={T}, h_eff={h_eff:.4f}:")
    print(f"  N = {int(T/DT):,} steps")
    for label, sig, *_ in snr_levels:
        km_n = sig / DT
        wf_n = sig / np.sqrt(int(T / DT) * h_eff)
        print(f"  KM={km_n:.2f}   WF={wf_n:.5f}   advantage={km_n/wf_n:.0f}×")


if __name__ == "__main__":
    make_fig5()
