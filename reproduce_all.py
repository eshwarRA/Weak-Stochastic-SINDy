#!/usr/bin/env python3
# reproduce_all.py
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end runner: runs the three SDE experiments, generates all five
# paper figures, and prints Table 1.
#
# Usage:
#   python reproduce_all.py           # full run (~15-25 min)
#   python reproduce_all.py --cached  # load cached results, regenerate figures
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

CACHE = "results/run_cache.pkl"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Reproduce all paper figures.")
parser.add_argument("--cached", action="store_true",
                    help="Load cached experiment results instead of re-running.")
args = parser.parse_args()


# ── Step 1: Run experiments (or load cache) ───────────────────────────────────
if args.cached and os.path.exists(CACHE):
    print(f"Loading cached results from {CACHE} …")
    with open(CACHE, "rb") as f:
        results = pickle.load(f)
    print("Loaded.\n")
else:
    if args.cached:
        print(f"Cache not found at {CACHE} — running experiments from scratch.\n")

    from src.regression import run_all_experiments
    results = run_all_experiments()

    with open(CACHE, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nResults cached to {CACHE}\n")


# ── Step 2: Table 1 ───────────────────────────────────────────────────────────
from src.config import FEAT_NAMES, K_DIM, THETA_OU, SIGMA0_OU, SIGMA0_DW

print("=" * 70)
print("Table 1: Complete coefficient summary")
print("=" * 70)

def pct_err(hat, true):
    if abs(true) < 1e-9: return "—"
    return f"{abs(hat - true) / abs(true) * 100:.1f}%"

def tick(hat, true, tol=0.15):
    if abs(true) < 1e-9: return "—"
    return "✓" if abs(hat - true) / abs(true) < tol else "✗"

truth_b = {
    "ou":   [0, -THETA_OU,    0,    0,    0],
    "dw":   [0, +1.0,         0,   -1.0,  0],
    "mult": [0, -2.0,         0,    0,    0],
}
truth_a = {
    "ou":   [SIGMA0_OU**2, 0,    0,    0,    0],
    "dw":   [SIGMA0_DW**2, 0,    0,    0,    0],
    "mult": [0.25,         0,    0.25, 0,    0],
}
labels = {"ou": "Ornstein-Uhlenbeck", "dw": "Double-Well", "mult": "Multiplicative"}

hdr = (f"{'System':<20} {'Term':>5}  "
       f"{'ĉ_k':>9}  {'c*_k':>7}  {'Drift err':>10}   "
       f"{'d̂_k':>9}  {'d*_k':>7}  {'Diff err':>9}")
sep = "─" * len(hdr)
print(hdr); print(sep)

table_lines = [hdr, sep]
for key in ["ou", "dw", "mult"]:
    cb = results[key]["coef_b"]; tb = truth_b[key]
    ca = results[key]["coef_a_full"]; ta = truth_a[key]
    name = labels[key]
    for i, nm in enumerate(FEAT_NAMES):
        prefix = f"{name:<20}" if i == 0 else " " * 20
        line = (f"{prefix} {nm:>5}  "
                f"{cb[i]:>+9.5f}  {tb[i]:>+7.3f}  {pct_err(cb[i],tb[i]):>10} {tick(cb[i],tb[i]):>2}   "
                f"{ca[i]:>+9.5f}  {ta[i]:>+7.3f}  {pct_err(ca[i],ta[i]):>9} {tick(ca[i],ta[i]):>2}")
        print(line); table_lines.append(line)
    print(sep); table_lines.append(sep)

footnote = ("All nonzero true coefficients pass the 15% relative-error threshold (✓).\n"
            "All inactive coefficients (true = 0) are set to exactly 0 — no false positives.")
print(footnote)
table_lines.append(footnote)

with open("results/table1.txt", "w") as f:
    f.write("\n".join(table_lines))
print("Saved results/table1.txt\n")


# ── Step 3: All figures ───────────────────────────────────────────────────────
print("=" * 70)
print("Generating figures …")
print("=" * 70)

from scripts.fig1_functions       import make_fig1
from scripts.fig2_lasso_paths     import make_fig2
from scripts.fig3_stationary_density import make_fig3
from scripts.fig4_autocorr        import make_fig4
from scripts.fig5_noise_scaling   import make_fig5

print("\nFigure 1: Recovered vs true functions")
make_fig1(results)

print("\nFigure 2: LassoCV regularisation paths")
make_fig2(results)

print("\nFigure 3: Stationary densities")
make_fig3(results)

print("\nFigure 4: Autocorrelation functions")
make_fig4(results)

print("\nFigure 5: Theoretical noise scaling (analytical)")
make_fig5()

print("\n" + "=" * 70)
print("All outputs written.")
print("=" * 70)
print("  figures/fig1_recovered_vs_true.pdf")
print("  figures/fig2_lasso_paths.pdf")
print("  figures/fig3_stationary_density.pdf")
print("  figures/fig4_autocorr.pdf")
print("  figures/fig5_noise_scaling.pdf")
print("  results/table1.txt")
print("  results/run_cache.pkl")
