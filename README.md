# Sparse Weak-Form Discovery of Stochastic Generators

Reproducibility package for the paper:

> **Sparse Weak-Form Discovery of Stochastic Generators**  
> Eshwar R A and Gajanan V. Honnavar  
> PES University (EC Campus), Bengaluru  
> arXiv: https://arxiv.org/abs/2603.20904

---

## What this reproduces

| Output | Description |
|--------|-------------|
| `figures/fig1_recovered_vs_true.pdf` | Recovered vs. true drift and diffusion functions (Figure 1) |
| `figures/fig2_lasso_paths.pdf` | LassoCV regularisation paths (Figure 2) |
| `figures/fig3_stationary_density.pdf` | Analytical stationary densities — true vs recovered (Figure 3) |
| `figures/fig4_autocorr.pdf` | Autocorrelation functions — true vs recovered SDE (Figure 4) |
| `figures/fig5_noise_scaling.pdf` | Theoretical noise scaling: Weak Form vs Kramers–Moyal (Figure 5) |
| `results/table1.txt` | Complete coefficient recovery table (Table 1) |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ required. Tested on 3.10 and 3.12.

---

## Reproducing all results

```bash
python reproduce_all.py
```

This runs the three SDE experiments sequentially, then generates all figures and
Table 1. Figures are saved to `figures/` and the coefficient table to `results/`.

**Expected runtime:** 15–25 minutes on a standard laptop (dominated by 120 × 3
Euler–Maruyama simulations and grouped LassoCV on each system).

To skip the slow experiment runs and use cached results:

```bash
python reproduce_all.py --cached     # loads results/run_cache.pkl if it exists
```

---

## Running individual figure scripts

After `reproduce_all.py` has generated `results/run_cache.pkl`, each figure
can be regenerated independently in seconds:

```bash
python scripts/fig1_functions.py
python scripts/fig2_lasso_paths.py
python scripts/fig3_stationary_density.py
python scripts/fig4_autocorr.py
python scripts/fig5_noise_scaling.py   # purely analytical — no cache needed
```

---

## Project layout

```
WeakStochasticSINDy/
├── reproduce_all.py          # end-to-end runner
├── requirements.txt
├── src/
│   ├── config.py             # global hyperparameters
│   ├── sde.py                # SDE definitions + Euler–Maruyama integrator
│   ├── library.py            # polynomial library + spatial Gaussian kernels
│   ├── weak_matrices.py      # weak-form matrix construction (Algorithm 1, lines 1–6)
│   └── regression.py         # LassoCV + OLS debias + STLSQ + bias correction
└── scripts/
    ├── fig1_functions.py
    ├── fig2_lasso_paths.py
    ├── fig3_stationary_density.py
    ├── fig4_autocorr.py
    └── fig5_noise_scaling.py
```

---

## Hyperparameters (matching the paper exactly)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DT` | 0.002 | Euler–Maruyama time step |
| `T` | 100.0 | Trajectory horizon |
| `R` | 120 | Independent trajectories per system |
| `M` | 50 | Spatial kernel centres |
| `h` (OU, DW) | 0.22 | Gaussian kernel bandwidth |
| `h` (Mult) | 0.27 | Wider bandwidth for heavier-tailed system |
| `STLS_THR` (OU, DW) | 0.25 | STLSQ relative threshold |
| `STLS_THR` (Mult) | 0.30 | STLSQ threshold for multiplicative system |
| Seeds | 42 / 123 / 7 | Random seeds (OU / double-well / multiplicative) |

---

## Notes

- All stationary densities in Figure 3 are computed analytically via the
  Fokker–Planck formula — no Monte Carlo variance contaminates the comparison.
- The code uses `np.trapezoid` (NumPy ≥ 2.0). On older NumPy installations,
  replace `np.trapezoid` with `np.trapz` in `src/` and `scripts/`.
