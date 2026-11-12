# Two-chambered bladder (section "Where the gas is compressed", Fig. 5)

Body with the dimensions of the goldfish of Table 1 (Di Santo et al. 2025): $L = 5.88$ cm, $m = 3.14$ g,
$h = 0.64$ mm, $\ell = 0.25L$, bladder volume 6.5 % of body volume, depth 1 m, $\beta = 1$, total buoyancy
stiffness from Boyle's law. Chambers at $+0.10L$ and $-0.12L$ with stiffnesses $f k_b$ and $(1-f)k_b$.
The lifting surface is at $r_L = -0.053L$, in the middle of the stable interval of a single bladder at the
measured centroid ($-0.14L$) at $U = 2U_0$, $U_0 = 1.80\,L\,\mathrm{s^{-1}}$.

The scripts need only NumPy and SciPy and write their results to `data/two_chambers/`.
Run them from any directory, e.g. from the repository root:

```bash
python scripts/two_chambers/fig5_nature_data.py
python scripts/two_chambers/two.py
python scripts/two_chambers/dive.py
python scripts/two_chambers/checks.py
```

| Script | Output | What it computes |
|---|---|---|
| `dive.py` | `dive.json` | parameters ($k_b$, $U_0$, $r_L$) and the right-hand side with several chambers; response to a fin step of $-2^\circ$ for a bladder behind, in front, incompressible and with the zebrafish partition. The part before `configs = {` is executed by the other scripts as the common set-up. |
| `fig5_nature_data.py` | `fig5.json` | data of Fig. 5: slow root $\lambda(\bar r)$ at $U = 1.5$, 2 and $3U_0$ and its zeros ($\bar r = -0.115$, $-0.088$, $-0.068L$); depth and pitch after the fin step (0.86, 0.40 and 0.44 m after 60 s) |
| `two.py` | `two.json`, `map.npz` | exact equivalence of two chambers and one chamber at $\bar r$ with $k_\theta - k_b\sigma^2$ (eigenvalues), $k_b\sigma^2/k_\theta$ for equal compliance, map of the largest real part over ($f$, $U$), time to move 1 m down and up |
| `checks.py` | `checks.json` | static pitch response $d\theta/dz = k_b\bar r/k_\theta$ for the goldfish with the zebrafish partition and the height $h$ that reproduces the response measured in zebrafish (Stoyek et al. 2011); error of formula (8) for $\lambda$; depth change for four bladders over $\beta = 1, 0.1$, $U = 1.5$–$3U_0$, three positions of the lifting surface and fin steps of 1–5$^\circ$; constant fin command that holds the new depth |

`fig5.json` is included, so Fig. 5 can be redrawn without recomputing it.
