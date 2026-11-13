# Figures of the Nature version

Scripts that assemble the figures of the Nature-format manuscript (main text and Extended Data) from the data
in `data/` and the package `bladder`. Run from the repository root:

```bash
python scripts/fish_render/ct_zebrafish.py "<path>/bmp1a + +_sa2416_455_nr16.DCM"   # optional: data/fig1_ct
python scripts/nature_figures/figures.py both            # Figs. 1-4, Extended Data Figs. 1-2 -> figures/nature/
python scripts/nature_figures/fig5_nature_fish.py both   # Fig. 5 -> figures/nature/
```

| file | purpose |
|---|---|
| `figures.py` | Figs. 1-4 and Extended Data Figs. 1-2, English and Russian labels (`en`, `ru`, `both`; figure names select single figures, e.g. `fig1`) |
| `zebrafish_draw.py` | vector drawing of an adult zebrafish with the two-chambered swim bladder (`draw_anatomy`, `draw_fish_at`), used in Fig. 1b,c |
| `fig5_nature_fish.py` | Fig. 5 (data `data/two_chambers/fig5.json`, fish renders of `scripts/fish_render`) |

**Fig. 1a.** Micro-computed tomography of an adult wild-type zebrafish from Hur et al. (2017, eLife 6:e26014),
scan `bmp1a + +_sa2416_455_nr16.DCM`, Zenodo doi:10.5281/zenodo.5011917, licence CC0 1.0. The DICOM file
(252 MB) is not kept here; `scripts/fish_render/ct_zebrafish.py` turns it into `data/fig1_ct/zebrafish_nr16.npz`
(side projection, swim-bladder gas, pelvic-fin mask, centres of mass and buoyancy; method in the script's
docstring), which is included, so Fig. 1 can be redrawn without the scan.

**Fish renders.** Fig. 5a uses side views of the BarramundiFish model (CC0) rendered with Blender by
`scripts/fish_render/make_renders.py`; see `scripts/fish_render/README.md`. The renders are not kept in the
repository.

When the scripts are run inside the manuscript folder (`nature_fish/figures_src`), they write to the manuscript's
`figures/` folders instead.
