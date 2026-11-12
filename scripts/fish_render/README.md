# Fish drawings of Figs. 1 and 5

Side-view renders of a three-dimensional fish model with the swim bladder shown in a cut-away, used as the
background of the schematic in Fig. 1 and of the two-chamber panel in Fig. 5. Forces, offsets and angles are
drawn on top with Matplotlib (see `fishart.py`).

## Model and licence

* Model: **BarramundiFish**, Khronos glTF Sample Assets,
  <https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BarramundiFish>
  (file `glTF-Binary/BarramundiFish.glb`, 12 488 144 bytes,
  SHA-256 `ecc3bafb6b00f2c8b810863c388e3768a7b7ea0d0335e8cb8c574c266e571f4a`).
* Licence: **CC0 1.0 Universal** (<https://creativecommons.org/publicdomain/zero/1.0/legalcode>); the model
  README states "© 2017, Public" and credits Microsoft.
* The file is larger than 10 MB and is not kept in the repository: `download_model.py` fetches it and checks
  its size and hash.
* The body shape is illustrative: the barramundi is a perciform fish, not a cyprinid. No three-dimensional
  model of a cyprinid with a verified CC0 or CC BY licence was found (see the paper's figure captions).

## Files

| file | purpose |
|---|---|
| `download_model.py` | downloads `BarramundiFish.glb` next to the scripts |
| `render_fish_side.py` | Blender (Cycles, CPU): orthographic side view, transparent background, soft neutral light; the fish is pitched about its centre of mass (centre of volume of the model); ellipsoids of the bladder are added inside the body. For every position it writes the textured fish and masks of the bladder, the shaded bladder, the pelvic fin and the near pectoral fin, plus `meta.json` with the pixel scale |
| `make_renders.py` | render jobs of Fig. 1a, Fig. 1b and Fig. 5 (bladder geometry, pitch); the six positions of Fig. 1b are computed from the model with the package `bladder` |
| `fishart.py` | post-processing in Matplotlib: cut-away of the body wall over the bladder, light tint of the pelvic fin, the near pectoral fin redrawn over the cut, hatched stiff chamber, placement of a render in plot coordinates (`Render.draw`, `Render.body`, `Render.belly`) |

Geometry (body lengths L from the centre of mass, head forward): single bladder at r_A = -0.125 L
(fish-like set, rho_A = -0.5, l = 0.25 L), 0.06 L above the centre of mass; chambers of Fig. 5 at +0.10 L and
-0.12 L; the compliant chamber is drawn compressed (x0.6 of its semi-axes).

## Run

Requires Blender as the Python module `bpy` (tested with bpy 5.2.3 LTS, Cycles on CPU), NumPy, SciPy and
Matplotlib. From the repository root:

    python scripts/fish_render/download_model.py
    python scripts/fish_render/make_renders.py            # fig1a fig1b fig5; about 10 min on 2 CPU cores

The renders are written to `scripts/fish_render/fish_renders/` (about 11 MB of PNG, not kept in the
repository). A render is drawn into Matplotlib axes with

    from fishart import Render
    r = Render("scripts/fish_render/fish_renders/fig1a", "a")
    r.draw(ax, x0, y0, length)          # fish with its centre of mass at (x0, y0)
