"""Writes the render jobs for Fig. 1 and Fig. 5 and runs render_fish_side.py (Blender as the python module bpy).

    python make_renders.py [fig1a] [fig1b] [fig5]     (default: all)

Geometry of the bladder (body lengths L from the centre of mass, head forward):
  Fig. 1: one bladder at r_A = -0.125 L (fish-like set: rho_A = -0.5, l = 0.25 L; Methods), raised to 0.06 L above
          the centre of mass (towards the spine).
  Fig. 5: anterior chamber at +0.10 L, posterior chamber at -0.12 L (fig5_nature_data.py), equal semi-axes; the
          compliant chamber is drawn compressed (x0.6 of its semi-axes) and blue; the stiff one is redrawn flat in
          post-processing (fishart.py).
Fig. 1b: pitch and bladder scale of the six positions follow the numerical solution of the model (vehicle-like
set, U = 2 m/s, d = -0.15 m, z0 = -0.3 m) at t = 0, 2.6, 5.2, 8.4, 11.8, 15.4 s; the pitch is multiplied by 2.2 and
the bladder scale is clip(1 + 0.6 z_b, 0.4, 1.45) with z_b = z + r_A sin(theta), for visibility. They are computed
here with the package `bladder` (fig1b_states()) and written to fish_renders/fig1b_states.json.

Run from the repository root after `python scripts/fish_render/download_model.py`, in a Python that has Blender
as the module bpy (tested with bpy 5.2.3).
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLB = HERE / "BarramundiFish.glb"
OUT = HERE / "fish_renders"
BL1 = (-0.125, 0.06, 0.115, 0.042)                 # raised towards the spine (10.11-b)
FRONT, REAR = (0.10, 0.055, 0.085, 0.042), (-0.12, 0.055, 0.085, 0.042)   # equal chambers
SQUEEZE = 0.6


def frames(name, pitch, bladders, masks=("bm", "bs", "fin", "pect")):
    fr = [{"name": name, "pitch_deg": pitch, "mode": "opaque", "bladders": bladders}]
    mode = {"bm": "bladder", "bs": "bladder_shaded", "fin": "fin", "pect": "pect", "xr": "xray"}
    fr += [{"name": f"{name}_{m}", "pitch_deg": pitch, "mode": mode[m], "bladders": bladders} for m in masks]
    return fr


def run(job, tag):
    p = OUT / tag
    job.update({"glb": str(GLB), "out": str(p)})
    p.mkdir(parents=True, exist_ok=True)
    (p / "job.json").write_text(json.dumps(job, indent=1))
    subprocess.run([sys.executable, "-I", str(HERE / "render_fish_side.py"), str(p / "job.json")], check=True)


def fig1a():
    run({"res": 1800, "samples": 64, "frames": frames("a", 17.0, [list(BL1) + [1.0]])}, "fig1a")


PITCH_GAIN = 2.2
S01_TIMES = (0.0, 2.6, 5.2, 8.4, 11.8, 15.4)


def fig1b_states():
    """[pitch (deg, x PITCH_GAIN), bladder scale] of the six positions of Fig. 1b."""
    import numpy as np
    from scipy.integrate import solve_ivp
    sys.path.insert(0, str(HERE.parents[1]))           # repository root: package bladder
    from bladder import P, rhs, rL_on_B0
    ra = -4.6
    sol = solve_ivp(rhs, (0, 16.4), [-0.3, 0, 0, 0], args=(2.0, ra, rL_on_B0(ra, P) - 0.15, P),
                    method="DOP853", rtol=1e-9, atol=1e-11, dense_output=True).sol
    out = []
    for tk in S01_TIMES:
        z, _, th, _ = sol(tk)
        vol = float(np.clip(1 + 0.6 * (z + ra * np.sin(th)), 0.4, 1.45))
        out.append([round(float(np.degrees(PITCH_GAIN * th)), 3), round(vol, 4)])
    return out


def fig1b():
    st = fig1b_states()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig1b_states.json").write_text(json.dumps(st))
    fr = []
    for i, (pitch, vol) in enumerate(st):
        fr += frames(f"b{i}", pitch, [list(BL1) + [vol]])
    run({"res": 900, "samples": 48, "frames": fr}, "fig1b")


def fig5():
    post = [list(FRONT) + [1.0, "grey"], list(REAR) + [SQUEEZE]]
    ant = [list(FRONT) + [SQUEEZE], list(REAR) + [1.0, "grey"]]
    run({"res": 1500, "samples": 64, "frames": frames("post", 14.0, post) + frames("ant", -14.0, ant)}, "fig5")


if __name__ == "__main__":
    for n in sys.argv[1:] or ["fig1a", "fig1b", "fig5"]:
        globals()[n]()
