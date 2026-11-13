"""Panel a of Fig. 1: adult zebrafish from micro-computed tomography.

Data: Hur M. et al. (2017) microCT-based phenomics in the zebrafish skeleton reveals virtues of deep phenotyping
in a distributed organ system. eLife 6:e26014. Scans on Zenodo, doi:10.5281/zenodo.5011917 (licence CC0 1.0).
File used: "bmp1a + +_sa2416_455_nr16.DCM" (wild-type clutchmate, Scanco vivaCT40, isotropic voxel 21 um,
multi-frame DICOM 1383 x 367 x 248; axis 0 snout -> tail, axis 1 dorsal -> ventral, axis 2 left -> right).
The DICOM file (252 MB) is not kept in the repository: download it from Zenodo.

    python ct_zebrafish.py <path to the .DCM file> [<output .npz>]

Writes (default data/fig1_ct/zebrafish_nr16.npz and .json in the repository):
  proj    side view: sum of (HU + 1000) * slope-scaled attenuation over the body along axis 2 (x, y)
  sb      side projection of the swim-bladder gas (largest enclosed air cavity, both chambers)
  pelvic  side mask of the pelvic fin (its rays in the maximum-intensity projection)
  cm, cb, sbc   centre of mass, centre of buoyancy and bladder centroid, (x, y) in mm from the scan origin
  vx      pixel size, mm
Method:
  * tissue: HU > -400 (the scan is in air); the body is the largest connected component at 42 um (2x
    downsampling, binary opening), up-sampled back and intersected with the full-resolution threshold;
  * bladder: per slice along x, holes of the closed body mask are filled; the cavities (filled minus body,
    opened) are labelled in 3D and the largest is the bladder (16.9 mm^3, 5.5 % of body + bladder);
  * centre of mass: density 1 + HU/1000 g/cm^3 clipped to [0.9, 2.5] (an approximation; the HU calibration
    of the scanner for fixed tissue is not known, uncertainty about 1 mm); centre of buoyancy: centroid of the
    body with the bladder;
  * pelvic fin: pixels of the maximum-intensity projection above 1200 HU in the window x = 11.95-15.4 mm,
    y = 6.2-7.7 mm (behind the pelvic girdle, below the rib tips), largest 8-connected component, closed with
    a disc of radius 6 px, holes filled, opened with a 3x3 square.
Needs numpy, scipy, pydicom.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent


def repo_root():
    for p in [HERE, *HERE.parents]:
        if (p / "bladder").is_dir():
            return p
    return HERE


def process(dcm):
    d = pydicom.dcmread(dcm)
    slope, icpt = float(d.RescaleSlope), float(d.RescaleIntercept)
    vx = float(d.PixelSpacing[0])                       # mm (isotropic)
    raw = d.pixel_array                                 # (x, y, z) int16
    thr = int((-400 - icpt) / slope)
    m = raw > thr
    # body: largest component at 2x downsampling
    m2 = ndi.binary_opening(m[::2, ::2, ::2])
    lab, n = ndi.label(m2)
    sz = ndi.sum(m2, lab, range(1, n + 1))
    body2 = lab == (np.argmax(sz) + 1)
    up = np.repeat(np.repeat(np.repeat(body2, 2, 0), 2, 1), 2, 2)[:raw.shape[0], :raw.shape[1], :raw.shape[2]]
    body = up & m
    hu_plus = raw.astype(np.float32) * slope           # HU + 1000 (attenuation relative to air)
    proj = np.where(body, hu_plus, 0).sum(axis=2)
    # bladder
    filled = np.zeros_like(body2)
    for i in range(body2.shape[0]):
        filled[i] = ndi.binary_fill_holes(ndi.binary_closing(body2[i], iterations=3))
    cav = ndi.binary_opening(filled & ~body2)
    cl, cn = ndi.label(cav)
    cs = ndi.sum(cav, cl, range(1, cn + 1))
    sb2 = cl == (np.argmax(cs) + 1)
    sb = np.repeat(np.repeat(sb2.any(axis=2), 2, 0), 2, 1)[:raw.shape[0], :raw.shape[1]]
    # centres (x, y) in mm
    v2 = 2 * vx
    X = np.arange(body2.shape[0]) * v2
    Y = np.arange(body2.shape[1]) * v2
    a2 = raw[::2, ::2, ::2].astype(np.float32) * slope + icpt
    rho = np.clip(1 + a2 / 1000, 0.9, 2.5) * body2
    cm = [float((rho.sum(axis=(1, 2)) * X).sum() / rho.sum()), float((rho.sum(axis=(0, 2)) * Y).sum() / rho.sum())]
    fb = body2 | sb2
    cb = [float((fb.sum(axis=(1, 2)) * X).sum() / fb.sum()), float((fb.sum(axis=(0, 2)) * Y).sum() / fb.sum())]
    sbc = [float((sb2.sum(axis=(1, 2)) * X).sum() / sb2.sum()), float((sb2.sum(axis=(0, 2)) * Y).sum() / sb2.sum())]
    # pelvic fin
    mx = raw.max(axis=2).astype(np.float32) * slope + icpt
    xg, yg = np.meshgrid(np.arange(mx.shape[0]) * vx, np.arange(mx.shape[1]) * vx, indexing="ij")
    win = (xg > 11.95) & (xg < 15.4) & (yg > 6.2) & (yg < 7.7)
    b = (mx > 1200) & win
    lab, n = ndi.label(b, structure=np.ones((3, 3)))
    sz = ndi.sum(b, lab, range(1, n + 1))
    f = lab == (np.argmax(sz) + 1)
    yy, xx = np.mgrid[-6:7, -6:7]
    disk = (xx ** 2 + yy ** 2) <= 36
    pel = ndi.binary_fill_holes(ndi.binary_closing(np.pad(f, 8), structure=disk))[8:-8, 8:-8]
    pel = ndi.binary_opening(pel, structure=np.ones((3, 3)))
    vb = float(sb2.sum() * v2 ** 3)
    info = {"source": "Hur et al. 2017, eLife 6:e26014; Zenodo doi:10.5281/zenodo.5011917 (CC0 1.0)",
            "file": Path(dcm).name, "voxel_mm": vx, "body_x_mm": [float(np.where(body2.any(axis=(1, 2)))[0][[0, -1]][i] * v2) for i in (0, 1)],
            "bladder_volume_mm3": round(vb, 2),
            "bladder_fraction_percent": round(100 * sb2.sum() / fb.sum(), 2),
            "bladder_x_mm": [float(np.where(sb2.any(axis=(1, 2)))[0][[0, -1]][i] * v2) for i in (0, 1)],
            "cm_mm": cm, "cb_mm": cb, "bladder_centroid_mm": sbc,
            "pelvic_fin_x_mm": [float(xg[pel].min()), float(xg[pel].max())],
            "note": "x from the scan origin near the snout, y dorsal -> ventral; CM from density 1 + HU/1000 (rough)"}
    return dict(proj=proj.astype(np.float32), sb=sb, pelvic=pel, cm=np.array(cm), cb=np.array(cb),
                sbc=np.array(sbc), vx=vx), info


if __name__ == "__main__":
    dcm = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else repo_root() / "data" / "fig1_ct" / "zebrafish_nr16.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays, info = process(dcm)
    np.savez_compressed(out, **arrays)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(out)
    print(json.dumps(info, indent=1))
