"""Side-view renders of the BarramundiFish model (CC0, Khronos glTF Sample Assets) for the figures of the
Nature version: orthographic camera, transparent background, soft neutral light. Inside the body one or two
ellipsoids of the swim bladder are placed where they are in the model.

    python render_fish_side.py <job.json>

job.json: {"glb": path, "out": dir, "res": px width, "samples": n, "frames": [
    {"name": str, "pitch_deg": float, "mode": "opaque"|"xray"|"bladder"|"fin",
     "bladders": [[x_over_L, z_over_L, a_over_L, c_over_L, scale], ...]}]}
x, z of a bladder are measured from the centre of mass (head towards +x, up +z) in body lengths L; a and c are
the half-length and half-height (the half-width is 0.55 of the half-height); an optional sixth item "grey" gives
a grey chamber instead of a blue one; scale multiplies all three semi-axes (volume change).
mode "opaque": textured fish; "xray": body semi-transparent, bladder seen through it; "bladder": bladder alone,
flat white (mask for the cut-away); "bladder_shaded": bladder alone, shaded; "fin": pelvic fin alone, flat white
(mask for the highlight); "pect": near-side pectoral fin alone, flat white (drawn over the cut-away).
For each frame a PNG with alpha is written, and meta.json with the pixel mapping:
    px = cx + (x * L) * k,  py = cy - (z * L) * k   (x, z in body lengths from the centre of mass).
"""
import json
import math
import sys
from pathlib import Path

import bpy  # noqa: I001  (bpy must be imported before bmesh/mathutils)
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

job = json.loads(Path(sys.argv[-1]).read_text())
OUT = Path(job["out"])
OUT.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=job["glb"])
meshes = [o for o in scene.objects if o.type == "MESH"]
for o in meshes:                                   # bake the import transform into the vertices
    o.data.transform(o.matrix_world)
    o.matrix_world = Matrix.Identity(4)
fish = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = fish
    bpy.ops.object.join()
me = fish.data
# model: length along Y with the head at -Y, height along Z. New frame: head towards +X, up +Z.
co = np.array([v.co[:] for v in me.vertices])
new = np.column_stack([-co[:, 1], co[:, 0], co[:, 2]])
L0 = np.ptp(new[:, 0])
new = new / L0                                     # body length = 1
me.vertices.foreach_set("co", new.ravel())
me.update()

# centre of mass of a uniform body: thickness along Y by ray casting on an XZ grid
bm = bmesh.new()
bm.from_mesh(me)
bvh = BVHTree.FromBMesh(bm)
xs = np.linspace(new[:, 0].min(), new[:, 0].max(), 220)
zs = np.linspace(new[:, 2].min(), new[:, 2].max(), 110)
mass = mx = mz = 0.0
for x in xs:
    for z in zs:
        hits, o, d = [], Vector((x, -1.0, z)), Vector((0, 1, 0))
        while True:
            h = bvh.ray_cast(o, d, 2.0)
            if h[0] is None:
                break
            hits.append(h[0].y)
            o = h[0] + d * 1e-5
        if len(hits) >= 2:
            th = max(hits) - min(hits)
            mass += th
            mx += th * x
            mz += th * z
CMX, CMZ = mx / mass, mz / mass
new[:, 0] -= CMX
new[:, 2] -= CMZ
me.vertices.foreach_set("co", new.ravel())
me.update()
# pelvic fin: faces below the belly line just ahead of the centre of mass (checked on the side view)
zraw = new[:, 2] + CMZ                              # height above the lowest point of the model, in L
finv = (zraw < 0.068) & (new[:, 0] > -0.02) & (new[:, 0] < 0.125)
fin_faces = [p.index for p in me.polygons if all(finv[v] for v in p.vertices)]
fin_info = {"x": [float(new[finv, 0].min()), float(new[finv, 0].max())],
            "z": [float(new[finv, 2].min()), float(new[finv, 2].max())], "n_faces": len(fin_faces)}


def part_object(faces):
    """Copy of the fish that keeps only the given faces (for masks)."""
    o = fish.copy()
    o.data = me.copy()
    scene.collection.objects.link(o)
    b = bmesh.new()
    b.from_mesh(o.data)
    keep = set(faces)
    bmesh.ops.delete(b, geom=[f for f in b.faces if f.index not in keep], context="FACES")
    b.to_mesh(o.data)
    o.hide_render = True
    return o


fin_obj = part_object(fin_faces)
# near-side pectoral fin (a separate small part of the mesh on the camera side, y < 0): drawn over the cut-away
bmp = bmesh.new()
bmp.from_mesh(me)
bmp.verts.ensure_lookup_table()
comp = -np.ones(len(bmp.verts), int)
nc = 0
for i in range(len(bmp.verts)):
    if comp[i] >= 0:
        continue
    stack = [i]
    comp[i] = nc
    while stack:
        v = bmp.verts[stack.pop()]
        for e in v.link_edges:
            u = e.other_vert(v).index
            if comp[u] < 0:
                comp[u] = nc
                stack.append(u)
    nc += 1
sizes = np.bincount(comp)
pect = np.array([sizes[comp[i]] < 500 and new[i, 1] < 0 for i in range(len(new))])
pect_faces = [p.index for p in me.polygons if all(pect[v] for v in p.vertices)]
pect_obj = part_object(pect_faces)
fin_info["pectoral_faces"] = len(pect_faces)

body_mat = me.materials[0]
bsdf = body_mat.node_tree.nodes["Principled BSDF"]

# bladder material
mat_b = bpy.data.materials.new("bladder")
mat_b.use_nodes = True
nb = mat_b.node_tree.nodes["Principled BSDF"]
nb.inputs["Base Color"].default_value = (0.42, 0.62, 0.90, 1)
nb.inputs["Roughness"].default_value = 0.35
nb.inputs["Coat Weight"].default_value = 0.3
mat_g = bpy.data.materials.new("bladder_grey")       # second chamber, not compliant (Fig. 5)
mat_g.use_nodes = True
ng = mat_g.node_tree.nodes["Principled BSDF"]
ng.inputs["Base Color"].default_value = (0.50, 0.52, 0.55, 1)
ng.inputs["Roughness"].default_value = 0.35
ng.inputs["Coat Weight"].default_value = 0.3

# world: neutral soft grey (only lights the scene; film is transparent)
world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = job.get("world", 0.55)


def light(name, kind, energy, rot, size=None, loc=(0, 0, 0)):
    o = bpy.data.objects.new(name, bpy.data.lights.new(name, kind))
    o.data.energy = energy
    o.rotation_euler = [math.radians(a) for a in rot]
    o.location = loc
    if size is not None:
        o.data.size = size
    scene.collection.objects.link(o)
    return o


light("key", "AREA", 60, (70, 0, -15), size=3.0, loc=(0.6, -2.0, 1.6))
light("fill", "AREA", 25, (110, 0, 20), size=3.0, loc=(-0.8, -2.2, -0.8))
light("top", "SUN", 1.2, (0, 0, 0))

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type = "ORTHO"
SPAN = job.get("span", 1.25)                       # visible width in body lengths
cam.data.ortho_scale = SPAN
cam.location = (0, -5, 0)
cam.rotation_euler = (math.radians(90), 0, 0)      # looks along +Y; +X right, +Z up
scene.collection.objects.link(cam)
scene.camera = cam

RES = job["res"]
ASPECT = job.get("aspect", 0.5)
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = job.get("samples", 64)
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6
scene.cycles.transparent_max_bounces = 16
scene.render.film_transparent = True
scene.render.resolution_x = RES
scene.render.resolution_y = int(RES * ASPECT)
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

ell = []


def set_bladders(spec):
    for o in ell:
        bpy.data.objects.remove(o)
    ell.clear()
    for item in spec:
        x, z, a, c, k = item[:5]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0, location=(x, 0, z))
        o = bpy.context.object
        o.scale = (a * k, 0.55 * c * k, c * k)
        bpy.ops.object.shade_smooth()
        o.data.materials.append(mat_g if len(item) > 5 and item[5] == "grey" else mat_b)
        o["mat"] = o.data.materials[0].name
        o.parent = pivot
        ell.append(o)


pivot = bpy.data.objects.new("pivot", None)
scene.collection.objects.link(pivot)
fish.parent = pivot
fin_obj.parent = pivot
pect_obj.parent = pivot

# emission-only material for masks
mat_m = bpy.data.materials.new("mask")
mat_m.use_nodes = True
nt = mat_m.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1, 1, 1, 1)
outn = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs[0], outn.inputs[0])

# x-ray: mix the textured body with a transparent shader


def body_mode(mode, alpha):
    for mt in me.materials:
        if mt is not None and mt.name not in ("mask", "bladder"):
            _body_mode(mt, mode, alpha)


def _body_mode(mt, mode, alpha):
    ntb = mt.node_tree
    bsdf = ntb.nodes["Principled BSDF"]
    out = ntb.nodes["Material Output"]
    for n in [n for n in ntb.nodes if n.name in ("xr_mix", "xr_tr", "xr_lw", "xr_mr")]:
        ntb.nodes.remove(n)
    if mode == "opaque":
        ntb.links.new(bsdf.outputs[0], out.inputs[0])
        return
    tr = ntb.nodes.new("ShaderNodeBsdfTransparent")
    tr.name = "xr_tr"
    mix = ntb.nodes.new("ShaderNodeMixShader")
    mix.name = "xr_mix"
    # facing-dependent opacity: edges stay more opaque (x-ray look)
    lw = ntb.nodes.new("ShaderNodeLayerWeight")
    lw.name = "xr_lw"
    lw.inputs["Blend"].default_value = 0.45
    mr = ntb.nodes.new("ShaderNodeMapRange")
    mr.name = "xr_mr"
    mr.inputs["To Min"].default_value = alpha
    mr.inputs["To Max"].default_value = min(1.0, alpha + 0.55)
    ntb.links.new(lw.outputs["Facing"], mr.inputs["Value"])
    ntb.links.new(mr.outputs["Result"], mix.inputs[0])
    ntb.links.new(tr.outputs[0], mix.inputs[1])
    ntb.links.new(bsdf.outputs[0], mix.inputs[2])
    ntb.links.new(mix.outputs[0], out.inputs[0])


meta = {"span": SPAN, "res": [scene.render.resolution_x, scene.render.resolution_y], "L_model_units": L0,
        "cm_from_snout": float(new[:, 0].max()), "pelvic": fin_info, "frames": {}}
k = RES / SPAN
for fr in job["frames"]:
    th = math.radians(fr["pitch_deg"])
    pivot.rotation_euler = (0, -th, 0)             # nose up for positive pitch
    set_bladders(fr.get("bladders", []))
    mode = fr.get("mode", "opaque")
    fish.hide_render = mode in ("bladder", "bladder_shaded", "fin", "pect")
    fin_obj.hide_render = mode != "fin"
    pect_obj.hide_render = mode != "pect"
    for ob in (fin_obj, pect_obj):
        for i in range(len(ob.data.materials)):
            ob.data.materials[i] = mat_m
    for o in ell:
        o.hide_render = mode in ("fin", "pect")
        o.data.materials[0] = mat_m if mode == "bladder" else bpy.data.materials[o["mat"]]
    body_mode(mode if mode == "xray" else "opaque", fr.get("alpha", 0.45))
    scene.render.filepath = str(OUT / f"{fr['name']}.png")
    bpy.ops.render.render(write_still=True)
    meta["frames"][fr["name"]] = {"pitch_deg": fr["pitch_deg"], "cx": scene.render.resolution_x / 2,
                                  "cy": scene.render.resolution_y / 2, "k_px_per_L": k}
    print("rendered", fr["name"], flush=True)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
