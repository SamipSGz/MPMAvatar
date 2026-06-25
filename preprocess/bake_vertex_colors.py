"""
bake_vertex_colors.py

Sample a UV texture into per-vertex RGB colors so the geometry mesh can be
rendered "textured" (Gouraud per-vertex) by generate_predicted_cloth_vertices.py
--vertex_colors_npy.

This matters because SegFormer v5 was fine-tuned on the ATR dataset (real human
photos). Grey geometry renders barely trigger it; textured renders that look
photo-like segment far better.

The UV OBJ (e.g. a1s1_uv.obj) shares vertex indexing/topology with the geometry
OBJ (FrameRec000460.obj) — faces are `vIdx/vtIdx` — so per-vertex colors keyed by
vIdx transfer directly to the geometry mesh.
"""

import argparse
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv_obj", required=True)
    ap.add_argument("--texture", required=True)
    ap.add_argument("--n_verts", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vts = []
    vert_to_vt = {}
    with open(args.uv_obj) as f:
        for line in f:
            if line.startswith("vt "):
                p = line.split()
                vts.append([float(p[1]), float(p[2])])
            elif line.startswith("f "):
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    if len(parts) >= 2 and parts[1]:
                        v = int(parts[0]) - 1
                        vt = int(parts[1]) - 1
                        if v not in vert_to_vt:
                            vert_to_vt[v] = vt
    vts = np.array(vts, dtype=np.float64)
    print("vt count=" + str(len(vts)) + "  verts mapped=" + str(len(vert_to_vt)))

    tex = np.array(Image.open(args.texture).convert("RGB"))
    TH, TW = tex.shape[:2]

    colors = np.full((args.n_verts, 3), 180, dtype=np.uint8)  # grey default
    for v, vt in vert_to_vt.items():
        if v >= args.n_verts:
            continue
        u, w = vts[vt]
        px = int(np.clip(u * (TW - 1), 0, TW - 1))
        py = int(np.clip((1.0 - w) * (TH - 1), 0, TH - 1))
        colors[v] = tex[py, px]

    np.save(args.out, colors)
    print("wrote " + args.out + "  shape=" + str(colors.shape) +
          "  mean RGB=" + str(colors.mean(axis=0).round(1).tolist()))


if __name__ == "__main__":
    main()
