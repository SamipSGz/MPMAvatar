"""
export_log_garment_glb.py

Final step of the single-view SegFormer garment-split pipeline.

Takes the predicted cloth-vertex-index arrays (predicted_cloth_vertices.npy)
for upper and lower garments, exports each as a colored GLB mesh, and logs
both as separate W&B Object3D assets to js-teamm/MPMAvatarComparision.

W&B keys (accurate, separate upper/lower):
    {dataset}/garment_3d/upper_cloth
    {dataset}/garment_3d/lower_cloth
"""

import os
import argparse
import numpy as np
import trimesh
import wandb


def read_obj(path):
    verts, faces = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.split()
                idx = [int(x.split("/")[0]) - 1 for x in p[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


LABEL_COLORS = {
    "upper": np.array([255, 140, 40], dtype=np.uint8),   # orange
    "lower": np.array([40, 120, 255], dtype=np.uint8),   # blue
}

WANDB_KEY     = "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"
WANDB_ENTITY  = "js-teamm"
WANDB_PROJECT = "MPMAvatarComparision"


def export_garment_glb(verts, faces, cloth_v_idx, label_name, out_path):
    """Export only the faces whose vertices are all in cloth_v_idx, as a colored GLB."""
    cloth_v_mask = np.zeros(len(verts), dtype=bool)
    cloth_v_mask[cloth_v_idx] = True

    face_mask = np.all(cloth_v_mask[faces], axis=1)
    cloth_faces = faces[face_mask]

    kept_v = np.unique(cloth_faces.reshape(-1))
    remap = {int(old): new for new, old in enumerate(kept_v)}
    remapped = np.array([[remap[int(v)] for v in f] for f in cloth_faces], dtype=np.int32)

    col = LABEL_COLORS[label_name]
    colors = np.tile(col, (len(kept_v), 1))
    rgba = np.concatenate([colors, np.full((len(kept_v), 1), 255, dtype=np.uint8)], axis=1)

    mesh = trimesh.Trimesh(
        vertices=verts[kept_v], faces=remapped,
        vertex_colors=rgba, process=False,
    )
    mesh.export(out_path)
    print("  exported " + label_name + " GLB: " + out_path +
          " (" + str(len(kept_v)) + " v / " + str(len(remapped)) + " f, " +
          str(os.path.getsize(out_path) // 1024) + " KB)")
    return out_path, len(kept_v), len(remapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["4ddress", "actorshq"])
    ap.add_argument("--mesh_path", required=True)
    ap.add_argument("--upper_npy", required=True, help="predicted_cloth_vertices.npy for upper")
    ap.add_argument("--lower_npy", required=True, help="predicted_cloth_vertices.npy for lower")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overlay", default=None, help="optional segmentation overlay PNG")
    ap.add_argument("--wandb_run_id", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading mesh: " + args.mesh_path)
    verts, faces = read_obj(args.mesh_path)
    print("  verts=" + str(verts.shape) + "  faces=" + str(faces.shape))

    upper_idx = np.load(args.upper_npy).astype(np.int64).reshape(-1)
    lower_idx = np.load(args.lower_npy).astype(np.int64).reshape(-1)
    print("  upper cloth verts=" + str(len(upper_idx)) +
          "  lower cloth verts=" + str(len(lower_idx)))

    glb_upper = os.path.join(args.out_dir, "garment_upper_" + args.dataset + ".glb")
    glb_lower = os.path.join(args.out_dir, "garment_lower_" + args.dataset + ".glb")
    _, uv, uf = export_garment_glb(verts, faces, upper_idx, "upper", glb_upper)
    _, lv, lf = export_garment_glb(verts, faces, lower_idx, "lower", glb_lower)

    if not args.no_wandb:
        print("Logging to W&B " + WANDB_ENTITY + "/" + WANDB_PROJECT)
        os.environ["WANDB_API_KEY"] = WANDB_KEY
        init_kwargs = dict(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name="garment_seg_" + args.dataset,
            tags=["garment", "segformer", "single-view", args.dataset],
        )
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)
        prefix = args.dataset + "/garment_3d"
        log_dict = {
            prefix + "/upper_cloth": wandb.Object3D(glb_upper),
            prefix + "/lower_cloth": wandb.Object3D(glb_lower),
        }
        if args.overlay and os.path.exists(args.overlay):
            log_dict[args.dataset + "/segmentation_view"] = wandb.Image(
                args.overlay, caption="SegFormer single-view segmentation")
        wandb.log(log_dict)
        wandb.summary.update({
            args.dataset + "_upper_verts": uv, args.dataset + "_upper_faces": uf,
            args.dataset + "_lower_verts": lv, args.dataset + "_lower_faces": lf,
        })
        print("  logged keys: " + prefix + "/upper_cloth , " + prefix + "/lower_cloth")
        url = "https://wandb.ai/" + WANDB_ENTITY + "/" + WANDB_PROJECT + "/runs/" + run.id
        wandb.finish()
        print("W&B run: " + url)

    print("Done.")


if __name__ == "__main__":
    main()
