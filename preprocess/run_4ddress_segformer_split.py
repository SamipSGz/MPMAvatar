"""
run_4ddress_segformer_split.py

Single-view (or multi-view) SegFormer garment split on REAL 4D-DRESS captures.

For each 4D-DRESS frame we have:
  - scan mesh  : Meshes_pkl/mesh-fXXXXX.pkl  (vertices, faces, colors, GT-aligned)
  - real photo : Capture/<cam>/images/capture-fXXXXX.png   (940x1280)
  - calibration: Capture/cameras.pkl  -> {cam: {intrinsics 3x3, extrinsics 3x4}}
  - GT labels  : Semantic/labels/label-fXXXXX.pkl -> scan_labels (per-vertex 0..4)

Pipeline:
  1. SegFormer v5 on the real photo -> 3-class mask (0 bg, 1 upper, 2 lower)
  2. Project mesh vertices into the image, read the mask label for visible verts
  3. BFS-propagate labels across mesh edges to occluded verts, blocked by
     vertices that were visibly background (so it cannot flood the body)
  4. Export upper/lower garment GLBs, log to W&B (js-teamm/MPMAvatarComparision)
  5. Validate against GT scan_labels (IoU)
"""

import os, argparse, pickle
import numpy as np
import torch, trimesh, wandb
from collections import deque
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

WANDB_KEY     = "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"
WANDB_ENTITY  = "js-teamm"
WANDB_PROJECT = "MPMAvatarComparision"
COLORS = {1: np.array([255,140,40],np.uint8), 2: np.array([40,120,255],np.uint8)}


def load_mesh(pkl_path):
    m = pickle.load(open(pkl_path, "rb"))
    return np.asarray(m["vertices"], np.float64), np.asarray(m["faces"], np.int64)


def load_gt(label_path):
    if not os.path.exists(label_path):
        return None
    d = pickle.load(open(label_path, "rb"))
    return np.asarray(d["scan_labels"]).astype(np.int64)


def run_segformer(img, model_dir):
    proc = SegformerImageProcessor.from_pretrained(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).eval()
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        lg = model(**inp).logits
    return torch.nn.functional.interpolate(
        lg, size=(img.height, img.width), mode="bilinear", align_corners=False
    ).argmax(1)[0].numpy().astype(np.uint8)


def project(verts, K, RT):
    R, t = RT[:3, :3], RT[:3, 3]
    vc = (R @ verts.T).T + t
    z = vc[:, 2]
    px = K[0, 0] * vc[:, 0] / np.clip(z, 1e-6, None) + K[0, 2]
    py = K[1, 1] * vc[:, 1] / np.clip(z, 1e-6, None) + K[1, 2]
    return px, py, z


def build_adjacency(faces, n):
    adj = [set() for _ in range(n)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def backproject(verts, faces, masks_and_cams, img_wh):
    """masks_and_cams: list of (seg_mask, K, RT). Accumulate per-vertex votes
    across views, then BFS into unseen verts blocked by visible-background."""
    n = len(verts)
    votes = np.zeros((n, 3), np.int64)  # bg, upper, lower
    W, H = img_wh
    for seg, K, RT in masks_and_cams:
        px, py, z = project(verts, K, RT)
        vis = (z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        xi = np.clip(px[vis].astype(int), 0, W - 1)
        yi = np.clip(py[vis].astype(int), 0, H - 1)
        lbl = seg[yi, xi]
        idx = np.where(vis)[0]
        for v, l in zip(idx, lbl):
            votes[v, l] += 1

    seen = votes.sum(1) > 0
    garment = votes[:, 1:]
    best = garment.argmax(1) + 1
    is_seed = (garment.max(1) > 0) & (garment.max(1) >= votes[:, 0])
    seed = np.zeros(n, np.int32); seed[is_seed] = best[is_seed]
    visible_bg = seen & (~is_seed) & (votes[:, 0] > 0)

    adj = build_adjacency(faces, n)
    labels = seed.copy()
    q = deque(np.where(seed > 0)[0].tolist())
    while q:
        v = q.popleft()
        for nb in adj[v]:
            if labels[nb] == 0 and not visible_bg[nb]:
                labels[nb] = labels[v]; q.append(nb)
    return labels, seed, seen


def export_glb(verts, faces, idx, label, out):
    mask = np.zeros(len(verts), bool); mask[idx] = True
    fmask = np.all(mask[faces], axis=1)
    cf = faces[fmask]
    if len(cf) == 0:
        trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3),int)).export(out)
        return out, 0, 0
    kv = np.unique(cf.reshape(-1))
    remap = {int(o): i for i, o in enumerate(kv)}
    rf = np.array([[remap[int(v)] for v in f] for f in cf], np.int32)
    col = np.tile(COLORS[label], (len(kv), 1))
    rgba = np.concatenate([col, np.full((len(kv),1),255,np.uint8)], 1)
    trimesh.Trimesh(vertices=verts[kv], faces=rf, vertex_colors=rgba, process=False).export(out)
    return out, len(kv), len(rf)


def iou(pred_idx, gt_mask):
    pred = np.zeros_like(gt_mask, bool); pred[pred_idx] = True
    inter = (pred & gt_mask).sum(); union = (pred | gt_mask).sum()
    return float(inter) / float(union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--frame", required=True, help="e.g. 00061")
    ap.add_argument("--cam_ids", nargs="+", default=["0004"], help="one for single-view")
    ap.add_argument("--model_dir", default="checkpoints/finetuned_segformer_v5/best_model")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gt_upper_label", type=int, default=3)
    ap.add_argument("--gt_lower_label", type=int, default=4)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    f = args.frame
    verts, faces = load_mesh(os.path.join(args.take_dir, "Meshes_pkl", f"mesh-f{f}.pkl"))
    gt = load_gt(os.path.join(args.take_dir, "Semantic", "labels", f"label-f{f}.pkl"))
    print(f"mesh verts={len(verts)} faces={len(faces)} gt={'yes' if gt is not None else 'no'}")

    cams = pickle.load(open(os.path.join(args.take_dir, "Capture", "cameras.pkl"), "rb"))
    masks_and_cams = []
    img_wh = None
    for cam in args.cam_ids:
        img_path = os.path.join(args.take_dir, "Capture", cam, "images", f"capture-f{f}.png")
        img = Image.open(img_path).convert("RGB")
        img_wh = img.size
        seg = run_segformer(img, args.model_dir)
        K = np.asarray(cams[cam]["intrinsics"], np.float64)
        RT = np.asarray(cams[cam]["extrinsics"], np.float64)
        u, c = np.unique(seg, return_counts=True)
        print(f"  cam {cam}: mask px {dict(zip(u.tolist(), c.tolist()))}")
        masks_and_cams.append((seg, K, RT))
        # save overlay
        ov = np.array(img).copy()
        for l in (1, 2):
            mm = seg == l
            ov[mm] = (ov[mm] * 0.4 + COLORS[l] * 0.6).astype(np.uint8)
        Image.fromarray(ov).save(os.path.join(args.out_dir, f"overlay_{cam}.png"))

    labels, seed, seen = backproject(verts, faces, masks_and_cams, img_wh)
    upper_idx = np.where(labels == 1)[0]
    lower_idx = np.where(labels == 2)[0]
    print(f"seeds upper={int((seed==1).sum())} lower={int((seed==2).sum())} "
          f"seen={int(seen.sum())}/{len(verts)}")
    print(f"final upper={len(upper_idx)} lower={len(lower_idx)}")

    glb_u = os.path.join(args.out_dir, "garment_upper_4ddress.glb")
    glb_l = os.path.join(args.out_dir, "garment_lower_4ddress.glb")
    _, uv, uf = export_glb(verts, faces, upper_idx, 1, glb_u)
    _, lv, lf = export_glb(verts, faces, lower_idx, 2, glb_l)
    print(f"GLB upper {uv}v/{uf}f  lower {lv}v/{lf}f")

    summary = {"upper_verts": len(upper_idx), "lower_verts": len(lower_idx)}
    if gt is not None:
        gtu = gt == args.gt_upper_label
        gtl = gt == args.gt_lower_label
        iou_u = iou(upper_idx, gtu); iou_l = iou(lower_idx, gtl)
        garment_pred = np.zeros(len(verts), bool); garment_pred[upper_idx] = True; garment_pred[lower_idx] = True
        gt_garment = gt >= 1
        iou_all = iou(np.where(garment_pred)[0], gt_garment)
        print(f"IoU upper={iou_u:.3f} lower={iou_l:.3f} garment-vs-body={iou_all:.3f}")
        print(f"GT upper(label{args.gt_upper_label})={int(gtu.sum())} lower(label{args.gt_lower_label})={int(gtl.sum())}")
        summary.update({"iou_upper": iou_u, "iou_lower": iou_l, "iou_garment": iou_all})

    if not args.no_wandb:
        os.environ["WANDB_API_KEY"] = WANDB_KEY
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                         name=f"garment_seg_4ddress_f{f}",
                         tags=["garment","segformer","4ddress",
                               "single-view" if len(args.cam_ids)==1 else "multi-view"])
        wandb.log({
            "4ddress/garment_3d/upper_cloth": wandb.Object3D(glb_u),
            "4ddress/garment_3d/lower_cloth": wandb.Object3D(glb_l),
            "4ddress/segmentation_view": wandb.Image(
                os.path.join(args.out_dir, f"overlay_{args.cam_ids[0]}.png"),
                caption=f"SegFormer real-image segmentation (cam {args.cam_ids[0]})"),
        })
        wandb.summary.update(summary)
        url = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run.id}"
        wandb.finish()
        print("W&B run:", url)
    print("Done.")


if __name__ == "__main__":
    main()
