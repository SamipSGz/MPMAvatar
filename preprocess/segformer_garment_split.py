"""
segformer_garment_split.py

Drop-in replacement for the manual garment separation step in MPMAvatar.

Instead of requiring GT cloth_vertices.npz (manually annotated), this script:
  1. Loads a single RGB image + camera calibration
  2. Runs SegFormer to get 2D segmentation (0=bg, 1=upper_cloth, 2=lower_cloth)
  3. Backprojects 2D labels → 3D vertex labels via camera projection
  4. BFS-propagates labels to cover unseen/occluded vertices
  5. Calls split_cloth_human() to produce split_idx_upper.npz and split_idx_lower.npz
     in the exact same format the rest of MPMAvatar expects
  6. Logs upper+lower 3D garment mesh assets to W&B

Usage (4D-DRESS):
  python segformer_garment_split.py \\
    --dataset 4ddress \\
    --mesh_path data/s170_t1/mesh_processed.obj \\
    --image_path data/4D-DRESS/00170_Inner/Inner/Take1/Capture/0004/images/capture-f00021.png \\
    --cameras_pkl data/4D-DRESS/00170_Inner/Inner/Take1/Capture/cameras.pkl \\
    --cam_id 0004 \\
    --model_dir checkpoints/finetuned_segformer_v5/best_model \\
    --output_dir data/s170_t1

Usage (ActorsHQ):
  python segformer_garment_split.py \\
    --dataset actorshq \\
    --mesh_path data/a1_s1/FrameRec000460.obj \\
    --image_path data/ActorsHQ/Actor01/Sequence1/4x/rgbs/Cam001/Cam001_rgb000460.jpg \\
    --cam_id Cam001 \\
    --actorshq_calib data/a1_s1/calibration \\
    --model_dir checkpoints/finetuned_segformer_v5/best_model \\
    --output_dir data/a1_s1
"""

import os
import argparse
import pickle
import numpy as np
import torch
import trimesh
import wandb
from collections import deque
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from split_garments import split_cloth_human, read_obj

# ── label colors for W&B 3D mesh visualization ────────────────────────────────
LABEL_COLORS = {
    0: np.array([180, 180, 180], dtype=np.uint8),  # body  → grey
    1: np.array([255, 140,  40], dtype=np.uint8),  # upper → orange
    2: np.array([ 40, 120, 255], dtype=np.uint8),  # lower → blue
}

WANDB_KEY     = "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"
WANDB_ENTITY  = "js-teamm"
WANDB_PROJECT = "MPMAvatarComparision"


# ── SegFormer ──────────────────────────────────────────────────────────────────
def run_segformer(img_pil, model_dir, device):
    proc  = SegformerImageProcessor.from_pretrained(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device).eval()
    inp   = proc(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inp).logits
    mask = torch.nn.functional.interpolate(
        logits, size=(img_pil.height, img_pil.width),
        mode="bilinear", align_corners=False
    ).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


# ── camera loaders ─────────────────────────────────────────────────────────────
def load_camera_4ddress(cameras_pkl, cam_id):
    with open(cameras_pkl, "rb") as f:
        cams = pickle.load(f)
    cam = cams[cam_id]
    K  = np.array(cam["intrinsics"],  dtype=np.float64)
    RT = np.array(cam["extrinsics"], dtype=np.float64)
    if RT.shape == (4, 4):
        RT = RT[:3]
    return K, RT


def load_camera_actorshq(cam_info_json, cam_id, img_w, img_h):
    """Load ActorsHQ camera from cam_info.json (MPMAvatar format).
    Scales K to match actual image resolution (cam_info stores calibration at full res).
    """
    import json
    with open(cam_info_json) as f:
        data = json.load(f)
    cam   = data[cam_id]
    K_raw = np.array(cam["K"],  dtype=np.float64)   # 3×3 flat list
    RT    = np.array(cam["RT"], dtype=np.float64)    # 4×4 → take [:3]
    cam_W, cam_H = cam["W"], cam["H"]
    # Scale intrinsics to actual image resolution
    K = K_raw.copy()
    K[0] *= img_w / cam_W
    K[1] *= img_h / cam_H
    return K, RT[:3]


# ── backprojection + BFS propagation ──────────────────────────────────────────
def backproject_single_view(verts, faces, seg_mask, K, RT, img_w, img_h):
    """
    Project vertices into image, read SegFormer label for each visible vertex,
    then BFS-propagate to cover the full mesh.
    Returns per-vertex label array (int32), values 0/1/2.
    """
    R, t = RT[:3, :3], RT[:3, 3]
    vc    = (R @ verts.T).T + t
    depth = vc[:, 2]

    proj = (K @ vc.T).T
    px   = proj[:, 0] / np.clip(depth, 1e-6, None)
    py   = proj[:, 1] / np.clip(depth, 1e-6, None)

    labels = np.zeros(len(verts), dtype=np.int32)
    valid  = (depth > 0) & (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    xi = px[valid].astype(np.int32).clip(0, img_w - 1)
    yi = py[valid].astype(np.int32).clip(0, img_h - 1)
    labels[valid] = seg_mask[yi, xi]

    visible_cloth = int((labels > 0).sum())
    print("  visible cloth vertices: " + str(visible_cloth) + "/" + str(len(verts)))

    # BFS propagation
    adj = [set() for _ in range(len(verts))]
    for f in faces:
        adj[f[0]].add(f[1]); adj[f[0]].add(f[2])
        adj[f[1]].add(f[0]); adj[f[1]].add(f[2])
        adj[f[2]].add(f[0]); adj[f[2]].add(f[1])

    queue = deque()
    for v in range(len(verts)):
        if labels[v] > 0:
            for nb in adj[v]:
                if labels[nb] == 0:
                    queue.append(v)
                    break
    visited = set()
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for nb in adj[v]:
            if labels[nb] == 0:
                labels[nb] = labels[v]
                queue.append(nb)

    labels[labels == 0] = 1  # any still-unlabeled → default upper
    print("  after BFS: upper=" + str(int((labels==1).sum())) +
          " lower=" + str(int((labels==2).sum())))
    return labels


# ── build split_idx.npz for one garment label ─────────────────────────────────
def build_split(verts_np, faces_np, cloth_v_indices, output_path, label_name):
    """
    Given cloth vertex indices (0-based), call split_cloth_human and save npz.
    """
    verts_t = torch.tensor(verts_np).float().cuda()
    faces_t = torch.tensor(faces_np).int().cuda()

    cloth_v = torch.tensor(cloth_v_indices).int().cuda()
    # A face is cloth if ALL its vertices are cloth
    is_cloth_faces = torch.isin(faces_t, cloth_v).all(dim=1)

    fix_v = torch.empty(0).int().cuda()
    split_cloth_human(verts_t, faces_t, is_cloth_faces,
                      filename=output_path, fix_v=fix_v, iterations=0)
    print("  saved " + label_name + " split → " + output_path)


# ── W&B 3D mesh logging ────────────────────────────────────────────────────────
def export_garment_glb(verts, faces, v_labels, target_label, out_path):
    """Export mesh containing only vertices/faces for target_label as GLB."""
    cloth_v_mask = (v_labels == target_label)
    cloth_v_idx  = np.where(cloth_v_mask)[0]

    # Keep only faces where ALL vertices belong to this garment
    face_mask = np.all(cloth_v_mask[faces], axis=1)
    cloth_faces = faces[face_mask]

    # Remap face vertex indices to local (0-based) range
    remap = {old: new for new, old in enumerate(cloth_v_idx)}
    remapped = np.array([[remap[v] for v in f] for f in cloth_faces], dtype=np.int32)

    colors = np.tile(LABEL_COLORS[target_label], (len(cloth_v_idx), 1))
    rgba   = np.concatenate([colors, np.full((len(cloth_v_idx), 1), 255, dtype=np.uint8)], axis=1)

    mesh = trimesh.Trimesh(
        vertices=verts[cloth_v_idx],
        faces=remapped,
        vertex_colors=rgba,
        process=False,
    )
    mesh.export(out_path)
    print("  exported " + out_path + " (" + str(os.path.getsize(out_path)//1024) + " KB)")
    return out_path


def log_to_wandb(run, dataset, glb_upper, glb_lower, seg_overlay_path, upper_count, lower_count, total):
    prefix = dataset + "/garment_3d"
    wandb.log({
        prefix + "/upper_cloth":  wandb.Object3D(glb_upper),
        prefix + "/lower_cloth":  wandb.Object3D(glb_lower),
    })
    if seg_overlay_path and os.path.exists(seg_overlay_path):
        wandb.log({dataset + "/segmentation_overlay": wandb.Image(
            seg_overlay_path, caption="SegFormer single-view segmentation"
        )})
    wandb.summary.update({
        dataset + "_upper_verts": upper_count,
        dataset + "_lower_verts": lower_count,
        dataset + "_total_verts": total,
    })
    print("  logged to W&B: " + prefix + "/upper_cloth  +  " + prefix + "/lower_cloth")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",       choices=["4ddress", "actorshq"], required=True)
    ap.add_argument("--mesh_path",     required=True)
    ap.add_argument("--image_path",    required=True)
    ap.add_argument("--model_dir",     required=True)
    ap.add_argument("--output_dir",    required=True)
    # 4D-DRESS camera args
    ap.add_argument("--cameras_pkl",   default=None)
    ap.add_argument("--cam_id",        default="0004")
    # ActorsHQ camera args
    ap.add_argument("--actorshq_calib", default=None,
                    help="Path to cam_info.json (MPMAvatar format)")
    # W&B
    ap.add_argument("--wandb_run_id",  default=None,
                    help="Resume existing W&B run ID, or leave blank to create new")
    ap.add_argument("--no_wandb",      action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    # 1. Load mesh
    print("Loading mesh: " + args.mesh_path)
    verts_np, faces_np = read_obj(args.mesh_path)
    print("  verts=" + str(verts_np.shape) + "  faces=" + str(faces_np.shape))

    # 2. Load image (needed before camera so we know resolution for K scaling)
    img_pil = Image.open(args.image_path).convert("RGB")
    img_w, img_h = img_pil.size
    print("  image: " + str(img_w) + "×" + str(img_h))

    # 3. Load camera
    print("Loading camera (cam_id=" + args.cam_id + ")")
    if args.dataset == "4ddress":
        assert args.cameras_pkl, "--cameras_pkl required for 4ddress"
        K, RT = load_camera_4ddress(args.cameras_pkl, args.cam_id)
    else:
        assert args.actorshq_calib, "--actorshq_calib required for actorshq"
        K, RT = load_camera_actorshq(args.actorshq_calib, args.cam_id, img_w, img_h)

    # 4. Run SegFormer
    print("Running SegFormer...")
    seg_mask = run_segformer(img_pil, args.model_dir, device)
    unique, counts = np.unique(seg_mask, return_counts=True)
    for u, c in zip(unique, counts):
        print("  label " + str(u) + ": " + str(c) + " px")

    # Save segmentation overlay
    overlay = np.array(img_pil).copy()
    for lbl, col in [(1, LABEL_COLORS[1]), (2, LABEL_COLORS[2])]:
        m = seg_mask == lbl
        overlay[m] = (overlay[m] * 0.4 + col * 0.6).astype(np.uint8)
    overlay_path = os.path.join(args.output_dir, "seg_overlay_" + args.dataset + ".png")
    Image.fromarray(overlay).save(overlay_path)

    # 5. Backproject → 3D vertex labels
    print("Backprojecting to 3D...")
    v_labels = backproject_single_view(
        verts_np, faces_np, seg_mask, K, RT, img_w, img_h
    )

    upper_idx = np.where(v_labels == 1)[0]
    lower_idx = np.where(v_labels == 2)[0]
    print("  upper=" + str(len(upper_idx)) + "  lower=" + str(len(lower_idx)))

    # 6. Build split_idx.npz for upper and lower
    print("Building split_idx files...")
    upper_split_path = os.path.join(args.output_dir, "split_idx_upper_segformer.npz")
    lower_split_path = os.path.join(args.output_dir, "split_idx_lower_segformer.npz")
    build_split(verts_np, faces_np, upper_idx, upper_split_path, "upper")
    build_split(verts_np, faces_np, lower_idx, lower_split_path, "lower")

    # 7. Export garment-only GLB meshes
    print("Exporting garment GLBs...")
    glb_upper = os.path.join(args.output_dir, "garment_upper_" + args.dataset + ".glb")
    glb_lower = os.path.join(args.output_dir, "garment_lower_" + args.dataset + ".glb")
    export_garment_glb(verts_np, faces_np, v_labels, 1, glb_upper)
    export_garment_glb(verts_np, faces_np, v_labels, 2, glb_lower)

    # 8. W&B logging
    if not args.no_wandb:
        print("Logging to W&B (" + WANDB_ENTITY + "/" + WANDB_PROJECT + ")...")
        os.environ["WANDB_API_KEY"] = WANDB_KEY
        init_kwargs = dict(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="garment_seg_" + args.dataset,
            tags=["garment", "segformer", "single-view", args.dataset],
        )
        if args.wandb_run_id:
            init_kwargs["id"]     = args.wandb_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        log_to_wandb(run, args.dataset, glb_upper, glb_lower,
                     overlay_path, len(upper_idx), len(lower_idx), len(verts_np))
        wandb.finish()
        print("W&B run: https://wandb.ai/" + WANDB_ENTITY + "/" + WANDB_PROJECT + "/runs/" + run.id)

    print("\nOutputs:")
    print("  Upper split : " + upper_split_path)
    print("  Lower split : " + lower_split_path)
    print("  Upper GLB   : " + glb_upper)
    print("  Lower GLB   : " + glb_lower)
    print("Done.")


if __name__ == "__main__":
    main()
