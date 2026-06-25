"""
singleview_backproject_bfs.py

Single-view garment backprojection with BFS label propagation.

Why this exists:
  A single rendered view only sees the FRONT of the mesh. The standard
  multi-view fuse (generate_predicted_cloth_vertices.py) selects faces whose
  projected area is >=face_mask_ratio masked — with one view that captures only
  a thin front sliver and never the occluded back. This script instead:

    1. Uses face_id (pixel->face) + the SegFormer upper/lower masks to seed
       per-vertex garment labels for all FRONT-VISIBLE vertices.
    2. Records which vertices were visibly BACKGROUND (front body skin) so the
       region grow cannot leak into the body.
    3. BFS-propagates each garment label across mesh edges into UNSEEN (occluded,
       i.e. back-of-garment) vertices, stopping at visible-background vertices.

  Result: a garment region that wraps front-to-back from one view, without
  swallowing the visible body.

Inputs:  mesh OBJ, face_id_000.npy, upper mask PNG, lower mask PNG
Outputs: <out_dir>/pred_upper/predicted_cloth_vertices.npy
         <out_dir>/pred_lower/predicted_cloth_vertices.npy
"""

import os
import argparse
import numpy as np
from collections import deque
from PIL import Image


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


def build_adjacency(faces, n_verts):
    adj = [set() for _ in range(n_verts)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj


def per_vertex_votes(faces, face_id, mask, label_val, votes):
    """Add votes[v, label_val] for every vertex whose face is hit by a masked pixel."""
    ys, xs = np.where(mask)
    hit_faces = face_id[ys, xs]
    hit_faces = hit_faces[hit_faces >= 0]
    if hit_faces.size == 0:
        return
    uniq, cnt = np.unique(hit_faces, return_counts=True)
    for fidx, c in zip(uniq, cnt):
        for v in faces[fidx]:
            votes[int(v), label_val] += int(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_path", required=True)
    ap.add_argument("--face_id", required=True, help="face_id_000.npy from the render")
    ap.add_argument("--upper_mask", required=True)
    ap.add_argument("--lower_mask", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mask_threshold", type=int, default=128)
    ap.add_argument("--min_seed_votes", type=int, default=1,
                    help="min masked-pixel votes for a vertex to be a garment seed")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "pred_upper"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "pred_lower"), exist_ok=True)

    verts, faces = read_obj(args.mesh_path)
    n = len(verts)
    print("mesh: " + str(n) + " verts, " + str(len(faces)) + " faces")

    face_id = np.load(args.face_id)
    H, W = face_id.shape
    upper = np.array(Image.open(args.upper_mask).convert("L").resize((W, H))) > args.mask_threshold
    lower = np.array(Image.open(args.lower_mask).convert("L").resize((W, H))) > args.mask_threshold
    print("upper mask px=" + str(int(upper.sum())) + "  lower mask px=" + str(int(lower.sum())))

    # votes[:,0]=bg, [:,1]=upper, [:,2]=lower
    votes = np.zeros((n, 3), dtype=np.int64)

    # background = visible pixels that are neither upper nor lower
    bg = (face_id >= 0) & (~upper) & (~lower)
    per_vertex_votes(faces, face_id, bg, 0, votes)
    per_vertex_votes(faces, face_id, upper, 1, votes)
    per_vertex_votes(faces, face_id, lower, 2, votes)

    seen = votes.sum(axis=1) > 0
    # a vertex is a garment seed if its garment votes dominate its bg votes
    seed_label = np.zeros(n, dtype=np.int32)  # 0 none, 1 upper, 2 lower
    garment_votes = votes[:, 1:]
    best = garment_votes.argmax(axis=1) + 1
    best_cnt = garment_votes.max(axis=1)
    is_seed = (best_cnt >= args.min_seed_votes) & (best_cnt >= votes[:, 0])
    seed_label[is_seed] = best[is_seed]

    # visible background vertices = seen, not seed, bg votes dominate -> BFS barrier
    is_visible_bg = seen & (~is_seed) & (votes[:, 0] > 0)

    print("seeds: upper=" + str(int((seed_label == 1).sum())) +
          " lower=" + str(int((seed_label == 2).sum())) +
          "  visible_bg=" + str(int(is_visible_bg.sum())) +
          "  unseen=" + str(int((~seen).sum())))

    # BFS: propagate seed labels into unseen vertices, blocked by visible_bg
    adj = build_adjacency(faces, n)
    labels = seed_label.copy()
    q = deque(np.where(seed_label > 0)[0].tolist())
    while q:
        v = q.popleft()
        for nb in adj[v]:
            if labels[nb] == 0 and not is_visible_bg[nb]:
                labels[nb] = labels[v]
                q.append(nb)

    upper_idx = np.where(labels == 1)[0].astype(np.int64)
    lower_idx = np.where(labels == 2)[0].astype(np.int64)
    print("after BFS: upper=" + str(len(upper_idx)) + "  lower=" + str(len(lower_idx)))

    np.save(os.path.join(args.out_dir, "pred_upper", "predicted_cloth_vertices.npy"), upper_idx)
    np.save(os.path.join(args.out_dir, "pred_lower", "predicted_cloth_vertices.npy"), lower_idx)
    print("wrote pred_upper / pred_lower predicted_cloth_vertices.npy")


if __name__ == "__main__":
    main()
