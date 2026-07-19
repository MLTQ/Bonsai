"""Builds the 2D state-space map for the manifold explorer panel.

Samples the creature's z-space with deliberate structure — anchor clusters,
anchor-to-anchor interpolation paths (the roads between moods), and a uniform
sprinkle — then embeds to 2D with UMAP (PCA fallback) and exports JSON the
Swift panel uses for display and kNN inversion (2D drag -> 10D z).

Usage: python3 make_statemap.py [--out ../weights/statemap_shoggoth3d.json]
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import argparse as _ap
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--creature", default="3d", choices=["2d", "3d"])
_known, _ = _pre.parse_known_args()
if _known.creature == "3d":
    from manifold_shoggoth3d import ANCHORS, ZDIM  # noqa: E402
else:
    from manifold_shoggoth import ANCHORS, ZDIM  # noqa: E402


def build_samples(rng):
    anchor_names = list(ANCHORS)
    anchor_z = np.array([ANCHORS[n] for n in anchor_names], dtype=np.float32)

    zs = [anchor_z]
    # anchor clusters: jittered replicates so each mood is a visible island
    for a in anchor_z:
        zs.append(np.clip(a + rng.normal(0, 0.05, (30, ZDIM)), 0, 1).astype(np.float32))
    # roads: interpolation paths between every anchor pair
    for i, j in itertools.combinations(range(len(anchor_z)), 2):
        t = rng.random((14, 1)).astype(np.float32)
        road = anchor_z[i] * (1 - t) + anchor_z[j] * t
        zs.append(np.clip(road + rng.normal(0, 0.02, road.shape), 0, 1).astype(np.float32))
    # sprinkle: the wilderness between the roads
    zs.append(rng.random((400, ZDIM), dtype=np.float32))
    return anchor_names, anchor_z, np.concatenate(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", default="3d", choices=["2d", "3d"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(__file__), "..", "weights",
                                f"statemap_shoggoth{args.creature}.json")

    rng = np.random.default_rng(7)
    anchor_names, anchor_z, zs = build_samples(rng)

    try:
        import umap
        emb = umap.UMAP(n_neighbors=25, min_dist=0.25, random_state=7).fit_transform(zs)
        method = "umap"
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(zs)
        method = "pca"

    # normalize to [0,1]^2 for the panel
    emb = (emb - emb.min(axis=0)) / (emb.max(axis=0) - emb.min(axis=0) + 1e-9)

    out = {
        "method": method,
        "points": np.round(emb, 4).tolist(),
        "z": np.round(zs, 4).tolist(),
        "anchors": {name: np.round(emb[i], 4).tolist() for i, name in enumerate(anchor_names)},
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"{method} map: {len(zs)} points -> {args.out}")


if __name__ == "__main__":
    main()
