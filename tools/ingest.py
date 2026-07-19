"""Asset ingestion: turn artist-made images and 3D models into NCA training targets.

The escape hatch from primitive-based modeling — draw a sprite or sculpt/pose a
mesh in any normal tool, export, and this produces the target arrays the
trainers consume (via their --target flag).

Subcommands
  image     one PNG              -> 2D static target        (creature.npz, kind=2d)
  sheet     spritesheet PNG      -> 2D cycle frames         (kind=2d_cycle)
            (--frames N horizontal tiles; --behaviors M rows)
  mesh      one .glb/.obj/.ply   -> voxelized 3D target     (kind=3d)
  meshcycle directory of meshes  -> 3D cycle frames         (kind=3d_cycle)
            (frames sorted by filename: pose_00.glb, pose_01.glb, ...)

Blender workflow for animated creatures: pose your model at 12 evenly spaced
points of a loop, export each as glTF (with vertex colors or a baked texture),
drop them in a directory, run meshcycle. Loop closure is on you: frame 11 must
flow back into frame 0.

Voxelization: surface sampled at the target resolution, colors from the mesh
visual (texture UV or vertex colors), interior filled with a darkened body
color so the creature is solid, alpha=1 inside, soft at the shell.
"""

import argparse
import glob
import os
import sys

import numpy as np

GRID2 = int(os.environ.get("BONSAI_GRID2", "64"))
GRID3 = int(os.environ.get("BONSAI_GRID3", "32"))


def _premul(rgba):
    out = rgba.astype(np.float32)
    out[..., :3] *= out[..., 3:4]
    return out


def _key_white(img, threshold=238, soft=26):
    """Near-white background -> transparency, with a soft edge band."""
    import numpy as np
    from PIL import Image

    a = np.asarray(img.convert("RGBA"), dtype=np.float32)
    lum = a[..., :3].min(axis=-1)  # white needs ALL channels high
    alpha = np.clip((threshold - lum) / soft, 0.0, 1.0)
    a[..., 3] = np.minimum(a[..., 3], alpha * 255.0)
    return Image.fromarray(a.astype("uint8"), "RGBA")


def load_image(path, grid, key_white=False):
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    if key_white:
        img = _key_white(img)
    img.thumbnail((grid - 8, grid - 8), Image.LANCZOS)  # margin for growth overshoot
    canvas = Image.new("RGBA", (grid, grid), (0, 0, 0, 0))
    canvas.paste(img, ((grid - img.width) // 2, (grid - img.height) // 2))
    return _premul(np.asarray(canvas, dtype=np.float32) / 255.0)


def ingest_image(args):
    target = load_image(args.input, GRID2, key_white=args.key_white)
    np.savez_compressed(args.out, kind="2d", target=target)
    print(f"2d target {target.shape} -> {args.out}")


def ingest_sheet(args):
    from PIL import Image

    img = Image.open(args.input).convert("RGBA")
    fw, fh = img.width // args.frames, img.height // args.behaviors
    frames = np.zeros((args.behaviors, args.frames, GRID2, GRID2, 4), np.float32)
    for b in range(args.behaviors):
        for f in range(args.frames):
            tile = img.crop((f * fw, b * fh, (f + 1) * fw, (b + 1) * fh))
            tmp = f"/tmp/_bonsai_tile.png"
            tile.save(tmp)
            frames[b, f] = load_image(tmp, GRID2, key_white=getattr(args, "key_white", False))
    np.savez_compressed(args.out, kind="2d_cycle", frames=frames.astype(np.float16))
    print(f"2d cycle {frames.shape} -> {args.out}")


def voxelize_mesh(path, grid):
    """Mesh -> (grid, grid, grid, 4) float32, (z, y, x, c), y-up, premultiplied."""
    import trimesh
    from scipy import ndimage

    loaded = trimesh.load(path, force="mesh")
    mesh = loaded if isinstance(loaded, trimesh.Trimesh) else loaded.to_mesh()

    # normalize into the grid with margin
    margin = grid * 0.12
    scale = (grid - 2 * margin) / max(mesh.extents)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(scale)
    mesh.apply_translation([grid / 2] * 3)

    # surface occupancy + color by sampling points on faces
    occ = np.zeros((grid, grid, grid), bool)
    col = np.zeros((grid, grid, grid, 3), np.float32)
    cnt = np.zeros((grid, grid, grid), np.int32)
    n_samples = max(60000, grid ** 2 * 30)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    colors = None
    vis = mesh.visual
    try:
        # texture -> per-vertex colors first, so UV-mapped assets work
        if getattr(vis, "uv", None) is not None:
            vis = vis.to_color()
    except Exception:
        pass
    vc = getattr(vis, "vertex_colors", None)
    fc = getattr(vis, "face_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices) and np.ptp(np.asarray(vc)[:, :3]) > 0:
        colors = np.asarray(vc)[mesh.faces[face_idx]].mean(axis=1)[:, :3] / 255.0
    elif fc is not None and np.ptp(np.asarray(fc)[:, :3]) > 0:
        colors = np.asarray(fc)[face_idx, :3] / 255.0
    if colors is None:
        colors = np.full((len(pts), 3), 0.6, np.float32)
    ijk = np.clip(pts.astype(int), 0, grid - 1)
    for (x, y, z), c in zip(ijk, colors):
        occ[x, y, z] = True
        col[x, y, z] += c
        cnt[x, y, z] += 1
    col[cnt > 0] /= cnt[cnt > 0, None]

    # solid interior: fill holes, color it a darkened mean body color
    filled = ndimage.binary_fill_holes(occ)
    interior = filled & ~occ
    body = col[cnt > 0].mean(axis=0) * 0.55 if (cnt > 0).any() else np.array([0.3, 0.3, 0.3])
    col[interior] = body

    vol = np.zeros((grid, grid, grid, 4), np.float32)
    vol[..., :3] = col
    vol[..., 3] = ndimage.gaussian_filter(filled.astype(np.float32), 0.6)  # soft shell
    vol[..., :3] *= vol[..., 3:4]
    # trimesh axes (x, y, z) -> our (z, y, x, c)
    return np.ascontiguousarray(vol.transpose(2, 1, 0, 3))


def ingest_mesh(args):
    target = voxelize_mesh(args.input, GRID3)
    np.savez_compressed(args.out, kind="3d", target=target)
    print(f"3d target {target.shape}, occupied {(target[..., 3] > 0.1).sum()} vox -> {args.out}")


def ingest_meshcycle(args):
    paths = sorted(glob.glob(os.path.join(args.input, "*")))
    paths = [p for p in paths if os.path.splitext(p)[1].lower() in
             (".glb", ".gltf", ".obj", ".ply", ".stl")]
    if not paths:
        sys.exit(f"no mesh files in {args.input}")
    frames = np.stack([voxelize_mesh(p, GRID3) for p in paths])
    np.savez_compressed(args.out, kind="3d_cycle",
                        frames=frames[None].astype(np.float16))  # (1 behavior, F, ...)
    print(f"3d cycle {frames.shape} from {len(paths)} poses -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("image", ingest_image), ("sheet", ingest_sheet),
                     ("mesh", ingest_mesh), ("meshcycle", ingest_meshcycle)):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("--out", default="creature.npz")
        if name in ("image", "sheet"):
            p.add_argument("--key-white", action="store_true",
                           help="convert near-white background to transparency")
        if name == "sheet":
            p.add_argument("--frames", type=int, default=12)
            p.add_argument("--behaviors", type=int, default=1)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
