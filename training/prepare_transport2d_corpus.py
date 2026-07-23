"""Prepare a reviewed 2x2 diffusion gait sheet for 2D NCA training."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_NAMES = ("contact_gold", "passing_gold", "contact_teal", "passing_teal")


def split_sheet(image):
    """Return four equal cells in row-major order."""
    width, height = image.size
    if width % 2 or height % 2:
        raise ValueError("sheet width and height must both be divisible by two")
    cell_width, cell_height = width // 2, height // 2
    return [
        image.crop((column * cell_width, row * cell_height,
                    (column + 1) * cell_width, (row + 1) * cell_height))
        for row in range(2)
        for column in range(2)
    ]


def key_near_white(image, threshold=228, soft=25):
    """Return straight RGBA float data with near-white converted to alpha."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    minimum_channel = rgba[..., :3].min(axis=-1) * 255.0
    matte = np.clip((threshold - minimum_channel) / soft, 0.0, 1.0)
    rgba[..., 3] = np.minimum(rgba[..., 3], matte)
    return rgba


def keep_largest_component(frame):
    """Remove detached foreground specks while retaining antialiased edges."""
    from scipy import ndimage

    support = frame[..., 3] > 0.05
    labels, count = ndimage.label(
        support, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count <= 1:
        return frame
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = labels == sizes.argmax()
    keep = ndimage.binary_dilation(largest, iterations=2)
    filtered = frame.copy()
    filtered[..., 3] *= keep
    return filtered


def _union_box(frames, pad_fraction=0.04):
    support = np.stack([frame[..., 3] > 0.02 for frame in frames]).any(axis=0)
    ys, xs = np.nonzero(support)
    if not len(xs):
        raise ValueError("white-keying removed the entire sheet")
    height, width = support.shape
    pad = max(2, int(round(max(height, width) * pad_fraction)))
    x0, x1 = max(0, xs.min() - pad), min(width, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(height, ys.max() + pad + 1)
    if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
        raise ValueError("foreground reaches a cell edge; reject this candidate")
    return x0, y0, x1, y1


def prepare_frames(image, grid=128, threshold=228, soft=25, keep_largest=True):
    """Key, union-crop, and consistently place the four gait anchors."""
    keyed = [key_near_white(cell, threshold, soft) for cell in split_sheet(image)]
    if keep_largest:
        keyed = [keep_largest_component(frame) for frame in keyed]
    x0, y0, x1, y1 = _union_box(keyed)
    crop_width, crop_height = x1 - x0, y1 - y0
    available = grid - 8
    scale = min(available / crop_width, available / crop_height)
    out_width = max(1, int(round(crop_width * scale)))
    out_height = max(1, int(round(crop_height * scale)))
    frames = []
    straight_pngs = []
    for frame in keyed:
        crop = np.clip(frame[y0:y1, x0:x1] * 255.0, 0, 255).astype(np.uint8)
        resized = Image.fromarray(crop, "RGBA").resize(
            (out_width, out_height), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGBA", (grid, grid), (0, 0, 0, 0))
        offset = ((grid - out_width) // 2, (grid - out_height) // 2)
        canvas.alpha_composite(resized, offset)
        straight = np.asarray(canvas, dtype=np.float32) / 255.0
        premultiplied = straight.copy()
        premultiplied[..., :3] *= premultiplied[..., 3:4]
        frames.append(premultiplied)
        straight_pngs.append(canvas)
    return np.stack(frames), straight_pngs


def save_preview(frames, path, scale=3):
    strip = Image.new("RGBA", (sum(frame.width for frame in frames), frames[0].height))
    x = 0
    for frame in frames:
        strip.alpha_composite(frame, (x, 0))
        x += frame.width
    strip.resize((strip.width * scale, strip.height * scale),
                 Image.Resampling.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--threshold", type=int, default=228)
    parser.add_argument("--soft", type=int, default=25)
    parser.add_argument("--keep-all-components", action="store_true")
    parser.add_argument(
        "--frame-names", nargs=4, default=FRAME_NAMES,
        metavar=("F0", "F1", "F2", "F3"),
    )
    args = parser.parse_args()
    if args.grid < 32:
        parser.error("--grid must be at least 32")

    sheet = Image.open(args.sheet).convert("RGB")
    frames, pngs = prepare_frames(
        sheet, args.grid, args.threshold, args.soft,
        keep_largest=not args.keep_all_components,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        kind="2d_cycle",
        frames=frames[None].astype(np.float16),
        frame_names=np.array(args.frame_names),
        source=str(Path(args.sheet).resolve()),
    )
    frame_dir = output.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(args.frame_names, pngs):
        frame.save(frame_dir / f"{name}.png")
    preview = output.with_name(output.stem + "_preview.png")
    save_preview(pngs, preview)
    print(f"2d cycle {(1, *frames.shape)} -> {output}")
    print(f"preview -> {preview}")


if __name__ == "__main__":
    main()
