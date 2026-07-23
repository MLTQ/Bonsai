"""Extract and align the authored Mega Man run sequence from a JPEG sheet."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


# The authored run loop in the third animation row. Coordinates are in the
# original 728x1279 sheet. Anchors are helmet centers, not crop centers.
FRAMES = (
    {
        "name": "flight_extension",
        "box": (282, 258, 376, 364),
        "anchor": (330, 284),
    },
    {
        "name": "support",
        "box": (378, 258, 450, 364),
        "anchor": (411, 283),
    },
    {
        "name": "compression",
        "box": (442, 258, 531, 364),
        "anchor": (479, 285),
    },
    {
        "name": "flight_recovery",
        "box": (536, 258, 625, 364),
        "anchor": (578, 282),
    },
)


def checker_distance(image, square=15):
    """Measure RGB distance from the sheet's two baked checker colors."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    first = rgb[4:12, 4:12].mean(axis=(0, 1))
    second = rgb[4:12, square + 4:square + 12].mean(axis=(0, 1))
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    parity = ((xx // square) + (yy // square)) & 1
    expected = np.where(parity[..., None] == 0, first, second)
    return np.sqrt(((rgb - expected) ** 2).sum(axis=-1))


def extract_sprite(source, distance, spec):
    """Return one RGBA sprite using checker rejection and topology repair."""
    x0, y0, x1, y1 = spec["box"]
    crop = np.asarray(source.convert("RGB"), dtype=np.uint8)[y0:y1, x0:x1]
    evidence = distance[y0:y1, x0:x1]
    chroma = crop.max(axis=-1).astype(np.int16) - crop.min(axis=-1).astype(np.int16)
    intensity = crop.astype(np.float32).mean(axis=-1)
    # Checker seams can differ from the ideal template after JPEG compression,
    # but remain neutral and bright. Sprite seed pixels are colored or dark.
    support = (evidence > 18.0) & ((chroma > 11) | (intensity < 190.0))
    support = ndimage.binary_closing(support, iterations=1)
    labels, count = ndimage.label(support)
    if not count:
        raise ValueError(f"no sprite found in {spec['name']}")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == sizes.argmax()
    # Checker-white glove interiors have no color evidence in a JPEG; their
    # dark authored outlines make them enclosed holes that can be restored.
    component = ndimage.binary_fill_holes(component)
    component = ndimage.binary_closing(component, iterations=1)
    alpha = ndimage.gaussian_filter(component.astype(np.float32), sigma=0.55)
    alpha = np.clip(alpha, 0.0, 1.0)
    rgba = np.concatenate((crop, np.round(alpha[..., None] * 255).astype(np.uint8)), axis=-1)
    return Image.fromarray(rgba, "RGBA")


def align_sprite(sprite, spec, cell_size=144, target_anchor=(72, 38)):
    """Place a sprite by its authored helmet anchor on a shared canvas."""
    x0, y0, _, _ = spec["box"]
    local_anchor = (spec["anchor"][0] - x0, spec["anchor"][1] - y0)
    offset = (target_anchor[0] - local_anchor[0],
              target_anchor[1] - local_anchor[1])
    canvas = Image.new("RGBA", (cell_size, cell_size), (0, 0, 0, 0))
    canvas.alpha_composite(sprite, offset)
    return canvas


def white_composite(frame):
    background = Image.new("RGBA", frame.size, "white")
    background.alpha_composite(frame)
    return background.convert("RGB")


def save_sheet(frames, path):
    size = frames[0].width
    sheet = Image.new("RGB", (size * 2, size * 2), "white")
    for index, frame in enumerate(frames):
        sheet.paste(white_composite(frame), ((index % 2) * size, (index // 2) * size))
    sheet.save(path)


def save_animation(frames, path, scale=4, duration_ms=120):
    """Write a large nearest-neighbor loop for mandatory motion review."""
    rendered = [
        white_composite(frame).resize(
            (frame.width * scale, frame.height * scale), Image.Resampling.NEAREST
        )
        for frame in frames
    ]
    rendered[0].save(
        path, save_all=True, append_images=rendered[1:], duration=duration_ms,
        loop=0, disposal=2,
    )


def diagnostics(frames):
    rgba = np.stack([
        np.asarray(frame, dtype=np.float32) / 255.0 for frame in frames
    ])
    premult = rgba.copy()
    premult[..., :3] *= premult[..., 3:4]
    adjacent = []
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
        adjacent.append(float(((premult[first] - premult[second]) ** 2).mean()))
    opposite = [
        float(((premult[0] - premult[2]) ** 2).mean()),
        float(((premult[1] - premult[3]) ** 2).mean()),
    ]
    return {
        "adjacent_mse": adjacent,
        "opposite_mse": opposite,
        "alpha_pixels": [int((frame[..., 3] > 0.05).sum()) for frame in rgba],
        "note": "numeric uniqueness is not a semantic gait acceptance test",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet", nargs="?", default="sheets/megaman")
    parser.add_argument("--out", default="experiments/megaman_walk_extract")
    parser.add_argument("--cell-size", type=int, default=144)
    args = parser.parse_args()
    if args.cell_size < 128:
        parser.error("--cell-size must be at least 128")

    source = Image.open(args.sheet).convert("RGB")
    if source.size != (728, 1279):
        parser.error(f"expected the reviewed 728x1279 sheet, got {source.size}")
    distance = checker_distance(source)
    output = Path(args.out)
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for spec in FRAMES:
        sprite = extract_sprite(source, distance, spec)
        frame = align_sprite(sprite, spec, args.cell_size)
        frame.save(frame_dir / f"{spec['name']}.png")
        frames.append(frame)
    save_sheet(frames, output / "sheet.png")
    save_animation(frames, output / "loop.gif")
    report = {
        "source": str(Path(args.sheet).resolve()),
        "pose_order": [spec["name"] for spec in FRAMES],
        "frames": list(FRAMES),
        **diagnostics(frames),
    }
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"sheet -> {output / 'sheet.png'}")
    print(f"loop -> {output / 'loop.gif'}")


if __name__ == "__main__":
    main()
