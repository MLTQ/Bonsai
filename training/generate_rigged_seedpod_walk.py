"""Render a deterministic four-anchor gait with persistent semantic limbs."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


OUTLINE = (13, 31, 38, 255)
BODY = (0, 174, 163, 255)
BODY_LIGHT = (71, 226, 194, 255)
VISOR = (12, 26, 34, 255)
LEAF = (96, 205, 73, 255)
RIGHT_GOLD = (246, 185, 59, 255)
LEFT_VIOLET = (108, 92, 231, 255)
WHITE = (255, 255, 255, 255)


POSES = (
    {
        "name": "contact_right",
        "front_leg": "right",
        "right": ((0.53, 0.59), (0.64, 0.70), (0.72, 0.83), (0.80, 0.84)),
        "left": ((0.47, 0.59), (0.39, 0.70), (0.31, 0.82), (0.24, 0.84)),
    },
    {
        "name": "passing_left",
        "front_leg": "left",
        "right": ((0.53, 0.59), (0.53, 0.72), (0.50, 0.84), (0.57, 0.85)),
        "left": ((0.47, 0.59), (0.57, 0.68), (0.61, 0.77), (0.67, 0.79)),
    },
    {
        "name": "contact_left",
        "front_leg": "left",
        "right": ((0.53, 0.59), (0.42, 0.70), (0.32, 0.82), (0.24, 0.84)),
        "left": ((0.47, 0.59), (0.60, 0.70), (0.70, 0.82), (0.78, 0.84)),
    },
    {
        "name": "passing_right",
        "front_leg": "right",
        "right": ((0.53, 0.59), (0.43, 0.68), (0.39, 0.77), (0.33, 0.79)),
        "left": ((0.47, 0.59), (0.47, 0.72), (0.50, 0.84), (0.57, 0.85)),
    },
)


def _xy(point, scale):
    return tuple(round(value * scale) for value in point)


def _ellipse(draw, center, radius, scale, fill, outline=None, width=1):
    x, y = _xy(center, scale)
    rx, ry = round(radius[0] * scale), round(radius[1] * scale)
    draw.ellipse(
        (x - rx, y - ry, x + rx, y + ry),
        fill=fill, outline=outline, width=max(1, round(width * scale)),
    )


def _capsule(draw, first, second, width, scale, fill, outline=OUTLINE):
    points = [_xy(first, scale), _xy(second, scale)]
    outline_width = round((width + 0.022) * scale)
    fill_width = round(width * scale)
    draw.line(points, fill=outline, width=outline_width)
    draw.line(points, fill=fill, width=fill_width)
    radius = width / 2
    for point in (first, second):
        _ellipse(draw, point, (radius, radius), scale, fill)


def _mask_leg(points, scale):
    mask = Image.new("L", (scale, scale), 0)
    draw = ImageDraw.Draw(mask)
    hip, knee, ankle, toe = points
    width = round(0.082 * scale)
    draw.line([_xy(hip, scale), _xy(knee, scale), _xy(ankle, scale)],
              fill=255, width=width, joint="curve")
    draw.line([_xy(ankle, scale), _xy(toe, scale)],
              fill=255, width=round(0.105 * scale))
    for point, radius in ((hip, 0.041), (knee, 0.041),
                          (ankle, 0.052), (toe, 0.052)):
        _ellipse(draw, point, (radius, radius), scale, 255)
    return mask


def _leg_layer(points, color, scale):
    layer = Image.new("RGBA", (scale, scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    hip, knee, ankle, toe = points
    _capsule(draw, hip, knee, 0.067, scale, color)
    _capsule(draw, knee, ankle, 0.061, scale, color)
    _ellipse(draw, knee, (0.043, 0.043), scale, color,
             outline=OUTLINE, width=0.011)
    _capsule(draw, ankle, toe, 0.088, scale, color)
    # Boot sole and a small highlight preserve direction at NCA resolution.
    boot_draw = ImageDraw.Draw(layer)
    boot_draw.line([_xy(ankle, scale), _xy(toe, scale)], fill=OUTLINE,
                   width=max(2, round(0.010 * scale)))
    highlight = ((ankle[0] * 0.35 + toe[0] * 0.65), toe[1] - 0.018)
    _ellipse(boot_draw, highlight, (0.012, 0.009), scale, WHITE)
    return layer


def _body_layer(scale):
    layer = Image.new("RGBA", (scale, scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # A compact side-facing seedpod robot, held pixel-identical across frames.
    _capsule(draw, (0.43, 0.48), (0.31, 0.59), 0.053, scale, BODY)
    _capsule(draw, (0.57, 0.48), (0.67, 0.57), 0.053, scale, BODY)
    _ellipse(draw, (0.29, 0.60), (0.042, 0.042), scale, BODY_LIGHT,
             outline=OUTLINE, width=0.010)
    _ellipse(draw, (0.69, 0.58), (0.042, 0.042), scale, BODY_LIGHT,
             outline=OUTLINE, width=0.010)

    _ellipse(draw, (0.50, 0.51), (0.125, 0.145), scale, BODY,
             outline=OUTLINE, width=0.014)
    _ellipse(draw, (0.51, 0.55), (0.069, 0.074), scale, BODY_LIGHT,
             outline=OUTLINE, width=0.010)
    _ellipse(draw, (0.50, 0.62), (0.102, 0.043), scale, BODY,
             outline=OUTLINE, width=0.012)

    _ellipse(draw, (0.50, 0.29), (0.205, 0.190), scale, BODY,
             outline=OUTLINE, width=0.016)
    visor_box = tuple(round(value * scale) for value in (0.40, 0.235, 0.695, 0.375))
    draw.rounded_rectangle(visor_box, radius=round(0.055 * scale), fill=VISOR,
                           outline=OUTLINE, width=round(0.011 * scale))
    _ellipse(draw, (0.61, 0.30), (0.028, 0.041), scale, RIGHT_GOLD,
             outline=OUTLINE, width=0.008)
    _ellipse(draw, (0.675, 0.30), (0.018, 0.030), scale, (255, 239, 150, 255))
    _ellipse(draw, (0.34, 0.30), (0.038, 0.055), scale, BODY_LIGHT,
             outline=OUTLINE, width=0.010)
    _ellipse(draw, (0.43, 0.18), (0.034, 0.022), scale, WHITE)

    # Sprout antennae are deliberately asymmetric to fix facing direction.
    _capsule(draw, (0.48, 0.13), (0.43, 0.045), 0.022, scale, LEAF)
    _capsule(draw, (0.50, 0.13), (0.59, 0.055), 0.022, scale, LEAF)
    leaf_left = [
        _xy(point, scale) for point in
        ((0.43, 0.045), (0.38, 0.015), (0.39, 0.090), (0.46, 0.105))
    ]
    leaf_right = [
        _xy(point, scale) for point in
        ((0.59, 0.055), (0.68, 0.035), (0.63, 0.115), (0.52, 0.125))
    ]
    draw.polygon(leaf_left, fill=LEAF, outline=OUTLINE)
    draw.line(leaf_left + [leaf_left[0]], fill=OUTLINE,
              width=round(0.010 * scale), joint="curve")
    draw.polygon(leaf_right, fill=LEAF, outline=OUTLINE)
    draw.line(leaf_right + [leaf_right[0]], fill=OUTLINE,
              width=round(0.010 * scale), joint="curve")
    return layer


def render_pose(pose, size=512, supersample=3):
    scale = size * supersample
    frame = Image.new("RGBA", (scale, scale), WHITE)
    layers = {
        "right": _leg_layer(pose["right"], RIGHT_GOLD, scale),
        "left": _leg_layer(pose["left"], LEFT_VIOLET, scale),
    }
    far_leg = "left" if pose["front_leg"] == "right" else "right"
    frame.alpha_composite(layers[far_leg])
    frame.alpha_composite(_body_layer(scale))
    frame.alpha_composite(layers[pose["front_leg"]])
    frame = frame.resize((size, size), Image.Resampling.LANCZOS)
    masks = {
        name: _mask_leg(pose[name], scale).resize(
            (size, size), Image.Resampling.LANCZOS
        )
        for name in ("right", "left")
    }
    return frame, masks


def validate(frames, masks):
    measurements = []
    for frame_masks in masks:
        item = {}
        for name in ("right", "left"):
            alpha = np.asarray(frame_masks[name], dtype=np.float32) / 255.0
            _, xs = np.mgrid[:alpha.shape[0], :alpha.shape[1]]
            mass = float(alpha.sum())
            item[name] = {
                "mass": mass,
                "x": float((xs * alpha).sum() / mass / alpha.shape[1]),
            }
        measurements.append(item)
    deltas = [item["right"]["x"] - item["left"]["x"]
              for item in measurements]
    valid = (
        deltas[0] > 0.12
        and deltas[2] < -0.12
        and abs(deltas[1]) < max(abs(deltas[0]), abs(deltas[2]))
        and abs(deltas[3]) < max(abs(deltas[0]), abs(deltas[2]))
        and [pose["front_leg"] for pose in POSES]
        == ["right", "left", "left", "right"]
    )
    return {
        "valid": valid,
        "reason": (
            "named legs exchange screen order and foreground layer"
            if valid else "semantic gait contract failed"
        ),
        "right_minus_left_x": deltas,
        "measurements": measurements,
        "front_leg": [pose["front_leg"] for pose in POSES],
    }


def save_sheet(frames, path):
    size = frames[0].width
    sheet = Image.new("RGB", (size * 2, size * 2), "white")
    for index, frame in enumerate(frames):
        sheet.paste(frame.convert("RGB"), ((index % 2) * size, (index // 2) * size))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rigged_seedpod_walk")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--supersample", type=int, default=3)
    args = parser.parse_args()
    if args.size < 128 or args.supersample < 1:
        parser.error("--size must be >=128 and --supersample must be >=1")

    output = Path(args.out)
    frame_dir = output / "frames"
    right_dir, left_dir = output / "masks" / "right", output / "masks" / "left"
    for directory in (frame_dir, right_dir, left_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frames, all_masks = [], []
    for pose in POSES:
        frame, masks = render_pose(pose, args.size, args.supersample)
        frame.save(frame_dir / f"{pose['name']}.png")
        masks["right"].save(right_dir / f"{pose['name']}.png")
        masks["left"].save(left_dir / f"{pose['name']}.png")
        frames.append(frame)
        all_masks.append(masks)

    save_sheet(frames, output / "sheet.png")
    validation = validate(frames, all_masks)
    metadata = {
        **validation,
        "pose_order": [pose["name"] for pose in POSES],
        "palette": {"right": RIGHT_GOLD[:3], "left": LEFT_VIOLET[:3]},
        "size": args.size,
        "supersample": args.supersample,
        "poses": [
            {key: value for key, value in pose.items()}
            for pose in POSES
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    print(f"sheet -> {output / 'sheet.png'}")
    if not validation["valid"]:
        raise SystemExit("REJECTED: semantic gait contract failed")


if __name__ == "__main__":
    main()
