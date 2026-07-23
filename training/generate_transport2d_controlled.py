"""Pose-controlled four-frame diffusion generation with semantic leg validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import (ControlNetModel,
                       StableDiffusionXLControlNetImg2ImgPipeline)
from PIL import Image, ImageDraw


BASE_PROMPT = (
    "masterpiece, best quality, no humans, same cute chibi sprout robot, strict "
    "side profile facing right, full body, two separate articulated legs, right "
    "leg and boot golden yellow, left leg and boot dark teal, round teal helmet, "
    "two green leaf antennae, flat clean anime game sprite, thick dark outline, "
    "plain white background, isolated"
)
NEGATIVE_PROMPT = (
    "human, person, front view, back view, three-quarter view, both boots same "
    "color, merged legs, extra legs, missing legs, detached limbs, text, motion "
    "lines, floor, shadow, scenery, gray background"
)

# OpenPose order: nose, neck, R shoulder/elbow/wrist, L shoulder/elbow/wrist,
# R hip/knee/ankle, L hip/knee/ankle, R eye, L eye, R ear, L ear.
BASE_POINTS = [
    (0.56, 0.18), (0.50, 0.31),
    (0.51, 0.34), None, None,
    (0.48, 0.35), None, None,
    (0.51, 0.55), None, None,
    (0.48, 0.56), None, None,
    None, None, None, None,
]

POSES = (
    ("contact_right", {
        3: (0.43, 0.42), 4: (0.34, 0.50),
        6: (0.59, 0.43), 7: (0.68, 0.50),
        9: (0.61, 0.69), 10: (0.71, 0.85),
        12: (0.39, 0.70), 13: (0.28, 0.85),
    }),
    ("passing_left", {
        3: (0.45, 0.43), 4: (0.39, 0.52),
        6: (0.57, 0.42), 7: (0.64, 0.48),
        9: (0.56, 0.70), 10: (0.58, 0.86),
        12: (0.59, 0.65), 13: (0.53, 0.78),
    }),
    ("contact_left", {
        3: (0.59, 0.43), 4: (0.68, 0.50),
        6: (0.43, 0.42), 7: (0.34, 0.50),
        9: (0.39, 0.70), 10: (0.28, 0.85),
        12: (0.61, 0.69), 13: (0.71, 0.85),
    }),
    ("passing_right", {
        3: (0.57, 0.42), 4: (0.64, 0.48),
        6: (0.45, 0.43), 7: (0.39, 0.52),
        9: (0.41, 0.65), 10: (0.47, 0.78),
        12: (0.44, 0.70), 13: (0.42, 0.86),
    }),
)

LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
)


def pose_points(overrides):
    points = list(BASE_POINTS)
    for index, point in overrides.items():
        points[index] = point
    return points


def make_pose_map(points, size=768):
    """Render a standard colored OpenPose body map on black."""
    image = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(image)
    width = max(5, round(size / 96))
    for (first, second), color in zip(LIMBS, COLORS):
        if points[first] is None or points[second] is None:
            continue
        xy = tuple(
            (round(point[0] * size), round(point[1] * size))
            for point in (points[first], points[second])
        )
        dimmed = tuple(round(channel * 0.6) for channel in color)
        draw.line(xy, fill=dimmed, width=width, joint="curve")
    radius = width
    for point, color in zip(points, COLORS):
        if point is None:
            continue
        x, y = round(point[0] * size), round(point[1] * size)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def load_pipeline(checkpoint, controlnet_path, offload="auto"):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16, use_safetensors=True
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        checkpoint, controlnet=controlnet, torch_dtype=torch.float16
    )
    if offload == "auto":
        gpu_bytes = torch.cuda.get_device_properties(0).total_memory
        offload = "model" if gpu_bytes >= 16 * 1024**3 else "sequential"
    if offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        # SDXL + its full ControlNet exceed 8 GB when whole components move to
        # the GPU together. Layer-wise offload keeps the 2070S viable.
        pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()
    return pipe


def leg_centroids(image):
    """Return lower-body gold/teal horizontal centroids and pixel counts."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    y, x = np.mgrid[:height, :width]
    # Exclude the torso so the palette accents there cannot fake a leg swap.
    lower = y > height * 0.60
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gold = lower & (red > 0.55) & (green > 0.32) & (blue < 0.38) & ((red - blue) > 0.28)
    teal = lower & (red < 0.40) & (green > 0.28) & (blue > 0.25) & ((green - red) > 0.05)
    result = {}
    for name, mask in (("gold", gold), ("teal", teal)):
        count = int(mask.sum())
        result[name] = {
            "count": count,
            "x": float(x[mask].mean() / width) if count else float("nan"),
        }
    return result


def validate_leg_swap(frames):
    measurements = [leg_centroids(frame) for frame in frames]
    for index, measurement in enumerate(measurements):
        if min(measurement["gold"]["count"], measurement["teal"]["count"]) < 80:
            return False, measurements, f"frame {index} lacks a distinct gold or teal leg"
    deltas = [item["gold"]["x"] - item["teal"]["x"] for item in measurements]
    margin = 0.06
    if not (deltas[0] > margin and deltas[2] < -margin):
        return False, measurements, (
            f"contact order did not swap: delta0={deltas[0]:+.3f}, "
            f"delta2={deltas[2]:+.3f}"
        )
    if abs(deltas[1]) >= max(abs(deltas[0]), abs(deltas[2])):
        return False, measurements, "first passing pose is not between contact extremes"
    if abs(deltas[3]) >= max(abs(deltas[0]), abs(deltas[2])):
        return False, measurements, "second passing pose is not between contact extremes"
    return True, measurements, "semantic gold/teal leg order swaps across contact poses"


def save_sheet(frames, path):
    width, height = frames[0].size
    sheet = Image.new("RGB", (width * 2, height * 2), "white")
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % 2) * width, (index // 2) * height))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hero")
    parser.add_argument("--checkpoint")
    parser.add_argument("--controlnet")
    parser.add_argument("--out", default="transport2d_controlled")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--strength", type=float, default=0.82)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--control-scale", type=float, default=1.15)
    parser.add_argument(
        "--offload", choices=("auto", "model", "sequential"), default="auto",
        help="auto uses faster model offload on >=16 GB GPUs",
    )
    parser.add_argument(
        "--poses-only", action="store_true",
        help="render the four OpenPose controls without loading diffusion models",
    )
    args = parser.parse_args()

    output = Path(args.out)
    pose_dir, frame_dir = output / "poses", output / "frames"
    pose_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    controls = []
    for name, overrides in POSES:
        control = make_pose_map(pose_points(overrides))
        control.save(pose_dir / f"{name}.png")
        controls.append(control)
    save_sheet(controls, output / "pose_controls_sheet.png")
    if args.poses_only:
        print(f"pose controls -> {output / 'pose_controls_sheet.png'}", flush=True)
        return
    missing = [
        flag for flag, value in (
            ("--hero", args.hero),
            ("--checkpoint", args.checkpoint),
            ("--controlnet", args.controlnet),
        ) if not value
    ]
    if missing:
        parser.error(f"required unless --poses-only: {', '.join(missing)}")

    hero = Image.open(args.hero).convert("RGB").resize((768, 768), Image.Resampling.LANCZOS)
    pipe = load_pipeline(args.checkpoint, args.controlnet, args.offload)

    frames = []
    for (name, _), control in zip(POSES, controls):
        generator = torch.Generator("cuda").manual_seed(args.seed)
        frame = pipe(
            prompt=BASE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=hero,
            control_image=control,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.control_scale,
            control_guidance_start=0.0,
            control_guidance_end=0.92,
            generator=generator,
        ).images[0]
        frame.save(frame_dir / f"{name}.png")
        frames.append(frame)
        print(f"saved {name}", flush=True)
    sheet_path = output / "poses_sheet.png"
    save_sheet(frames, sheet_path)
    valid, measurements, reason = validate_leg_swap(frames)
    validation = {
        "valid": valid,
        "reason": reason,
        "measurements": measurements,
        "pose_order": [name for name, _ in POSES],
        "seed": args.seed,
        "strength": args.strength,
        "steps": args.steps,
        "guidance": args.guidance,
        "control_scale": args.control_scale,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)
    print(f"sheet -> {sheet_path}", flush=True)
    if not valid:
        raise SystemExit(f"REJECTED: {reason}")


if __name__ == "__main__":
    main()
