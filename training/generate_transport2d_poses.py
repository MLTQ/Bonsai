"""Repaint four gait anchors from one approved diffusion hero."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


BASE_PROMPT = (
    "masterpiece, best quality, no humans, same chibi seed robot, full body, "
    "walking, teal round head, two leaf antennae, gold boots, black joints, "
    "connected legs, flat colors, dark outline, plain white background, isolated, "
    "no shadow"
)
NEGATIVE_PROMPT = (
    "human, person, realistic, text, watermark, different character, extra legs, "
    "missing legs, merged legs, detached limbs, floating boot, cropped, scenery, "
    "floor, shadow, gray background"
)
POSES = (
    ("contact_gold", (-28, 4), (28, -2), "wide contact pose, left boot forward, right boot back"),
    ("passing_gold", (-3, 1), (18, -22), "passing pose, left boot planted, right knee lifted"),
    ("contact_teal", (28, -2), (-28, 4), "opposite wide contact pose, right boot forward, left boot back"),
    ("passing_teal", (18, -22), (-3, 1), "opposite passing pose, left knee lifted, right boot planted"),
)


def extract_hero(sheet, crop_box, canvas_size=768, border_distance=38, soft=18):
    """Extract one cell from its border color and center it on white."""
    crop = np.asarray(sheet.crop(crop_box).convert("RGB"), dtype=np.float32)
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(crop - background, axis=-1)
    alpha = np.clip((distance - border_distance) / soft, 0.0, 1.0)
    support = alpha > 0.05
    ys, xs = np.nonzero(support)
    if not len(xs):
        raise ValueError("hero extraction found no foreground")
    pad = 12
    x0, x1 = max(0, xs.min() - pad), min(crop.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(crop.shape[0], ys.max() + pad + 1)
    rgba = np.concatenate((crop / 255.0, alpha[..., None]), axis=-1)[y0:y1, x0:x1]
    subject = Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8), "RGBA")
    available = int(canvas_size * 0.86)
    scale = min(available / subject.width, available / subject.height)
    subject = subject.resize(
        (max(1, round(subject.width * scale)), max(1, round(subject.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (canvas_size, canvas_size), "white")
    offset = ((canvas_size - subject.width) // 2, (canvas_size - subject.height) // 2)
    canvas.alpha_composite(subject, offset)
    return canvas.convert("RGB")


def make_pose_guide(hero, left_shift, right_shift, cutoff_fraction=0.68):
    """Continuously warp two lower-half regions into a diffusion pose guide."""
    rgb = np.asarray(hero.convert("RGB"), dtype=np.float32)
    distance = np.linalg.norm(rgb - 255.0, axis=-1)
    alpha = np.clip((distance - 3.0) / 18.0, 0.0, 1.0)
    rgba_data = np.concatenate((rgb / 255.0, alpha[..., None]), axis=-1)
    rgba = Image.fromarray(
        np.clip(rgba_data * 255.0, 0, 255).astype(np.uint8), "RGBA"
    )
    width, height = rgba.size
    cutoff = int(height * cutoff_fraction)
    center = width // 2
    guide = Image.new("RGBA", rgba.size, "white")
    guide.alpha_composite(rgba.crop((0, 0, width, cutoff)), (0, 0))

    lower_height = height - cutoff
    left = Image.new("RGBA", (width, lower_height), (0, 0, 0, 0))
    right = Image.new("RGBA", (width, lower_height), (0, 0, 0, 0))
    left.alpha_composite(rgba.crop((0, cutoff, center, height)), (0, 0))
    right.alpha_composite(rgba.crop((center, cutoff, width, height)), (center, 0))

    def warp_piece(piece, shift):
        horizontal, vertical = shift
        warped_height = max(8, lower_height + vertical)
        piece = piece.resize((width, warped_height), Image.Resampling.LANCZOS)
        shear = horizontal / max(warped_height - 1, 1)
        piece = piece.transform(
            (width, warped_height),
            Image.Transform.AFFINE,
            (1.0, -shear, 0.0, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BICUBIC,
        )
        layer = Image.new("RGBA", (width, lower_height), (0, 0, 0, 0))
        layer.alpha_composite(piece, (0, 0))
        return layer

    guide.alpha_composite(warp_piece(left, left_shift), (0, cutoff))
    guide.alpha_composite(warp_piece(right, right_shift), (0, cutoff))
    return guide.convert("RGB")


def save_sheet(frames, path):
    width, height = frames[0].size
    sheet = Image.new("RGB", (width * 2, height * 2), "white")
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % 2) * width, (index // 2) * height))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet")
    parser.add_argument("--crop", type=int, nargs=4, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="transport2d_poses")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--strength", type=float, default=0.58)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=6.0)
    args = parser.parse_args()

    output = Path(args.out)
    guide_dir, frame_dir = output / "guides", output / "frames"
    guide_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    hero = extract_hero(Image.open(args.sheet), tuple(args.crop))
    hero.save(output / "hero.png")

    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        args.checkpoint, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    frames = []
    for name, left_shift, right_shift, suffix in POSES:
        guide = make_pose_guide(hero, left_shift, right_shift)
        guide.save(guide_dir / f"{name}.png")
        generator = torch.Generator("cuda").manual_seed(args.seed)
        frame = pipe(
            prompt=f"{BASE_PROMPT}, {suffix}",
            negative_prompt=NEGATIVE_PROMPT,
            image=guide,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        frame.save(frame_dir / f"{name}.png")
        frames.append(frame)
        print(f"saved {name}", flush=True)
    save_sheet(frames, output / "poses_sheet.png")
    metadata = {
        "source": str(Path(args.sheet).resolve()),
        "crop": args.crop,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": args.seed,
        "strength": args.strength,
        "steps": args.steps,
        "guidance": args.guidance,
        "base_prompt": BASE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "pose_order": [pose[0] for pose in POSES],
    }
    (output / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"sheet -> {output / 'poses_sheet.png'}", flush=True)


if __name__ == "__main__":
    main()
