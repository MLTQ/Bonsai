"""Generate four-pose diffusion sheets for the 2D transport experiment."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


PROMPT = (
    "masterpiece, best quality, no humans, chibi seed robot animation sheet, "
    "exactly 4 images in a 2x2 grid, same character, full body, strict side view "
    "facing right, 4 walking gait poses, round teal seed body, leaf antenna, "
    "gold near boot, teal far boot, flat colors, dark outline, plain white "
    "background, generous padding, no shadows"
)

NEGATIVE_PROMPT = (
    "human, person, realistic, text, letters, watermark, more than 4 images, "
    "uneven grid, cropped, front view, back view, different character, extra legs, "
    "missing legs, merged legs, scenery, floor, shadow, gradient background"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="transport2d_generation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23, 29])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=6.5)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.checkpoint, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    metadata = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "steps": args.steps,
        "guidance": args.guidance,
        "seeds": args.seeds,
        "layout": "2x2 row-major: contact-gold, passing, contact-teal, passing",
    }
    (output / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for seed in args.seeds:
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=1024,
            height=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        path = output / f"seedpod_walk_sheet_{seed}.png"
        image.save(path)
        print(f"saved {path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
