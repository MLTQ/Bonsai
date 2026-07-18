"""Assemble a directory of frame_NNN.png files (from `Bonsai --render-seq`) into a GIF.

Usage: python3 frames_to_gif.py <framesdir> <out.gif> [ms_per_frame=80]
"""

import sys
from pathlib import Path

from PIL import Image

frames_dir = Path(sys.argv[1])
out = sys.argv[2]
ms = int(sys.argv[3]) if len(sys.argv) > 3 else 80

paths = sorted(frames_dir.glob("frame_*.png"))
if not paths:
    sys.exit(f"no frame_*.png in {frames_dir}")

frames = []
for p in paths:
    img = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", img.size, (24, 24, 24, 255))  # dark bg so alpha edges read
    frames.append(Image.alpha_composite(bg, img).convert("P", palette=Image.ADAPTIVE))

frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0,
               disposal=2)
print(f"wrote {out} ({len(paths)} frames @ {ms}ms)")
