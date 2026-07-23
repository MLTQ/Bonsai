"""Export a phase anchor from a 2D cycle corpus as a live NCS1 state."""

import argparse
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
STATE_CHANNELS = 16


def make_state(corpus_path, frame_index=0):
    """Embed premultiplied RGBA in visible channels and alpha in hidden state."""
    payload = np.load(corpus_path, allow_pickle=True)
    if str(payload["kind"]) != "2d_cycle":
        raise ValueError("expected kind=2d_cycle")
    frames = payload["frames"].astype(np.float32)
    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[-1] != 4:
        raise ValueError(f"expected (1,F,H,W,4), got {frames.shape}")
    if not 0 <= frame_index < frames.shape[1]:
        raise ValueError(f"frame index {frame_index} outside 0..{frames.shape[1] - 1}")
    visible = frames[0, frame_index]
    if not np.isfinite(visible).all():
        raise ValueError("visible target contains non-finite values")
    if np.any(visible[..., :3] > visible[..., 3:4] + 2e-3):
        raise ValueError("target RGB is not premultiplied by alpha")
    state = np.zeros((*visible.shape[:2], STATE_CHANNELS), dtype=np.float32)
    state[..., :4] = visible
    state[..., 4:] = visible[..., 3:4]
    return state


def write_ncs1(state, path):
    """Write cell-major little-endian float32 with the runtime NCS1 header."""
    if state.ndim != 3 or state.shape[2] != STATE_CHANNELS:
        raise ValueError(f"expected (H,W,{STATE_CHANNELS}), got {state.shape}")
    height, width, channels = state.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4s3i", b"NCS1", width, height, channels))
        state.astype("<f4", copy=False).tofile(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", default=str(ROOT / "training" / "corpus_megaman_walk_v1_128.npz")
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--out", default=str(ROOT / "weights" / "megaman_walk_mature.ncs")
    )
    args = parser.parse_args()
    state = make_state(args.corpus, args.frame)
    write_ncs1(state, args.out)
    alive = int((state[..., 3] > 0.1).sum())
    print(f"wrote {args.out}: {state.shape[1]}x{state.shape[0]}x{state.shape[2]}, alive={alive}")


if __name__ == "__main__":
    main()
