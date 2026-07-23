"""Export a fused-expert checkpoint and canonical start state for Bonsai."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from fused_expert_nca2d import EDGE_COUNT, load_fused_checkpoint
from fused_state2d import canonical_key_state
from train_cyclic import CH, FIRE_RATE
from transport_targets2d import load_cycle_frames


MAGIC = b"FX2D"
HEADER = struct.Struct("<4s7i2f2i")
WEIGHT_ORDER = (
    "pose_w1", "pose_b1", "pose_w2", "pose_b2",
    "edge_flow_w1", "edge_flow_b1", "edge_flow_w2", "edge_flow_b2",
    "edge_slot_w1", "edge_slot_b1", "edge_slot_w2", "edge_slot_b2",
    "edge_repair_w1", "edge_repair_b1", "edge_repair_w2", "edge_repair_b2",
)


def export_fused(checkpoint, target, output, state_output, transition_steps=24,
                 handoff_steps=8):
    """Write the portable FX2D weights and matching pose-zero NCS1 state."""
    model, iteration, stage = load_fused_checkpoint(checkpoint, "cpu")
    if not model.hard_slots:
        raise ValueError("the Bonsai FX2D runtime requires hard motion slots")
    if model.grid <= 1 or model.slots <= 0:
        raise ValueError("invalid fused model geometry")
    if min(transition_steps, handoff_steps) <= 0:
        raise ValueError("transition and handoff step counts must be positive")

    output = Path(output)
    state_output = Path(state_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    with output.open("wb") as stream:
        stream.write(HEADER.pack(
            MAGIC, model.grid, CH, EDGE_COUNT, model.slots,
            model.expert_hidden, model.flow_hidden, model.position_frequencies,
            model.max_flow, FIRE_RATE, transition_steps, handoff_steps,
        ))
        for name in WEIGHT_ORDER:
            values = state_dict[name].detach().contiguous().numpy().astype("<f4")
            stream.write(values.tobytes(order="C"))

    frames = load_cycle_frames(target, torch.device("cpu"))
    if frames.shape[-2:] != (model.grid, model.grid):
        raise ValueError("target grid does not match checkpoint grid")
    initial = canonical_key_state(frames, torch.tensor([0]))[0]
    cell_major = initial.permute(1, 2, 0).contiguous().numpy().astype("<f4")
    with state_output.open("wb") as stream:
        stream.write(struct.pack("<4s3i", b"NCS1", model.grid, model.grid, CH))
        stream.write(cell_major.tobytes(order="C"))

    metadata = {
        "format": MAGIC.decode("ascii"),
        "checkpoint": str(Path(checkpoint).resolve()),
        "target": str(Path(target).resolve()),
        "state": str(state_output.resolve()),
        "iteration": iteration,
        "stage": stage,
        "grid": model.grid,
        "channels": CH,
        "experts": EDGE_COUNT,
        "slots": model.slots,
        "expert_hidden": model.expert_hidden,
        "flow_hidden": model.flow_hidden,
        "position_frequencies": model.position_frequencies,
        "max_flow": model.max_flow,
        "fire_rate": FIRE_RATE,
        "transition_steps": transition_steps,
        "handoff_steps": handoff_steps,
        "weight_order": list(WEIGHT_ORDER),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--state-out", required=True)
    parser.add_argument("--transition-steps", type=int, default=24)
    parser.add_argument("--handoff-steps", type=int, default=8)
    args = parser.parse_args()
    metadata = export_fused(
        args.checkpoint, args.target, args.out, args.state_out,
        args.transition_steps, args.handoff_steps,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
