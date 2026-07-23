# fused_state2d.py

## Purpose

Defines the explicit recurrent-state interface shared by independently
parameterized pose and transition experts. It prevents each expert from
inventing an incompatible hidden-channel language.

## Components

### `canonical_key_state`
- **Does**: Converts one of four RGBA anchors into a 16-channel state containing
  visible RGBA, support, masked coordinates, alpha gradients, one-cell soft support,
  one-hot pose code, luminance, and saturation.
- **Interacts with**: `train_fused_expert2d.py` source/destination states and
  `FusedExpertNCA2D` recurrent state.
- **Rationale**: The representation is deterministic and contains no
  independently learned channel permutation, making every expert interoperable.
  Hidden support is restricted to the one-cell life-mask neighborhood so a
  zero-update pose expert preserves a key state exactly.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Fused trainer | Output `(B,16,H,W)` with input RGBA unchanged in channels 0:4 | Channel order or count |
| Pose/edge experts | Pose one-hot occupies hidden offsets 6:10 | Hidden encoding |
| Runtime export | No encoder is needed after an initial key state is constructed | Making encoding recurrently required |
