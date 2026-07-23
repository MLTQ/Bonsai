# test_transport_nca3d_contract.py

## Purpose
Provides CPU-fast gates for the transport experiment's global controller, axis convention, donor transplant, and checkpoint format.

## Components

### `TransportNCA3DContracts`
- **Does**: Verify a 240-step closed ring, directed edge order, positive-X advection, zero-flow equivalence to walking NC3C, and exact TN3D1 round trips

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment launch | `python3 training/test_transport_nca3d_contract.py` exits zero | Controller, transport, transplant, or checkpoint semantics |
