# test_momentum_contract.py

## Purpose
Provides CPU-fast regression tests for the NCA4 integrator and byte layout before longer autonomous training runs.

## Components

### `MomentumContractTests`
- **Does**: Verifies decay-before-integration with zero force and checks every NCA4 header field plus payload size
- **Interacts with**: `MomentumNCA`, `lift_state`, and `export_nca4` from `momentum_nca.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Developers / CI-by-hand | `python3 training/test_momentum_contract.py` exits 0 | Update math or NCA4 layout changes |
