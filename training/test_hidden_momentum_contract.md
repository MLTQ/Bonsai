# test_hidden_momentum_contract.py

## Purpose
Provides CPU-fast gates for hidden-only integration and the NCA5 byte contract.

## Components

### `HiddenMomentumContractTests`
- **Does**: Verifies residual RGBA, decay-before-hidden-integration, header fields, payload size, and exact export/load round trips

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment launch | `python3 training/test_hidden_momentum_contract.py` exits zero | NCA5 math/layout changes |
