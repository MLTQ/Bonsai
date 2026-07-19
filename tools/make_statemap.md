# make_statemap.py

## Purpose
Builds each manifold creature's 2D state map (UMAP, PCA fallback) for the explorer panel: anchor islands + inter-anchor roads + uniform sprinkle, normalized to [0,1]².

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| `StateMapPanel.swift` | json {method, points, z, anchors}; kNN inversion uses points↔z pairing | key names, pairing |
| `--creature 2d|3d` | matches manifold_shoggoth / manifold_shoggoth3d Z_SPECs | anchor renames |
