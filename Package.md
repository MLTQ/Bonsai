# Package.swift

## Purpose

Defines the macOS 13+ Bonsai executable target and excludes source companion
documentation from Swift compilation/resource discovery.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| SwiftPM | `Sources/Bonsai` executable target | Target/path/platform changes |
| Companion docs | Every source `.md` is excluded explicitly | Adding a source companion without updating this list |
