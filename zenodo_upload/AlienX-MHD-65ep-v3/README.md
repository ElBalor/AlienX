---
language: en
tags:
- geometric-neural-operator
- so3-equivariance
- magnetohydrodynamics
- scientific-machine-learning
license: mit
---

# AlienX 3D MHD — 65-epoch release

## What this release is

AlienX MHD specializes the AlienX 3D operator to coupled velocity and magnetic fields. It is a physics-facing operator experiment: the reusable geometric substrate remains intact while the MHD outer shell supplies physical state variables, regime parameters, spectral targets, divergence constraints, and MHD diagnostics.

## Architecture

The shared substrate is built from an SO(3)-equivariant local frame, relevance-gated complex QSA, frame-relative phase modulation, regime conditioning, residual updates, and conservation-aware objectives. The MHD shell supplies `u` and `b` fields, Reynolds and magnetic-Reynolds conditioning, spectral solver targets, and divergence-free diagnostics. This is why the AlienX architecture transfers across physical systems without replacing the core operator.

## Protocol

- 32³ protocol
- six structural gates before training
- seeded multi-regime data generation
- five-epoch learning-rate warmup
- K=1→2→3→5→8 autoregressive unroll curriculum
- held-out hard regimes `(3000,3000)`, `(3000,600)`, and `(600,3000)`
- paired-rotation gradient check

The package preserves the exact source, checkpoints, logs, gate results, rollout plots, and the partial diagnostic evaluation. The latest resume directory records the run state available from the Kaggle execution.

## Reproduce

```bash
python -u source/AlienX-mhd.py
```

The source writes run metadata, gate results, checkpoints, training logs, held-out rollouts, and plots to the configured output directory. The paper documents the protocol and the distinction between the recorded partial diagnostic and the full release benchmark.

## Files

- `source/AlienX-mhd.py` — canonical MHD source
- `training/latest_resume/run/` — latest checkpoint/log package
- `evaluation/partial_checkpoint/` — reproducible local diagnostic
- `diagnostics/` — gate, rollout, invariant, and ablation artifacts
- `AlienX_MHD_Paper.docx` — full reproducibility paper
- `space/` — interactive local MHD demo bundle

## Scaling

The proposed 128³ work scales the same operator substrate with memory-bounded trajectory streaming, larger QSA chunks, and activation checkpointing. It is a compute expansion, not a different architecture.

From the Grimoire of Elbàlor, the Digital Necromancer.
