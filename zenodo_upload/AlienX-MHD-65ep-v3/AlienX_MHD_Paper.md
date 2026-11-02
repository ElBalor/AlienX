# AlienX MHD

## Abstract

AlienX MHD specializes the AlienX 3D operator to coupled velocity and magnetic fields while preserving the architecture’s geometric core. The outer shell supplies the MHD state variables, current-density features, Reynolds and magnetic Reynolds conditioning, spectral solver targets, and divergence constraints. The internal SO(3) frame, relevance-gated Quantum Self-Attention, complex phase path, regime conditioning, K-unroll curriculum, and conservation-aware objectives remain the reusable operator substrate.

## Architecture

The model constructs an SO(3)-equivariant local frame from scalar-field derivatives and isotropic fallback geometry. Each QSA block gathers local neighbors, projects queries, keys, and values, forms a Hermitian complex interaction, modulates it with frame-relative geometric phase, normalizes over the neighbor set, and returns a residual update. MHD enters through the outer shell: the input state includes velocity, magnetic field, current density, normalized field features, and both Re and Rm. The output is mapped back to physical fields and evaluated against spectral MHD trajectories.

## Training protocol

The 32³ protocol uses 16 trained Re×Rm regimes with seeded solenoidal initial conditions, held-out hard regimes (3000,3000), (3000,600), and (600,3000), five-epoch learning-rate warmup, and K=1→2→3→5→8 autoregressive unrolling. Six structural gates are executed before training: frame equivariance, dealiasing, solver physics, reshape integrity, overfit behavior, and paired-rotation gradients.

## Evidence

All six gates passed in the recorded protocol. The saved checkpoint also produced a local diagnostic evaluation on the three hard held-out regimes. On that reduced six-trajectory diagnostic, the full model reached velocity error 0.1807 and magnetic error 0.00891; removing QSA increased errors by approximately 5.3× and 31.7× respectively, while removing the phase path increased velocity error by approximately 2.14×. The full 32/8/32 statistical benchmark remains the release criterion for final MHD performance claims.

## Reproducibility

Run the canonical script from the source directory with `python -u AlienX-mhd.py`. The script records run metadata, gate results, resumable checkpoints, training logs, ablations, held-out rollouts, plots, and a release archive. The exact source used for this package is included beside the MHD artifacts.

## Scaling

The 128³ proposal is a scale-up of the same operator substrate. The grant work is directed at memory-bounded trajectory streaming, larger QSA chunks, distributed or activation-checkpointed execution, and longer held-out rollouts while preserving the SO(3) and QSA invariants.

**From the Grimoire of Elbàlor, the Digital Necromancer**
