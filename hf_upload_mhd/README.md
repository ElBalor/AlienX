---
license: agpl-3.0
library_name: pytorch
tags:
  - neural-operator
  - magnetohydrodynamics
  - mhd
  - so3-equivariance
  - quantum-self-attention
  - physics-informed
  - scientific-machine-learning
---

# AlienX 3D for Magnetohydrodynamics

This repository contains the MHD specialization of AlienX 3D, its trained checkpoint, and the run metadata needed to interpret the reported results. The MHD specialization is a cross-PDE experiment built from the AlienX geometric operator; it is distinct from the separately evaluated Navier–Stokes checkpoint.

## Model

The model consumes velocity and magnetic fields on a periodic three-dimensional grid. It forms the current density \(J=\nabla\times B\), builds local geometric features from \(|J|\), and conditions on Reynolds and magnetic Reynolds numbers. Four Quantum Self-Attention blocks operate over a 26-neighbor isotropic stencil. The output predicts current and vorticity; spectral Biot–Savart reconstruction recovers magnetic and velocity fields while enforcing their divergence constraints.

The high-sample 32³ protocol trains on the full Cartesian product Re,Rm ∈ {300, 600, 1200, 2400}, with 32 independent seeded trajectories per parameter pair and 8 validation trajectories per pair. Three difficult zero-shot regimes are kept separate: (3000,3000), (3000,600), and (600,3000), with 32 independent trajectories each. The tests include joint extrapolation beyond the trained range and high-Re/low-Rm versus low-Re/high-Rm imbalance. Initial fields use seeded low-frequency solenoidal perturbations.

Training uses a five-epoch learning-rate warm-up followed by cosine decay and a K=1→2→3→5→8 autoregressive curriculum (10/10/15/15/15 epochs). Each update uses starting states with enough future snapshots to complete the full active K. The default truncated-BPTT mode accumulates loss gradients at every unrolled step and detaches only the predicted state passed to the next step, bounding activation memory; the selected mode is recorded in run metadata. QSA block activation checkpointing and VRAM-adaptive frame/QSA chunking further bound CUDA memory. Validation unrolls the active K for four independent trajectories per trained regime; the release checkpoint is selected only during K=8. Final zero-shot rollout metrics aggregate all held-out trajectories across the full 11-step horizon and report per-step mean, standard deviation, and worst-case errors. The model architecture is unchanged by this protocol update.

## Results and provenance

Read `run_metadata.json` first. It identifies the grid, seed, parameter-pair split, initial-condition perturbation, training schedule, and software/runtime. `results.json` contains validation, ablation, and aggregated rollout metrics; `gate_results.json` records the numerical and implementation gates; `train_log.json` contains the training history. A result from a smoke or custom run is not a 32³ protocol result. These settings define the protocol; they are not claims that a run has completed or passed its final evaluation.

The 4³ smoke run is for plumbing only and must not be used as evidence of MHD accuracy or generalization. No 128³ performance claim is made here. Grid transfer and higher-resolution training require their own measured evaluations.

## Reproduce

Install PyTorch with CUDA support, NumPy, SciPy, and Matplotlib. Then run the six-gate training script:

```bash
python -u AlienX-mhd.py
```

By default the script runs at 32³, saves the best validation checkpoint and resumable epoch checkpoint under `alienx_mhd_gates_hard_v1`, and writes a metadata file before running gates. On Colab, the default mirror is `AlienX_MHD/hard_regimes_v1`. To resume after an interruption, set `ALIENX_RESUME=1` with the same run settings and output directory.

To publish a completed full 32³ run from the training environment, provide a Hugging Face write token as `HF_TOKEN`, set `ALIENX_HF_UPLOAD=1`, and optionally set `ALIENX_HF_REPO`. The script refuses to publish smoke/custom runs.

## License

Source code is released under AGPL-3.0; see `LICENSE`. The checkpoint and generated solver data are distributed with this repository under the same declared repository license.

## Citation

```bibtex
@software{yaka2026alienx3dmhd,
  author = {Yaka, Eric Heylel Danjuma},
  title = {AlienX 3D for Magnetohydrodynamics},
  year = {2026}
}
```
