# AlienX 3D MHD — 65-epoch release package

This package contains the AlienX 3D MHD source, grant documentation, training
artifacts, diagnostics, and evaluation location for the 65-epoch protocol.

## Current status

`training/latest_resume/run/` is the latest Kaggle checkpoint package. It is a
resume checkpoint, not the final release: it contains `last_training.pt`, the
training log, run metadata, and all six passed gate results. The run stopped
before the K=8 phase produced `best_alienx_mhd.pt`.

`evaluation/partial_checkpoint/` contains a reproducible GTX-1650 diagnostic
evaluation of that saved state. It uses 1 trajectory per trained regime and 2
per hard held-out regime because the full 736-trajectory protocol requires
about 6.47 GiB of host tensors before runtime overhead. It is useful evidence
that the implementation and ablations execute, but it is not a substitute for
the full grant benchmark.

## Layout

- `source/` — exact `AlienX-mhd.py` used for the protocol.
- `docs/` — README, license, and 128^3 scaling proposal.
- `training/latest_resume/run/` — latest 65-epoch run state and logs.
- `evaluation/partial_checkpoint/` — local smoke/ablation/rollout results for
  the saved partial checkpoint.
- `evaluation/pending/` — reserved for final 32/8/32 benchmark outputs after
  the K=8 training phase.
- `space/` — dark Gradio MHD Flow Lab with interactive 3-D cones and slices.
- `diagnostics/` — prior smoke runs and root-level diagnostic artifacts.

The canonical working source remains at the S03 project root. This package is
the organized release/grant bundle and does not replace that source.
