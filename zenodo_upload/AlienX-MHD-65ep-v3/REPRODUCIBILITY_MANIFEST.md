# AlienX MHD v3 — Reproducibility manifest

This release contains the reproducible DOCX paper, canonical source, saved training/evaluation artifacts, diagnostics, and a Space bundle.

## Canonical entry points

- Paper: `AlienX_MHD_Paper.docx`
- Source: `source/AlienX-mhd.py`
- Checkpoint/evaluation: `evaluation/` and `training/`
- Space bundle: `space/`

## Re-run

From `source/`, run `python -u AlienX-mhd.py`. The script records metadata, gate results, checkpoints, logs, held-out rollouts, and plots in the configured output directory. The recorded 32^3 protocol uses the six structural gates, seeded regimes, five-epoch warmup, and K=1→2→3→5→8 unrolling schedule.

## Architecture boundary

The SO(3) local frame, relevance-gated complex QSA, phase modulation, regime conditioning, and conservation-aware objective are the reusable AlienX substrate. MHD is the outer shell: it supplies velocity/magnetic state variables, Re/Rm conditioning, spectral MHD targets, divergence constraints, and MHD diagnostics.

## Scope

The package reports the included run and its local diagnostics exactly. Scaling to 128^3 is a proposed compute expansion of the same operator substrate, not a claim that a 128^3 run is already contained here.
