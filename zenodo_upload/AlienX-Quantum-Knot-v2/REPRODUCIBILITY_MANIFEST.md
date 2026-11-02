# AlienX Quantum Knot v2 — Reproducibility manifest

This release contains the reproducible DOCX paper, canonical source, checkpoint, temporal benchmark, and a Space bundle.

## Canonical entry points

- Paper: `AlienX_Quantum_Knot_Paper.docx`
- Source: `AlienX-Qknot.py`
- Benchmark: `benchmark_quantum_knot.py`
- Results: `quantum_knot_benchmark.json`
- Space bundle: `space_app.py` and `space_readme.md`

## Re-run

Run `python -u AlienX-Qknot.py` for the 30-epoch training run, or set `ALIENX_KNOT_NO_LAUNCH=1` for a non-interactive run. Then run `python -u benchmark_quantum_knot.py` to regenerate the perturbation sweep, phase ablation, deterministic multi-frame sequence, and topology-sensitive phase proxy.

## Architecture boundary

The SO(3) local frame, relevance-gated complex QSA, geometric phase, and coherent neighbor superposition are the reusable AlienX substrate. The knot outer shell supplies trefoil coordinates, density/phase channels, perturbation and time conditioning, complex-density targets, and vortex diagnostics.

## Scope

The included evidence is a reproducible perturbation-conditioned temporal field response with active phase coupling. A solver-backed reconnection claim would require an independent temporal ground-truth trajectory; this release keeps that distinction explicit.
