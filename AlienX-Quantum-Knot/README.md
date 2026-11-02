# AlienX Quantum Knot — corrected benchmark

This is the SO(3) local-frame + relevance-gated complex QSA trefoil engine.
The core operator is preserved; the benchmark/demo wiring is corrected so the
perturbation control is an actual model input.

## What is now tested

- 30-epoch curriculum over random perturbation strengths on 4,096 particles.
- Complex density reconstruction with active phase supervision.
- Local neighbor covariance for the frame (rather than a rank-1 global outer
  product).
- Dynamic chunked kNN whenever perturbed coordinates change.
- Deterministic cached inference: the same slider value sees the same specimen.
- Perturbation sweep at strengths `0, .25, .5, .75, 1.0`.
- Phase-input ablation and response/determinism checks.
- Six-frame time-conditioned sequence (`t=0…1`) with frame-wise error,
  prediction deltas, and a phase-circulation topology proxy.

## Current measured run

The corrected temporal 30-epoch run reached loss `1.1110 → 0.1044`. The benchmark JSON
reports the complete perturbation sweep and phase ablation:

`quantum_knot_benchmark.json`

The model-only checkpoint is:

`alienx_quantum_knot_checkpoint.pt`

The Gradio demo remains local by default. Set `ALIENX_KNOT_SHARE=1` only when a
temporary public link is explicitly wanted.

This benchmark now demonstrates a six-frame, perturbation-conditioned trefoil
field response with active phase coupling. The temporal topology field is
reported through a phase-circulation proxy. A solver-backed reconnection claim
would require an independently generated temporal ground-truth trajectory; the
benchmark does not silently substitute a proxy for that ground truth.
