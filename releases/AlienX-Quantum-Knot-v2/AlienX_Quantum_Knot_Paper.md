# AlienX Quantum Knot

## Abstract

AlienX Quantum Knot applies the same geometric operator substrate to a trefoil vortex field. The outer shell defines the trefoil geometry, density field, perturbation trajectory, phase field, and complex-density targets. The internal SO(3) local frame and relevance-gated complex QSA remain unchanged in principle; only the physical state representation and target construction are specialized for the knot experiment.

## Architecture

The engine builds a local frame from the sampled trefoil coordinates and a local neighbor covariance. QSA computes frame-relative displacements, an out-of-plane relevance gate, geometric angular phase, complex Hermitian interactions, and coherent neighbor superposition. The input projection receives coordinates, density, perturbation strength, normalized time, and phase sine/cosine channels. The output is a complex density field, which makes phase supervision active rather than an inert scalar regularizer.

## Temporal benchmark

The corrected benchmark trains for 30 epochs on 4096 particles at a 32³ field resolution. Perturbation strength is sampled during training. Inference uses one cached specimen and one fixed noise basis, so changing the control changes a deterministic physical trajectory rather than resampling unrelated particles. The benchmark evaluates six frames from t=0 to t=1, records frame-wise complex reconstruction error, prediction deltas between frames, and a phase-circulation topology proxy.

## Measured evidence

The corrected temporal run reached loss 1.1110→0.1044. Six-frame complex reconstruction errors were approximately 0.0418 at the initial frame and 0.0504 at the final frame. Phase-input ablation reduced phase alignment from positive values to negative values. The perturbation sweep, phase ablation, temporal sequence, and determinism result are stored in quantum_knot_benchmark.json.

## Reproducibility

Run `python -u AlienX-Qknot.py` for the 30-epoch run. Set ALIENX_KNOT_NO_LAUNCH=1 for a non-interactive benchmark run. Run `python -u benchmark_quantum_knot.py` to regenerate the perturbation sweep, phase ablation, temporal sequence, and topology proxy. The checkpoint, source, benchmark script, and results JSON are included in this package.

## Scope

This result demonstrates perturbation-conditioned temporal field response with active geometric phase coupling. The phase-circulation quantity is reported as a topology-sensitive proxy. A solver-backed reconnection claim requires independently generated temporal ground-truth trajectories; the package keeps that distinction explicit so the result is reproducible and defensible.

**From the Grimoire of Elbàlor, the Digital Necromancer**
