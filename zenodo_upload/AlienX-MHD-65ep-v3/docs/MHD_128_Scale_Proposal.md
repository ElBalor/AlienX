# AlienX MHD Scaling from 32 Cubed to 128 Cubed

## Project summary

This project will establish whether the AlienX geometric neural operator can model three-dimensional magnetohydrodynamics across physical regimes and grid resolutions, then extend the validated implementation from 32³ to 128³. The proposed work treats the current 32³ MHD run as a baseline to complete and independently verify. It does not assume that the Navier–Stokes results transfer automatically to MHD or that a 128³ model already works.

## Research objective

Measure how a single SO(3)-equivariant, complex-attention operator predicts coupled velocity and magnetic-field evolution across Reynolds number \(Re\), magnetic Reynolds number \(Rm\), and spatial resolution. The target is a reproducible 128³ MHD training and evaluation workflow with documented accuracy, invariant preservation, memory use, runtime, and grid-transfer behavior.

## Current engineering baseline

The MHD adapter generates periodic pseudo-spectral MHD trajectories, applies 2/3 dealiasing and solenoidal projection, and trains on velocity and magnetic-field evolution. Its model constructs a local frame from current-density magnitude, uses complex QSA over 26 neighbors, conditions on \(Re\) and \(Rm\), and reconstructs velocity and magnetic fields through spectral Biot–Savart operators.

The local 4³ run is strictly a fast software smoke test. Its grid has only seven retained Fourier modes, its training schedule is intentionally short, and its metrics are not scientific evidence. The GTX 1650 has 4 GB VRAM; it is suitable for small smoke checks, not the proposed 128³ training campaign.

## Work plan and decision gates

### 1. Establish the 32³ MHD baseline

- Run the complete six-gate protocol at 32³ with K=1→2→3→5→8 autoregressive multi-step training, a five-epoch learning-rate warm-up, and 256 optimizer updates per epoch. Accumulate per-step gradients in truncated-BPTT mode and use activation checkpointing to bound K-unroll memory.
- Use 32 distinct seeded divergence-free training trajectories and 8 validation trajectories for each trained \((Re,Rm)\) pair in the Cartesian product \(\{300,600,1200,2400\}^2\).
- Keep the difficult extrapolation and cross-coupling pairs \((3000,3000)\), \((3000,600)\), and \((600,3000)\) out of training; generate 32 independent held-out trajectories per pair. Validate complete active-K autoregressive rollouts and select the release checkpoint only in the K=8 phase.
- Repeat the frozen protocol with at least three global seeds. Record relative error for \(u\) and \(B\), energy and enstrophy errors, divergence residuals, QSA/phase ablations, multi-step rollout error and spread across held-out trajectories, wall time, and peak memory.
- Freeze the configuration before publishing the checkpoint and include the exact source, run metadata, logs, gates, and evaluation results.

**Gate:** the full-resolution run completes without OOM or nonfinite values; training, validation, and held-out samples are disjoint by initial-condition seed and physical regime; the hard held-out multi-step evaluation is reproducible from the saved checkpoint.

### 2. Validate resolution transfer at 64³

- Evaluate the frozen 32³ checkpoint at 64³ without retraining where the discretization permits.
- Compare against an independently generated 64³ MHD solver trajectory, including divergence, energy, magnetic energy, and spectral error.
- Profile frame construction, QSA neighbor gathering, FFTs, data generation, and checkpoint I/O.

**Gate:** publish measured transfer error and resource use. If the solver or operator requires a resolution-specific correction, document and test it before moving to 128³.

### 3. Prepare the 128³ data and model path

- Replace the current all-trajectories-in-RAM dataset builder with deterministic disk-backed or on-demand trajectory generation.
- Stream batches and avoid keeping duplicate full-resolution copies of \(u\), \(B\), current, vorticity, and intermediate features.
- Extend QSA chunking to source-side key/value projections so the full \(N\times H\times D\) tensors do not become the memory ceiling at \(N=128^3\).
- Keep FFT and geometry operations in validated precision; compare chunked results against 32³ and 64³ reference paths.
- Add restartable checkpoints, run manifests, and peak-memory telemetry before the long run.

**Gate:** a single 128³ forward/backward optimizer step and a short multi-step training run complete on the target accelerator, match the unchunked reference on a smaller grid, and preserve the solver invariants.

### 4. Train and evaluate at 128³

- Use the frozen data split and training protocol established at 32³, then scale the grid while keeping training, validation, and held-out initial conditions disjoint.
- Evaluate both same-resolution performance and transfer among 32³, 64³, and 128³.
- Report the complete error and resource profile, plus independent-seed variation.

**Gate:** release only after the 128³ checkpoint, configuration, raw logs, gate results, and evaluation script reproduce the reported metrics.

## Compute and memory rationale

The current generator stores all trajectories in host memory. Under the expanded 32³ protocol (16 trained Re/Rm pairs, 32 training and 8 validation trajectories per pair, plus 32 trajectories for each of 3 held-out regimes), the 12-snapshot float32 tensors require about **6.5 GiB** at 32³ before Python/runtime overhead. Scaling the same in-memory dataset directly to 128³ multiplies storage by 64: about **288 GiB** for training, **72 GiB** for validation, and **54 GiB** for held-out trajectories, or **414 GiB** total before solver buffers and model activations. The 128³ data path must therefore be disk-backed/on-demand or streamed; the current all-in-memory builder cannot be used there.

The QSA path is chunked over query nodes, but some full-grid hidden, key, and value tensors remain resident. At 128³, the node count is 64 times that of 32³. The proposal therefore funds both accelerator memory and engineering for source-side chunking, streamed data, and checkpoint/restart. Hardware requirements should be finalized from measured 32³/64³ profiles; the GTX 1650 is a smoke-test device, not a training target.

## Deliverables

1. Reproducible 32³ MHD baseline with multi-seed held-out results.
2. 64³ grid-transfer benchmark and profiling report.
3. Disk-backed data pipeline and memory-bounded QSA implementation for 128³.
4. 128³ checkpoint, evaluation harness, logs, numerical gates, and Hugging Face model card.
5. A technical report documenting results, limits, compute consumption, and the exact reproduction steps.

## Risks and controls

- **Synthetic solver bias:** verify numerical convergence and compare against independently implemented solver settings before claiming physical generalization.
- **Single-family initial conditions:** expand the low-frequency perturbation family and report results across independent seeds and initial-condition families.
- **Resolution transfer failure:** treat 32→64→128 as measured milestones; do not claim grid independence from architectural intent alone.
- **Memory/runtime failure:** benchmark progressively at 32³, 64³, and 128³; use disk-backed data, query/source chunking, and resumable checkpoints.
- **Overstated generality:** MHD evidence supports only the MHD configuration tested. Each new PDE requires its own solver shell, invariants, training data, and validation gates.
