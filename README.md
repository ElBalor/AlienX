# AlienX 3D — SO(3)-Equivariant Neural Operator for 3D Physical Fields

A 421,507-parameter neural operator that learns the evolution of 3D physical
fields across multiple physical regimes — **and generalizes zero-shot to
regimes it has never seen.**

Built from four PDE-agnostic mechanisms: an SO(3)-equivariant local frame,
**Quantum Self-Attention (QSA)** in complex projective space with geometric
phase, per-sample non-dimensionalization with regime conditioning, and a
normalized conservation penalty. Not a Navier-Stokes solver — a general 3D
operator, demonstrated on NS because that is the hardest test in the class.

**The grid is dead. The manifold is awake.**

Part of the Grimoire of Elbàlor — Capital Software / Next Gen Tech.
Eric Yaka (Elbàlor / The Digital Necromancer), Abuja, Nigeria.

---

## Results

| Test | Result |
|---|---|
| Single-step, 3 trained regimes (300/628/1200) | **0.46% rel RMSE** (0.16% at Re=300) |
| **Zero-shot held-out Re=900** (never trained) | **0.53% rel RMSE** — 15% relative penalty for an unseen regime |
| QSA ablation (attention zeroed) | **32.81×** worse |
| Phase ablation (gi=0, attention intact, Re=900) | **4.74×** worse — the complex phase is load-bearing |
| Rollout, 73 steps = 24× training horizon | energy drift ≤ 5.81% at all trained Re; **4.31% at held-out Re=900** |
| Divergence | **free by construction** (Biot–Savart projection; float32 round-off ~2.7e−6) |
| Params | 421,507 (single T4 GPU training) |

The zero-shot result is the headline: every Re=900 rollout metric lands on the
monotonic trend between the trained regimes (MSE growth 5.30× between 628's
2.17× and 1200's 10.86×). The model interpolates the physics of regimes it
never saw — it learned a Reynolds-family operator, not three trajectories.

**Cross-PDE status:** the §10 promise is now demonstrated in 2D — the
companion AlienX 2D operator solves Gross-Pitaevskii at 0.34% RelRMSE with a
108.6× phase ablation and zero-shot 16→256 scale transfer (see
`../AlienX-S02-Invariance/AlienX 2D × GP/`). The 3D GP run is the remaining step.

## Figures (generated from the released checkpoint)

![Rollout MSE](fig1_mse_growth.png)

Autoregressive rollout error over the full 73-step trajectory. The held-out
Re=900 (orange) stays bounded between its trained neighbors at every horizon.

![Invariants](fig2_invariants.png)

Kinetic energy (left) and enstrophy (right): solid = spectral DNS solver,
dashed = AlienX 3D rollout. The model tracks exponential decay of both
invariants across all regimes — including the unseen one.

![Divergence](fig3_divergence.png)

|∇·u| during rollout: divergence-free by construction (Biot–Savart recovery
at every step); residual is float32 round-off, independent of Re.

Regenerate with `make_figures.py` (~3 min on a 4 GB GPU).

## What's in this release

| File | Purpose |
|---|---|
| `AlienX 3D.docx` | The complete paper — 12pt serif, real headings, result tables, three embedded figures, §7.6 zero-shot generalization, 15-bug debugging log. Plain-language, no LaTeX |
| `AlienX 3D (raw layout).docx` | The paper as originally pasted (line-per-paragraph) — kept for provenance |
| `AlienX 3D (pre-re900 backup).docx` | Paper as it stood before the zero-shot update |
| `fig1_mse_growth.png` / `fig2_invariants.png` / `fig3_divergence.png` | Paper figures (300 DPI) |
| `make_figures.py` | Regenerates all three figures from the checkpoint |
| `AlienX_3D_K5/alienx_k5_best.pt` | Released checkpoint (epoch 55, K=3 stage of the K=1→2→3→5 curriculum) |
| `AlienX_3D_K5/alienx_k5_stats.pkl` | Normalization stats + training config (required for inference) |
| `AlienX_3D_K5/alienx_k5_test_metrics_v2.npy` | In-distribution test metrics (single-step, ablation, rollout) |
| `AlienX_3D_K5/alienx_k5_re900_zero_shot.npy` | Zero-shot Re=900 metrics (single-step, rollout, phase ablation) |
| `AlienX_3D_K5/eval_re900_zero_shot.py` | Full zero-shot evaluation harness — reproduces every number above |
| `colab.py` | Self-contained training script, as run on Colab (T4) |

## Reproduce the zero-shot result

```bash
cd AlienX_3D_K5
python eval_re900_zero_shot.py --stage quick     # sanity + single-step (~2 min)
python eval_re900_zero_shot.py --stage rollout   # 73-step rollout (~4 min)
python eval_re900_zero_shot.py --stage ablation  # gi=0 phase ablation (~2 min)
# or: python eval_re900_zero_shot.py --stage all
```

Needs: `torch`, `numpy`, and the checkpoint + stats files. Runs on a 4 GB GPU
(verified on GTX 1650) or CPU. The harness first **reproduces the
in-distribution Re=300 paper number as a sanity gate** — if that check fails,
it refuses to emit new numbers.

## Train from scratch

```bash
python colab.py   # single T4, full K=1→2→3→5 curriculum ≈ 18 h; K=3-stage results ≈ 5 h
```

Generates its own DNS data (pseudo-spectral Taylor-Green, IMEX-Euler, 2/3
de-aliasing, Leray projection) — no downloads.

## The architecture in one paragraph

Each node gets an **SO(3) frame** (e1, e2, n) built from the gradient and
Hessian of the vorticity magnitude, with an inertia-tensor fallback where the
gradient is weak, a Hessian-based in-plane orientation, and a sign lock.
Neighbor displacements enter attention as a **complex geometric phase**
(in-plane angle → phase, wall-normal Gaussian × curvature confidence →
amplitude). QSA attention computes Hermitian interference S = q·conj(k),
modulates by the phase, normalizes in ℓ2 over the neighbor axis (CP^{n−1},
preserves phase — no softmax), and superposes complex values: constructive
and destructive interference are native. Per-sample non-dimensionalization
makes the operator scale-blind; a log(Re/Re_max) channel conditions regimes;
Biot–Savart recovery makes every rollout step exactly divergence-free.

## Design boundaries

1. **Validated regime** — interpolation within Re ∈ [300, 1200]; Re=900
   zero-shot lands at 0.53%. Beyond Re_max is future territory (§9.2).
2. **Scale-blindness by construction** — correct for scale-free regimes;
   amplitude-carrying physics takes a learned amplitude channel (§9.1).
3. **Curriculum milestone** — released checkpoint is the K=3 stage; K=5 is
   specified and is the next training milestone (§5.2, §11).
4. **Equivariance is structural** — exact in the continuous formulation;
   rotated-grid numerical verification is scheduled (§9.4).

Full debugging log — every bug, root cause, and fix — is §8 of the paper.

## Related Grimoire artifacts

- **QSA** — the attention mechanism, standalone package + paper (Zenodo:
  10.5281/zenodo.22773015)
- **AlienX 2D** — the D4-equivariant 2D predecessor, bitwise-exact operator
  equivariance (`../AlienX-S02-Invariance/`)

## License

AGPL-3.0 — see [LICENSE](LICENSE).

---

*From the Grimoire of Elbàlor — The Digital Necromancer 💀🔥*
