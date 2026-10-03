---
title: AlienX 3D Flow Lab
emoji: 💀
colorFrom: gray
colorTo: purple
sdk: gradio
app_file: app.py
pinned: true
license: agpl-3.0
short_description: 421K-param neural operator vs a Navier-Stokes solver — zero-shot Re
tags:
  - neural-operator
  - fluid-dynamics
  - so3-equivariant
  - complex-attention
  - qsa
---

# AlienX 3D · Flow Lab

A **421,507-parameter** SO(3)-equivariant neural operator races a spectral
Navier–Stokes solver on the Taylor–Green vortex. Trained on Reynolds numbers
{300, 628, 1200} — drag the slider anywhere else in [300, 1200] and it
interpolates the physics zero-shot. Flip **☠ Sever the Phase** to delete QSA's
complex channel live and watch accuracy collapse.

- **Model & paper:** [ElBalor/AlienX-3D-ISN-Operator](https://huggingface.co/ElBalor/AlienX-3D-ISN-Operator)
- Single-step: **0.46%** rel RMSE in-distribution · **0.53%** zero-shot (Re=900)
- Phase ablation: **4.74×** worse without the complex channel — try it yourself
- Divergence-free **by construction** (Biot–Savart recovery), ~2.6e-6 throughout

*The grid is dead. The manifold is awake.* — from the Grimoire of Elbàlor
