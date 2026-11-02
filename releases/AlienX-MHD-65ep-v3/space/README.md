---
title: AlienX 3D MHD Flow Lab
emoji: 🜂
colorFrom: slate
colorTo: cyan
sdk: gradio
app_file: app.py
pinned: false
---

# AlienX 3D · MHD Flow Lab

An interactive dark Gradio readout for the AlienX ISN/QSA MHD operator. It
renders a 32³ predicted velocity field, DNS-vs-predicted mid-plane maps, and
relative rollout errors for a selected `(Re, Rm)` regime.

This Space is a live diagnostic demonstration. The measured benchmark and its
limitations are in `../evaluation/partial_checkpoint/results.json`; it is not
presented as the completed 65-epoch release until the K=8 phase is finished.
