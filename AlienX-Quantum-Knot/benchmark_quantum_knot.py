"""Post-training benchmark for the corrected AlienX Quantum Knot engine."""
from __future__ import annotations
import importlib.util, json, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ["ALIENX_KNOT_EPOCHS"] = "0"
os.environ.setdefault("ALIENX_KNOT_NO_LAUNCH", "1")
spec = importlib.util.spec_from_file_location("qknot", ROOT / "AlienX-Qknot.py")
q = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(q)

ckpt = ROOT / "alienx_quantum_knot_checkpoint.pt"
state = torch.load(ckpt, map_location=q.device, weights_only=False)
q.model.load_state_dict(state["model"]); q.model.eval()
base = q.generate_trefoil_knot(n_particles=q.KNOT_PARTICLES, grid_size=32, seed=4242)
curve = q._trefoil_curve(torch.device("cpu"))
noise = torch.randn(base[0][0].shape, generator=torch.Generator().manual_seed(4243))

@torch.no_grad()
def score(strength, phase_input=True):
    coords, density, target, phases = q.apply_fixed_perturbation(base, noise, strength, curve)
    coords, density, target, phases = (x.to(q.device) for x in (coords, density, target, phases))
    p = q.model(coords, density, perturbation=strength, time_value=1.0,
                phases=phases if phase_input else torch.zeros_like(phases))
    target_c = torch.cat([target * phases.cos(), target * phases.sin()], dim=-1)
    amp = p.square().sum(-1, keepdim=True).sqrt()
    return {
        "complex_mse": float((p-target_c).square().mean()),
        "amplitude_rmse": float((amp-target).square().mean().sqrt()),
        "phase_alignment": float(torch.cos(torch.atan2(p[...,1:2], p[...,0:1])-phases).mean()),
        "mean_amplitude": float(amp.mean()),
    }

strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
full = {str(s): score(s, True) for s in strengths}
no_phase = {str(s): score(s, False) for s in strengths}
response = max(abs(full[strengths[-1].__str__()]["mean_amplitude"] - full[strengths[0].__str__()]["mean_amplitude"]), 0.0)
def phase_circulation(coords, phases):
    idx = q.knn_chunked(coords, k=8)
    bidx = torch.arange(coords.shape[0], device=coords.device).view(-1, 1, 1).expand_as(idx)
    nbr = phases[bidx, idx]
    delta = torch.atan2(torch.sin(nbr - phases.unsqueeze(2)), torch.cos(nbr - phases.unsqueeze(2)))
    return float(delta.abs().mean())

temporal = []
prev_pred = None
for i in range(q.KNOT_TIME_STEPS):
    t = i / max(1, q.KNOT_TIME_STEPS - 1)
    coords, density, target, phases = q.apply_temporal_perturbation(base, noise, 0.8, t, curve)
    coords, density, target, phases = (x.to(q.device) for x in (coords, density, target, phases))
    pred = q.model(coords, density, perturbation=0.8, time_value=t, phases=phases)
    target_c = torch.cat([target * phases.cos(), target * phases.sin()], dim=-1)
    row = {"t": t, "complex_mse": float((pred-target_c).square().mean().detach()),
           "phase_circulation_target": phase_circulation(coords, phases)}
    if prev_pred is not None:
        row["prediction_delta_from_previous"] = float((pred-prev_pred).square().mean().sqrt())
    prev_pred = pred.detach()
    temporal.append(row)

result = {
    "checkpoint": str(ckpt),
    "architecture": "SO(3) local frame + relevance-gated complex QSA + complex density head",
    "particles": q.KNOT_PARTICLES, "grid": 32,
    "training_epochs": int(state.get("epochs", len(state.get("loss_history", [])))),
    "loss_start": state["loss_history"][0]["loss"],
    "loss_final": state["loss_history"][-1]["loss"],
    "perturbation_sweep": full,
    "phase_input_ablation": no_phase,
    "temporal_sequence": temporal,
    "temporal_frames": q.KNOT_TIME_STEPS,
    "response_delta_mean_amplitude": response,
    "deterministic_inference": True,
}
(ROOT / "quantum_knot_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
