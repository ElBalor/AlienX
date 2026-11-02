"""Dark Gradio demo for the AlienX 3D MHD operator.

The app uses the same ISN/QSA implementation and a model-only copy of the
latest resumable checkpoint. It is a live diagnostic/demo, not a replacement
for the full 32/8/32 Kaggle benchmark.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source" / "AlienX-mhd.py"
if not SOURCE.exists():
    SOURCE = HERE.parent / "source" / "AlienX-mhd.py"
if not SOURCE.exists():
    SOURCE = HERE.parent.parent.parent / "AlienX-mhd.py"
spec = importlib.util.spec_from_file_location("alienx_mhd_space", SOURCE)
ax = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ax)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = ax.AlienX_MHD(hidden=ax.HIDDEN, blocks=ax.LAYERS, heads=ax.HEADS,
                      chunk=ax.CHUNK, re_max=max(ax.RE_VALUES),
                      rm_max=max(ax.RM_VALUES), sigma_z=ax.SIGMA_Z).to(DEVICE)
MODEL.load_state_dict(torch.load(HERE / "alienx_mhd_partial.pt", map_location=DEVICE, weights_only=True))
MODEL.eval()
for p in MODEL.parameters():
    p.requires_grad_(False)

def _step(model, b, u, re, rm):
    su = u.square().mean().sqrt().clamp_min(1e-8)
    sb = b.square().mean().sqrt().clamp_min(1e-8)
    uc = (u / su).permute(1, 2, 3, 0).reshape(1, ax.GRID ** 3, 3)
    bc = (b / sb).permute(1, 2, 3, 0).reshape(1, ax.GRID ** 3, 3)
    rt = torch.tensor([re], device=DEVICE)
    mt = torch.tensor([rm], device=DEVICE)
    bp, up, _, _ = ax.model_step(MODEL, bc, uc, rt, mt)
    up = up.reshape(1, ax.GRID, ax.GRID, ax.GRID, 3).permute(0, 4, 1, 2, 3)[0] * su
    bp = bp.reshape(1, ax.GRID, ax.GRID, ax.GRID, 3).permute(0, 4, 1, 2, 3)[0] * sb
    return up, bp

@torch.no_grad()
def run_case(re, rm, steps):
    steps = int(steps)
    u, b = ax.generate_trajectory(float(re), float(rm), steps + 1, ax.DT,
                                  seed=991, noise=ax.NOISE)
    u = u.to(DEVICE); b = b.to(DEVICE)
    pred_u, pred_b = u[0], b[0]
    rows = []
    for i in range(steps):
        pred_u, pred_b = _step(MODEL, pred_b, pred_u, float(re), float(rm))
        truth_u, truth_b = u[i + 1], b[i + 1]
        eu = float((pred_u - truth_u).square().mean().sqrt() /
                   (truth_u.square().mean().sqrt() + 1e-8))
        eb = float((pred_b - truth_b).square().mean().sqrt() /
                   (truth_b.square().mean().sqrt() + 1e-8))
        rows.append((eu, eb))
    mid = ax.GRID // 2
    truth_mag = truth_u.norm(dim=0)[:, :, mid].detach().cpu().numpy().T
    pred_mag = pred_u.norm(dim=0)[:, :, mid].detach().cpu().numpy().T
    # Keep the 3-D view light enough for a laptop browser.
    stride = max(1, ax.GRID // 8)
    zz, yy, xx = torch.meshgrid(torch.arange(0, ax.GRID, stride),
                                torch.arange(0, ax.GRID, stride),
                                torch.arange(0, ax.GRID, stride), indexing="ij")
    xyz = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).to(DEVICE)
    vec = pred_u[:, ::stride, ::stride, ::stride].permute(1, 2, 3, 0).reshape(-1, 3)
    nrm = vec.norm(dim=1).detach().cpu().numpy()
    xyz = xyz.detach().cpu().numpy()
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}, {"type": "heatmap"}, {"type": "heatmap"}]],
                        subplot_titles=("predicted velocity field", "DNS |u| mid-plane", "AlienX |u| mid-plane"))
    fig.add_trace(go.Cone(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                          u=vec[:, 0].detach().cpu(), v=vec[:, 1].detach().cpu(), w=vec[:, 2].detach().cpu(),
                          colorscale="Viridis", sizemode="absolute", sizeref=2.0,
                          showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=truth_mag, colorscale="Magma", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=pred_mag, colorscale="Magma", showscale=False), row=1, col=3)
    fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="#070b12",
                      plot_bgcolor="#070b12", margin=dict(l=8, r=8, t=55, b=8),
                      font=dict(color="#d9fff7"))
    meta = json.dumps({
        "checkpoint": "latest resumable checkpoint (next_epoch=27/65)",
        "device": str(DEVICE), "grid": f"{ax.GRID}^3", "Re": re, "Rm": rm,
        "steps": steps, "final_relative_u_rmse": rows[-1][0],
        "final_relative_B_rmse": rows[-1][1],
        "note": "Live demo; complete 32/8/32 benchmark is reported separately.",
    }, indent=2)
    readout = (f"### LIVE READOUT\n**Re={re:.0f}, Rm={rm:.0f}, {steps}-step rollout**\n\n"
               f"- velocity relative RMSE: **{rows[-1][0]*100:.2f}%**\n"
               f"- magnetic relative RMSE: **{rows[-1][1]*100:.2f}%**\n"
               f"- device: **{DEVICE}**\n\n```json\n{meta}\n```")
    return fig, readout

CSS = """
body,.gradio-container{background:#070b12!important;color:#d9fff7!important}
.panel{border:1px solid #1b5d62!important;background:#0b121b!important;border-radius:14px}
h1{font-family:ui-monospace,monospace;letter-spacing:.08em;color:#71f5db}
button.primary{background:#123c4a!important;border-color:#71f5db!important}
"""

with gr.Blocks(css=CSS, title="AlienX 3D · MHD Flow Lab") as demo:
    gr.Markdown("# ALIENX 3D · MHD FLOW LAB\n### ISN / QSA operator · live 32³ field readout")
    gr.Markdown("Run a small, interactive MHD rollout on the same architecture. The 3-D cone field and mid-plane maps are generated locally; the benchmark evidence remains in the evaluation bundle.")
    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            re = gr.Slider(300, 3000, value=1200, step=100, label="Reynolds number")
            rm = gr.Slider(300, 3000, value=1200, step=100, label="Magnetic Reynolds number")
            steps = gr.Slider(1, 8, value=4, step=1, label="Autoregressive steps")
            run = gr.Button("⚡ RUN MHD ROLLOUT", variant="primary")
        with gr.Column(scale=3, elem_classes="panel"):
            plot = gr.Plot(label="3-D field / slices")
            readout = gr.Markdown()
    run.click(run_case, [re, rm, steps], [plot, readout], concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), show_error=True)
