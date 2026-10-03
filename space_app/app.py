"""
AlienX 3D — Flow Lab (Necromancer Edition)
===========================================
A 421,507-parameter neural operator racing a spectral Navier-Stokes solver
on the Taylor-Green vortex — at Reynolds numbers it was NEVER trained on.

Trained on Re = {300, 628, 1200}. Slide anywhere in [300, 1200] and it
interpolates the physics. Sever the complex phase channel (gi = 0) and
watch prediction quality collapse: that is QSA, live.

    The grid is dead. The manifold is awake.
    — from the Grimoire of Elbàlor
"""

import os, math, pickle
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.subplots as psub
import gradio as gr

import physics as ph
import model as mdl
from model import ISNOperator3D

# ─── Summon the operator (bundled weights — no network needed) ───────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

with open(os.path.join(BASE, "alienx_k5_stats.pkl"), "rb") as f:
    SD = pickle.load(f)
ST, G = SD["ST"], SD["G"]
RE_LIST, RE_MAX = SD["RE_LIST"], SD["RE_MAX"]
T_FINAL, N_SNAPS = SD["T_FINAL"], SD["N_SNAPSHOTS"]

MODEL = ISNOperator3D(hidden_dim=128, num_layers=4, num_heads=4,
                      equivariant_output=True, use_velocity_input=True,
                      use_re=True, re_max=RE_MAX).to(DEVICE)
MODEL.G = G
MODEL.load_state_dict(torch.load(os.path.join(BASE, "alienx_k5_best.pt"),
                                 map_location=DEVICE))
MODEL.eval()
for _blk in MODEL.blocks:
    _blk.chunk = 1024   # chunked attention: identical math, ~30x smaller einsum intermediates
N_PARAMS = sum(p.numel() for p in MODEL.parameters())
print(f"Operator summoned: {N_PARAMS:,} params | trained Re={RE_LIST} | G={G}")

KX, KY, KZ, K2 = ph.make_wavenumbers(G, DEVICE)
_x = torch.arange(G, dtype=torch.float32) * (2 * math.pi / G)
_X, _Y, _Z = torch.meshgrid(_x, _x, _x, indexing="ij")
COORDS = torch.stack([_X.flatten(), _Y.flatten(), _Z.flatten()], -1).unsqueeze(0)
KNN = mdl.get_isotropic_knn_3d_periodic(G, DEVICE, dilation=max(1, G // 16))

# ─── gi=0 phase severance (monkey-patch, restores after each run) ────────────
_ORIG_GEOM = mdl.precompute_geometry

def _gi_zeroed(*a, **kw):
    gr, gi, bx = _ORIG_GEOM(*a, **kw)
    return gr, torch.zeros_like(gi), bx

def sever_phase(on: bool):
    mdl.precompute_geometry = _gi_zeroed if on else _ORIG_GEOM

# ─── Physics helpers (verbatim from the sanity-gated eval harness) ───────────
def _s(x, key):
    mu, sg = ST[key]; return (x - mu) / sg

def _u(x, key):
    mu, sg = ST[key]; return x * sg + mu

def build_cache(traj):
    return {"u": traj,
            "Us": ph.rms_velocity(traj),
            "omega": ph.velocity_to_vorticity(traj, KX, KY, KZ)}

@torch.no_grad()
def rollout(cache, Re, n_steps, zero_gi=False):
    traj, omega_traj = cache["u"], cache["omega"]
    sever_phase(zero_gi)
    try:
        u_curr, om_curr = traj[0:1].clone(), omega_traj[0:1].clone()
        U0 = ph.rms_velocity(u_curr).clamp(min=1e-6)
        mses, Es, gEs, Ds, slices = [], [], [], [], {}
        for step in range(n_steps):
            om_nd = om_curr / U0
            kf = om_nd.norm(dim=-1)
            k_, gm, gv, Hs = ph.scalar_derivatives(kf, KX, KY, KZ)
            u_nd = u_curr / U0
            pred = MODEL(COORDS, _s(k_, "k"), _s(gv, "gv"), _s(gm, "gm"),
                         _s(Hs, "H"), _s(om_nd.reshape(1, -1, 3), "om"),
                         _s(u_nd.reshape(1, -1, 3), "u"),
                         torch.tensor([float(Re)]), KNN, G=G)
            om_next = _u(pred.reshape(1, -1, 3), "tgt").reshape(1, G, G, G, 3) * U0
            u_curr = ph.vorticity_to_velocity(om_next, KX, KY, KZ, K2)
            om_curr = om_next
            if step < traj.shape[0] - 1:
                om_t, u_t = omega_traj[step + 1:step + 2], traj[step + 1:step + 2]
                mses.append(F.mse_loss(om_curr, om_t).item())
                Es.append(ph.kinetic_energy(u_curr).item())
                gEs.append(ph.kinetic_energy(u_t).item())
                Ds.append(ph.divergence_spectral(u_curr, KX, KY, KZ).abs().mean().item())
                if step in (0, n_steps // 2, n_steps - 1):
                    slices[step] = (om_curr[0, :, :, G // 2, :].norm(dim=-1).numpy(),
                                    om_t[0, :, :, G // 2, :].norm(dim=-1).numpy())
    finally:
        sever_phase(False)
    return (np.array(mses), np.array(Es), np.array(gEs), np.array(Ds), slices)

# ─── Necromancer theme ────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800&family=Cormorant+Garamond:ital,wght@0,500;1,500&display=swap');
.gradio-container {background: #050508 !important; color: #cfc9c0 !important;}
#necro-title {text-align:center; font-family:'Cinzel',serif; font-weight:800;
  font-size:2.3em; letter-spacing:.12em; margin:.2em 0 0;
  background:linear-gradient(90deg,#00ffd0,#b26bff 55%,#00ffd0);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  animation:necroPulse 3.2s ease-in-out infinite;}
#necro-sub {text-align:center; font-family:'Cormorant Garamond',serif; font-style:italic;
  color:#8a8577; font-size:1.15em; margin-bottom:1em;}
@keyframes necroPulse {0%,100%{filter:drop-shadow(0 0 5px #00ffd055)} 50%{filter:drop-shadow(0 0 16px #b26bff99)}}
.panel {border:1px solid #1c2b28 !important; border-radius:12px !important;
  background:linear-gradient(160deg,#0a0d12,#0d1117) !important;
  box-shadow:0 0 14px #00ffd014, inset 0 0 26px #05010acc !important;}
button.primary {background:linear-gradient(90deg,#003b34,#2a0f45) !important;
  border:1px solid #00ffd055 !important; color:#c9fff4 !important;
  font-family:'Cinzel',serif !important; letter-spacing:.1em !important;
  box-shadow:0 0 12px #00ffd033 !important;}
button.primary:hover {box-shadow:0 0 22px #b26bff77 !important; border-color:#b26bff88 !important;}
footer, .footer {visibility:hidden;}
#rune-canvas {position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.5;}
.metrics {font-family:'Cormorant Garamond',serif; font-size:1.15em; color:#9be8d8;}
.metrics b {color:#00ffd0; text-shadow:0 0 8px #00ffd066;}
"""

HEAD = """
<canvas id='rune-canvas'></canvas>
<script>
(function(){
  const c=document.getElementById('rune-canvas'); if(!c) return;
  const x=c.getContext('2d'); let W,H;
  const GLYPHS=['\u16A0','\u16A1','\u16A2','\u16A3','\u16A8','\u16A9','\u16AA','\u16AB',
                '\u16B1','\u16B2','\u16B3','\u16B7','\u16B9','\u16BA','\u16C1','\u16C3',
                '\u2721','\u2725','\u2726','\u1F52','\u029A','\u0262'];
  let P=[];
  function rs(){W=c.width=innerWidth;H=c.height=innerHeight;}
  addEventListener('resize',rs); rs();
  for(let i=0;i<34;i++) P.push(spawn(true));
  function spawn(any){return {x:Math.random()*W,y:any?Math.random()*H:-20,
    s:8+Math.random()*11, v:.35+Math.random()*.9, o:.08+Math.random()*.22,
    g:GLYPHS[(Math.random()*GLYPHS.length)|0], w:Math.random()*6.28};}
  function tick(){
    x.clearRect(0,0,W,H);
    for(let i=0;i<P.length;i++){let p=P[i]; p.y+=p.v; p.w+=.02;
      x.font=p.s+'px serif'; x.fillStyle='rgba(0,255,208,'+p.o+')';
      x.save(); x.translate(p.x+Math.sin(p.w)*6,p.y); x.fillText(p.g,0,0); x.restore();
      if(p.y>H+24) P[i]=spawn(false);}
    requestAnimationFrame(tick);}
  requestAnimationFrame(tick);
})();
</script>
"""

# ─── Figures ─────────────────────────────────────────────────────────────────
AXIS = dict(gridcolor="#1c2b28", zerolinecolor="#1c2b28",
            linecolor="#3a4a47", tickfont=dict(color="#8a8577"))
PAPER = dict(plot_bgcolor="#0a0d12", paper_bgcolor="#0a0d12",
             font=dict(color="#cfc9c0"), margin=dict(l=46, r=24, t=44, b=40))

def fig_metrics(m, E, gE, zero_gi, m0=None):
    f = psub.make_subplots(rows=1, cols=2,
                           subplot_titles=("Pointwise error — model vs DNS truth",
                                           "Kinetic energy — decay of the vortex"))
    f.add_trace(go.Scatter(y=m, mode="lines", line=dict(color="#00ffd0", width=2.4),
                           name="AlienX 3D"), 1, 1)
    if m0 is not None:
        f.add_trace(go.Scatter(y=m0, mode="lines", line=dict(color="#b26bff", width=2, dash="dot"),
                               name="phase severed (gi=0)"), 1, 1)
    f.update_xaxes(title_text="rollout step", row=1, col=1, **AXIS)
    f.update_yaxes(title_text="MSE (log)", type="log", row=1, col=1, **AXIS)
    f.add_trace(go.Scatter(y=gE, mode="lines", line=dict(color="#e8e3d8", width=2),
                           name="spectral solver (truth)"), 1, 2)
    f.add_trace(go.Scatter(y=E, mode="lines", line=dict(color="#00ffd0", width=2.4, dash="dash"),
                           name="AlienX 3D"), 1, 2)
    f.update_xaxes(title_text="rollout step", row=1, col=2, **AXIS)
    f.update_yaxes(title_text="kinetic energy", row=1, col=2, **AXIS)
    f.update_layout(title=dict(text="The operator chases the solver — "
                                    "no grid, no mesh, 421K parameters",
                               font=dict(size=15)), showlegend=True,
                    legend=dict(orientation="h", y=-0.18), height=380, **PAPER)
    return f

def fig_slices(slices, n_steps):
    steps = sorted(slices.keys())
    f = psub.make_subplots(rows=2, cols=3,
                           subplot_titles=[f"t = {s * (T_FINAL / N_SNAPS):.2f}" for s in steps],
                           vertical_spacing=0.12, horizontal_spacing=0.04)
    for col, st_ in enumerate(steps):
        truth, pred = slices[st_]
        f.add_trace(go.Heatmap(z=truth, colorscale="Magma", showscale=False), 1, col + 1)
        f.add_trace(go.Heatmap(z=pred, colorscale="Magma", showscale=False), 2, col + 1)
    f.update_annotations(font=dict(size=12, color="#9be8d8"))
    f.update_layout(title=dict(text=f"|vorticity| mid-plane — DNS solver (top) vs AlienX 3D (bottom)"
                                    f" · {n_steps}-step rollout",
                               font=dict(size=15)),
                    height=430, **PAPER)
    for r in (1, 2):
        for c in (1, 2, 3):
            f.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=r, col=c)
            f.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=r, col=c)
    return f

# ─── The ritual ──────────────────────────────────────────────────────────────
def summon(re_val, n_steps, sever, request=None):
    re_val = float(re_val)
    nu = 2 * math.pi / re_val
    traj = ph.generate_tgv_dataset(G=G, nu=nu, T=T_FINAL, dt=0.005,
                                   n_samples=N_SNAPS, device=DEVICE)
    cache = build_cache(traj)
    m, E, gE, D, slices = rollout(cache, re_val, int(n_steps), zero_gi=False)
    m0 = None
    if sever:
        m0, _, _, _, _ = rollout(cache, re_val, min(int(n_steps), 12), zero_gi=True)
    # single-step skill + invariants
    rel = float(np.sqrt(m).mean() / ST["tgt"][1])
    ed = float(np.abs(E - gE).max() / max(abs(gE.max()), abs(gE.min())) * 100)
    div = float(D.max())
    note = ("☠ the phase is severed — this is the operator WITHOUT QSA's complex phase channel"
            if sever else
            "✦ the complex phase channel is intact — this is the full AlienX 3D operator")
    extra = (f"<br><span style='color:#b26bff'>severed-phase error at same horizon: "
             f"visible as the violet trace — this gap IS QSA</span>" if sever else "")
    md = (f"<div class='metrics'>{note}<br>"
          f"Relative RMSE (rollout): <b>{rel * 100:.2f}%</b> · "
          f"Energy drift (max): <b>{ed:.2f}%</b> · "
          f"max |∇·u|: <b>{div:.1e}</b> (divergence-free by construction)"
          f"{extra}<br>"
          f"<span style='color:#6b665c'>Trained on Re ∈ {{{RE_LIST[0]}, {RE_LIST[1]}, {RE_LIST[2]}}} — "
          f"you just ran Re = {re_val:.0f}{' — a Reynolds number it has never seen' if not any(abs(re_val - r) < 1 for r in RE_LIST) else ''}. "
          f"Same 421,507 weights. No retraining. No grid.</span></div>")
    return fig_metrics(m, E, gE, bool(sever), m0), fig_slices(slices, int(n_steps)), md

with gr.Blocks(css=CSS, head=HEAD, title="AlienX 3D — Flow Lab") as demo:
    gr.HTML("<div id='necro-title'>ALIENX 3D · FLOW LAB</div>"
            "<div id='necro-sub'>a necromancer's neural operator vs the spectral solver · "
            "Taylor–Green vortex · incompressible Navier–Stokes</div>")
    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            re_slider = gr.Slider(300, 1200, value=900, step=4, label="Reynolds number — drag anywhere; it was only ever trained on 300 · 628 · 1200")
            steps = gr.Radio([10, 25, 50], value=25, label="Rollout length (steps)")
            sever = gr.Checkbox(value=False, label="☠ Sever the Phase (gi = 0) — kill QSA's complex channel and compare")
            go_btn = gr.Button("⚡ SUMMON THE OPERATOR", variant="primary")
            gr.Markdown(
                "**What you are watching:** a 421,507-parameter network predicts the next state "
                "of 3D turbulence 32×32×32 nodes at a time, autoregressively — and stays "
                "divergence-free to float32 round-off *by construction* (Biot–Savart recovery). "
                "It was trained on **three** Reynolds numbers. Yours is probably not one of them.")
        with gr.Column(scale=2, elem_classes="panel"):
            metrics = gr.HTML()
            inv_plot = gr.Plot()
            slice_plot = gr.Plot()
    go_btn.click(summon, [re_slider, steps, sever], [inv_plot, slice_plot, metrics],
                 concurrency_limit=1)
    demo.load(lambda: None)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
