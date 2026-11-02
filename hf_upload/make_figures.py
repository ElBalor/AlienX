# Generates paper figures from the released checkpoint + regenerated DNS.
#   stage A: rollouts for Re=300, 628   -> rollout_arrays_<Re>.npz
#   stage B: rollouts for Re=900, 1200  -> rollout_arrays_<Re>.npz
#   stage plot: renders the three paper figures (300 DPI, serif, grid)
# Figures:
#   fig1_mse_growth.png  - rollout pointwise MSE vs time, all Re, log-y
#   fig2_invariants.png  - kinetic energy + enstrophy: model vs DNS (2 panels)
#   fig3_divergence.png  - |div u| vs time, log-y, all Re
import os, math, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'hf_upload'))
from model import ISNOperator3D, get_isotropic_knn_3d_periodic
import physics as ph

parser = argparse.ArgumentParser()
parser.add_argument('--stage', default='plot', choices=['A', 'B', 'plot'])
ARGS = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(os.path.join(BASE, 'hf_upload', 'alienx_k5_stats.pkl'), 'rb') as f:
    SD = pickle.load(f)
ST = SD['ST']; RE_LIST = SD['RE_LIST']; RE_MAX = SD['RE_MAX']
G = SD['G']; T_FINAL = SD['T_FINAL']; N_SNAPSHOTS = SD['N_SNAPSHOTS']

model = ISNOperator3D(hidden_dim=128, num_layers=4, num_heads=4,
                      re_max=RE_MAX).to(device)
model.G = G
model.load_state_dict(torch.load(os.path.join(BASE, 'hf_upload', 'alienx_k5_best.pt'),
                                 map_location=device))
model.eval()

KX, KY, KZ, K2 = ph.make_wavenumbers(G, device)
x = torch.arange(G, device=device, dtype=torch.float32) * (2*math.pi/G)
X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).unsqueeze(0)
knn = get_isotropic_knn_3d_periodic(G, device, dilation=max(1, G//16))

def _s(v, key):
    mu, sg = ST[key]; return (v - mu) / sg
def _u(v, key):
    mu, sg = ST[key]; return v * sg + mu

@torch.no_grad()
def rollout_arrays(Re):
    """Returns per-step arrays: mse, E_model, E_dns, Z_model, Z_dns, div."""
    traj = ph.generate_tgv_dataset(G=G, nu=2*math.pi/Re, T=T_FINAL, dt=0.005,
                                   n_samples=N_SNAPSHOTS, device=device)
    u = traj.to(device)
    omega_traj = ph.velocity_to_vorticity(u, KX, KY, KZ)
    u_curr = u[0:1].clone()
    om_curr = omega_traj[0:1].clone()
    U0 = ph.rms_velocity(u_curr).clamp(min=1e-6)
    mses, Em, Ed, Zm, Zd, dv = [], [], [], [], [], []
    for step in range(u.shape[0] - 1):
        om_nd = om_curr / U0
        u_nd = u_curr / U0
        k_, gm, gv, Hs = ph.scalar_derivatives(om_nd.norm(dim=-1), KX, KY, KZ)
        re_t = torch.tensor([float(Re)], device=device)
        pred = model(coords, _s(k_, 'k'), _s(gv, 'gv'), _s(gm, 'gm'), _s(Hs, 'H'),
                     _s(om_nd.reshape(1, -1, 3), 'om'),
                     _s(u_nd.reshape(1, -1, 3), 'u'), re_t, knn, G=G)
        om_next = _u(pred.reshape(1, -1, 3), 'tgt').reshape(1, G, G, G, 3) * U0
        u_curr = ph.vorticity_to_velocity(om_next, KX, KY, KZ, K2)
        om_curr = om_next
        u_true = u[step+1:step+2]
        mses.append(F.mse_loss(om_curr, omega_traj[step+1:step+2]).item())
        Em.append(ph.kinetic_energy(u_curr).item())
        Ed.append(ph.kinetic_energy(u_true).item())
        Zm.append(ph.enstrophy(u_curr, KX, KY, KZ).item())
        Zd.append(ph.enstrophy(u_true, KX, KY, KZ).item())
        dv.append(ph.divergence_spectral(u_curr, KX, KY, KZ).abs().mean().item())
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return {k: np.array(v) for k, v in
            dict(mse=mses, Em=Em, Ed=Ed, Zm=Zm, Zd=Zd, div=dv).items()}

if ARGS.stage in ('A', 'B'):
    todo = [300, 628] if ARGS.stage == 'A' else [900, 1200]
    for Re in todo:
        out = os.path.join(BASE, f'rollout_arrays_{Re}.npz')
        if os.path.exists(out):
            print(f'Re={Re}: cached')
            continue
        print(f'Re={Re}: rolling out 73 steps ...')
        arrs = rollout_arrays(Re)
        np.savez(out, **arrs)
        print(f'  saved {out}  (final MSE {arrs["mse"][-1]:.5f}, max div {arrs["div"].max():.2e})')
    raise SystemExit

# ── plot ─────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {300: '#2c7bb6', 628: '#abd9e9', 900: '#f46d43', 1200: '#d7191c'}
LABELS = {300: 'Re=300 (trained)', 628: 'Re=628 (trained)',
          900: 'Re=900 (held-out)', 1200: 'Re=1200 (trained)'}
DT = T_FINAL / (N_SNAPSHOTS - 1)
t = np.arange(1, N_SNAPSHOTS) * DT

data = {Re: dict(np.load(os.path.join(BASE, f'rollout_arrays_{Re}.npz')))
        for Re in [300, 628, 900, 1200]}

# Figure 1: MSE growth
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for Re in [300, 628, 900, 1200]:
    m = data[Re]['mse']
    ax.plot(t, np.maximum(m, 1e-8), color=COLORS[Re], lw=1.8, label=LABELS[Re])
ax.set_yscale('log')
ax.set_xlabel('physical time  t')
ax.set_ylabel('pointwise MSE (model vs DNS, vorticity)')
ax.set_title('Autoregressive rollout error over the full trajectory (73 steps)', fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig1_mse_growth.png'), dpi=300)
plt.close(fig)

# Figure 2: invariants, model vs DNS — 2 panels
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, (km, kd, ylab, title) in zip(
        axes,
        [('Em', 'Ed', 'kinetic energy  E(t)', 'Kinetic energy'),
         ('Zm', 'Zd', 'enstrophy  Z(t)', 'Enstrophy')]):
    for Re in [300, 628, 900, 1200]:
        ax.plot(t, data[Re][kd], color=COLORS[Re], lw=2.4, alpha=0.9)
        ax.plot(t, data[Re][km], color=COLORS[Re], lw=1.3, linestyle='--')
    ax.set_xlabel('physical time  t')
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11)
from matplotlib.lines import Line2D
legend = [Line2D([0], [0], color=COLORS[r], lw=2.4, label=LABELS[r]) for r in [300, 628, 900, 1200]]
legend += [Line2D([0], [0], color='gray', lw=2.4, label='DNS (solver)'),
           Line2D([0], [0], color='gray', lw=1.3, linestyle='--', label='AlienX 3D (model)')]
fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(BASE, 'fig2_invariants.png'), dpi=300)
plt.close(fig)

# Figure 3: divergence
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for Re in [300, 628, 900, 1200]:
    ax.plot(t, np.maximum(data[Re]['div'], 1e-16), color=COLORS[Re], lw=1.8, label=LABELS[Re])
ax.set_yscale('log')
ax.set_ylim(1e-8, 1e-4)
ax.set_xlabel('physical time  t')
ax.set_ylabel(r'$|\nabla \cdot u|$  (mean abs)')
ax.set_title('Divergence during rollout — free by construction (Biot–Savart projection)', fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig3_divergence.png'), dpi=300)
plt.close(fig)

print('figures written: fig1_mse_growth.png, fig2_invariants.png, fig3_divergence.png')
