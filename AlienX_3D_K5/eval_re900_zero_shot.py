# ═══════════════════════════════════════════════════════════════════════════
#  AlienX 3D — ZERO-SHOT HELD-OUT REYNOLDS EVALUATION (Re=900)
#  Loads the released epoch-55 K=3 checkpoint and tests on a Reynolds number
#  never seen in training (trained on 300/628/1200 only).
#
#  Protocol:
#   [1] SANITY: reproduce the in-distribution Re=300 val pair (t=72→73).
#       Expected MSE ≈ 6.032e-05 (from alienx_k5_test_metrics_v2.npy).
#       If this fails, the harness is wrong — no new number is trustworthy.
#   [2] Re=900 single-step over ALL 73 pairs of the trajectory.
#   [3] Re=900 autoregressive rollout (73 steps, full T=10) with the same
#       diagnostics as the paper table (peak-normalized energy/enstrophy err).
#   [4] gi=0 phase ablation at Re=900 — is the complex phase channel
#       load-bearing on unseen regimes? (QSA-specific claim.)
#
#  Model / physics code is copied 1:1 from the v2 final test script.
# ═══════════════════════════════════════════════════════════════════════════
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc, math, pickle, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stages (so the run fits in short command windows):
#   quick    : sanity Re=300 + DNS Re=900 + single-step over all pairs
#   rollout  : 73-step autoregressive rollout at Re=900
#   ablation : gi=0 phase ablation at Re=900
#   all      : everything in one go
parser = argparse.ArgumentParser()
parser.add_argument('--stage', default='all', choices=['quick', 'rollout', 'ablation', 'all'])
ARGS = parser.parse_args()

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE, 'alienx_k5_best.pt')
STATS_PATH = os.path.join(BASE, 'alienx_k5_stats.pkl')
OUT_PATH = os.path.join(BASE, 'alienx_k5_re900_zero_shot.npy')
TRAJ900_CACHE = os.path.join(BASE, 're900_traj_cache.pt')
RESULTS_CACHE = os.path.join(BASE, 're900_results_cache.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

with open(STATS_PATH, 'rb') as f:
    SD = pickle.load(f)
ST = SD['ST']; RE_LIST = SD['RE_LIST']; RE_MAX = SD['RE_MAX']
G = SD['G']; T_FINAL = SD['T_FINAL']; N_SNAPSHOTS = SD['N_SNAPSHOTS']
print(f"G={G}  trained Re={RE_LIST}  T={T_FINAL}  snapshots={N_SNAPSHOTS}")

RE_HELD_OUT = 900  # never seen in training; interpolates in [300, 1200]

def _s(x, key):
    mu, sg = ST[key]; return (x - mu) / sg
def _u(x, key):
    mu, sg = ST[key]; return x * sg + mu

# ═══════════════════════════════════════════════════════════════════════════
#  SPECTRAL PRIMITIVES  (verbatim from v2 final test script)
# ═══════════════════════════════════════════════════════════════════════════
def make_wavenumbers(G, device):
    kx = 2*math.pi*torch.fft.fftfreq(G, d=1.0/G, device=device)
    ky = 2*math.pi*torch.fft.fftfreq(G, d=1.0/G, device=device)
    kz = 2*math.pi*torch.fft.fftfreq(G, d=1.0/G, device=device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1.0
    return KX, KY, KZ, K2

def projection(u_hat, v_hat, w_hat, KX, KY, KZ, K2):
    kd = KX*u_hat + KY*v_hat + KZ*w_hat
    return (u_hat - KX*kd/K2, v_hat - KY*kd/K2, w_hat - KZ*kd/K2)

def nonlinear_term(u_hat, v_hat, w_hat, KX, KY, KZ, K2):
    u = torch.fft.ifftn(u_hat, dim=(-3,-2,-1)).real
    v = torch.fft.ifftn(v_hat, dim=(-3,-2,-1)).real
    w = torch.fft.ifftn(w_hat, dim=(-3,-2,-1)).real
    du_dx = torch.fft.ifftn(1j*KX*u_hat, dim=(-3,-2,-1)).real
    du_dy = torch.fft.ifftn(1j*KY*u_hat, dim=(-3,-2,-1)).real
    du_dz = torch.fft.ifftn(1j*KZ*u_hat, dim=(-3,-2,-1)).real
    dv_dx = torch.fft.ifftn(1j*KX*v_hat, dim=(-3,-2,-1)).real
    dv_dy = torch.fft.ifftn(1j*KY*v_hat, dim=(-3,-2,-1)).real
    dv_dz = torch.fft.ifftn(1j*KZ*v_hat, dim=(-3,-2,-1)).real
    dw_dx = torch.fft.ifftn(1j*KX*w_hat, dim=(-3,-2,-1)).real
    dw_dy = torch.fft.ifftn(1j*KY*w_hat, dim=(-3,-2,-1)).real
    dw_dz = torch.fft.ifftn(1j*KZ*w_hat, dim=(-3,-2,-1)).real
    ax = u*du_dx + v*du_dy + w*du_dz
    ay = u*dv_dx + v*dv_dy + w*dv_dz
    az = u*dw_dx + v*dw_dy + w*dw_dz
    ax_hat = torch.fft.fftn(ax, dim=(-3,-2,-1))
    ay_hat = torch.fft.fftn(ay, dim=(-3,-2,-1))
    az_hat = torch.fft.fftn(az, dim=(-3,-2,-1))
    G = u_hat.shape[-1]; kmax = G // 2
    kmask = ((torch.abs(KX) <= 2*kmax/3) & (torch.abs(KY) <= 2*kmax/3) & (torch.abs(KZ) <= 2*kmax/3))
    ax_hat, ay_hat, az_hat = ax_hat*kmask, ay_hat*kmask, az_hat*kmask
    ax_hat, ay_hat, az_hat = projection(ax_hat, ay_hat, az_hat, KX, KY, KZ, K2)
    return -ax_hat, -ay_hat, -az_hat

def tgv_imex_step(u_hat, v_hat, w_hat, KX, KY, KZ, K2, nu, dt):
    nhx, nhy, nhz = nonlinear_term(u_hat, v_hat, w_hat, KX, KY, KZ, K2)
    vf = 1.0 + dt*nu*K2
    return (u_hat + dt*nhx)/vf, (v_hat + dt*nhy)/vf, (w_hat + dt*nhz)/vf

def generate_tgv_dataset(G=32, nu=0.01, T=10.0, dt=0.005, n_samples=74, device=device):
    print(f"    TGV G={G}, nu={nu:.5f} (Re~{2*math.pi/nu:.0f}), T={T}")
    x = torch.arange(G, device=device, dtype=torch.float32) * (2*math.pi/G)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    u = torch.sin(X)*torch.cos(Y)*torch.cos(Z)
    v = -torch.cos(X)*torch.sin(Y)*torch.cos(Z)
    w = torch.zeros_like(u)
    u_hat = torch.fft.fftn(u, dim=(-3,-2,-1))
    v_hat = torch.fft.fftn(v, dim=(-3,-2,-1))
    w_hat = torch.fft.fftn(w, dim=(-3,-2,-1))
    KX, KY, KZ, K2 = make_wavenumbers(G, device)
    snaps = []
    total = int(T/dt); save_every = max(1, total//n_samples)
    for step in range(total):
        u_hat, v_hat, w_hat = tgv_imex_step(u_hat, v_hat, w_hat, KX, KY, KZ, K2, nu, dt)
        u_hat, v_hat, w_hat = projection(u_hat, v_hat, w_hat, KX, KY, KZ, K2)
        if torch.isnan(u_hat).any():
            print(f"    NaN at step {step}"); return None
        if step % save_every == 0 and len(snaps) < n_samples:
            snaps.append(torch.stack([
                torch.fft.ifftn(u_hat, dim=(-3,-2,-1)).real.cpu(),
                torch.fft.ifftn(v_hat, dim=(-3,-2,-1)).real.cpu(),
                torch.fft.ifftn(w_hat, dim=(-3,-2,-1)).real.cpu(),
            ], dim=-1))
    return torch.stack(snaps, dim=0)

def velocity_to_vorticity(v, KX, KY, KZ):
    u = torch.fft.fftn(v[...,0], dim=(-3,-2,-1))
    vv = torch.fft.fftn(v[...,1], dim=(-3,-2,-1))
    w = torch.fft.fftn(v[...,2], dim=(-3,-2,-1))
    wx = torch.fft.ifftn(1j*(KY*w - KZ*vv), dim=(-3,-2,-1)).real
    wy = torch.fft.ifftn(1j*(KZ*u - KX*w), dim=(-3,-2,-1)).real
    wz = torch.fft.ifftn(1j*(KX*vv - KY*u), dim=(-3,-2,-1)).real
    return torch.stack([wx, wy, wz], dim=-1)

def vorticity_to_velocity(omega, KX, KY, KZ, K2):
    ox = torch.fft.fftn(omega[...,0], dim=(-3,-2,-1))
    oy = torch.fft.fftn(omega[...,1], dim=(-3,-2,-1))
    oz = torch.fft.fftn(omega[...,2], dim=(-3,-2,-1))
    ux_hat = 1j*(KY*oz - KZ*oy)/K2
    uy_hat = 1j*(KZ*ox - KX*oz)/K2
    uz_hat = 1j*(KX*oy - KY*ox)/K2
    ux_hat[...,0,0,0] = 0; uy_hat[...,0,0,0] = 0; uz_hat[...,0,0,0] = 0
    return torch.stack([
        torch.fft.ifftn(ux_hat, dim=(-3,-2,-1)).real,
        torch.fft.ifftn(uy_hat, dim=(-3,-2,-1)).real,
        torch.fft.ifftn(uz_hat, dim=(-3,-2,-1)).real,
    ], dim=-1)

def scalar_derivatives(k_grid, KX, KY, KZ):
    B, G_, _, _ = k_grid.shape
    k_hat = torch.fft.fftn(k_grid, dim=(-3,-2,-1))
    ops = torch.stack([1j*KX, 1j*KY, 1j*KZ, -(KX**2), -(KY**2), -(KZ**2),
                       -(KX*KY), -(KX*KZ), -(KY*KZ)], dim=0)
    d = torch.fft.ifftn(ops.unsqueeze(1)*k_hat.unsqueeze(0), dim=(-3,-2,-1)).real
    d = d.permute(1,2,3,4,0).reshape(B, -1, 9)
    gv = d[...,:3]; gm = gv.norm(dim=-1); k_flat = k_grid.reshape(B,-1)
    H_xx, H_yy, H_zz, H_xy, H_xz, H_yz = d[...,3:].unbind(-1)
    row0 = torch.stack([H_xx, H_xy, H_xz], dim=-1)
    row1 = torch.stack([H_xy, H_yy, H_yz], dim=-1)
    row2 = torch.stack([H_xz, H_yz, H_zz], dim=-1)
    H = torch.stack([row0, row1, row2], dim=-2)
    return k_flat, gm, gv, H

def kinetic_energy(v):
    return 0.5 * (v**2).sum(dim=-1).mean(dim=(-3,-2,-1))

def enstrophy(v, KX, KY, KZ):
    om = velocity_to_vorticity(v, KX, KY, KZ)
    return 0.5 * (om**2).sum(dim=-1).mean(dim=(-3,-2,-1))

def divergence_spectral(v, KX, KY, KZ):
    u = torch.fft.fftn(v[...,0], dim=(-3,-2,-1))
    vv = torch.fft.fftn(v[...,1], dim=(-3,-2,-1))
    w = torch.fft.fftn(v[...,2], dim=(-3,-2,-1))
    return torch.fft.ifftn(1j*(KX*u + KY*vv + KZ*w), dim=(-3,-2,-1)).real

def rms_velocity(v):
    return torch.sqrt((v**2).sum(dim=-1).mean(dim=(1,2,3)))

# ═══════════════════════════════════════════════════════════════════════════
#  STENCIL  (verbatim)
# ═══════════════════════════════════════════════════════════════════════════
ISO_OFFSETS_3D = []
for dx_ in range(-2, 3):
    for dy_ in range(-2, 3):
        for dz_ in range(-2, 3):
            if dx_ == 0 and dy_ == 0 and dz_ == 0: continue
            if dx_*dx_ + dy_*dy_ + dz_*dz_ <= 4:
                ISO_OFFSETS_3D.append((dx_, dy_, dz_))
K_ISO_3D = len(ISO_OFFSETS_3D)

def get_isotropic_knn_3d_periodic(grid_size, device, dilation=1):
    N = grid_size**3
    idx = torch.arange(N, device=device).view(1, 1, grid_size, grid_size, grid_size)
    pad = 2*dilation
    padded = F.pad(idx.float(), (pad,)*6, mode='circular').long().squeeze(0).squeeze(0)
    nl = []
    for dr, dc, dz in ISO_OFFSETS_3D:
        block = padded[pad+dr*dilation:pad+dr*dilation+grid_size,
                       pad+dc*dilation:pad+dc*dilation+grid_size,
                       pad+dz*dilation:pad+dz*dilation+grid_size]
        nl.append(block.reshape(-1))
    return torch.stack(nl, dim=-1).reshape(N, K_ISO_3D).unsqueeze(0)

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL  (verbatim)
# ═══════════════════════════════════════════════════════════════════════════
GI_ZERO = False  # global ablation flag: zero the imaginary (phase) geometry

class FrameBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('pert_mat', 1e-3 * torch.eye(3).unsqueeze(0))
        self.tau = 1e-3; self.epsilon_grad = 1e-4

    def forward(self, grad_k, H, coords, k):
        with torch.no_grad():
            B, N, _ = coords.shape
            mag = grad_k.norm(dim=-1, keepdim=True)
            n_grad = grad_k / (mag + 1e-8)
            w = torch.softmax(k.unsqueeze(-1), dim=1)
            S = torch.einsum('bnd,bnc->bdc', coords*w, coords)
            S = S / (S.norm(dim=(-2,-1), keepdim=True) + 1e-8)
            S = S + self.pert_mat
            S = 0.5*(S + S.transpose(-1, -2))
            evec = None
            for j in [0.0, 1e-4, 1e-3, 1e-2, 5e-2]:
                try:
                    S_try = S if j == 0 else S + j*torch.eye(3, device=S.device).unsqueeze(0)
                    if j > 0: S_try = 0.5*(S_try + S_try.transpose(-1, -2))
                    _, ev = torch.linalg.eigh(S_try)
                    evec = ev; break
                except RuntimeError: continue
            if evec is None:
                evec = torch.eye(3, device=S.device).unsqueeze(0).expand(B, -1, -1)
            fb = evec[..., -1].unsqueeze(1).expand(-1, N, -1)
            gate = torch.sigmoid((mag - self.epsilon_grad) / self.tau)
            n = F.normalize(gate*n_grad + (1-gate)*fb, dim=-1)
            evec_exp = evec.unsqueeze(1).expand(-1, N, -1, -1)
            dots = torch.einsum('bnij,bni->bnj', evec_exp, n)
            best_idx = dots.abs().argmin(dim=-1)
            ref = evec_exp.gather(3, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,3,1)).squeeze(-1)
            u1 = F.normalize(torch.cross(n, ref, dim=-1), dim=-1)
            u2 = torch.cross(n, u1, dim=-1)
            Hu1 = torch.einsum('bnij,bnj->bni', H, u1)
            Hu2 = torch.einsum('bnij,bnj->bni', H, u2)
            a = (u1*Hu1).sum(-1); b = (u1*Hu2).sum(-1); c = (u2*Hu2).sum(-1)
            th = 0.5*torch.atan2(2*b, a - c + 1e-8)
            e1 = torch.cos(th).unsqueeze(-1)*u1 + torch.sin(th).unsqueeze(-1)*u2
            Hn = torch.einsum('bnij,bnj->bni', H, grad_k)
            Ht = Hn - (Hn*n).sum(-1, keepdim=True)*n
            sgn = torch.sign((e1*Ht).sum(-1, keepdim=True))
            sgn = torch.where(Ht.norm(dim=-1, keepdim=True) > 1e-6, sgn, torch.ones_like(sgn))
            e1 = F.normalize(e1*sgn, dim=-1)
            e2 = torch.cross(n, e1, dim=-1)
            disc = torch.sqrt((a-c)**2 + 4*b**2 + 1e-8)
            conf = torch.sigmoid((disc - 1e-3)/1e-4)
        return e1, e2, n, conf

def precompute_geometry(coords, e1, e2, n, pc, knn_idx, sigma_z, G):
    B, N, _ = coords.shape
    K = knn_idx.shape[-1]
    dilation = max(1, G // 16)
    offsets = torch.tensor(ISO_OFFSETS_3D, device=coords.device, dtype=torch.float32)
    offsets = offsets * dilation * (2*math.pi / G)
    rel = offsets.view(1, 1, K, 3).expand(B, N, K, 3)
    bidx = torch.arange(B, device=coords.device).view(B, 1, 1)
    bx = bidx.expand(-1, N, K)
    with torch.no_grad():
        dxl = (rel * e1.unsqueeze(2)).sum(-1)
        dyl = (rel * e2.unsqueeze(2)).sum(-1)
        dzl = (rel * n.unsqueeze(2)).sum(-1)
        th = torch.atan2(dyl, dxl)
        wo = torch.exp(-(dzl**2)/(sigma_z**2))
        gr = wo*torch.cos(th)
        gi = wo*torch.sin(th)*pc.unsqueeze(-1)
        if GI_ZERO:
            gi = torch.zeros_like(gi)
    return gr, gi, bx

class QSA_ISNBlock3D(nn.Module):
    def __init__(self, C, vec_dim=3, heads=4, eps=1e-8):
        super().__init__()
        self.C = C; self.H = heads; self.D = C//heads
        self.eps = eps
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C + 2*vec_dim, 2*C)
        self.out_proj = nn.Linear(2*C, C)
        self.norm = nn.LayerNorm(C)
        self.last_im_re_ratio = 0.0

    def forward(self, h, gr, gi, bx, gk_l, om_l, knn):
        B, N, C = h.shape
        if knn.shape[0] != B:
            knn = knn.expand(B, -1, -1)
            gr = gr.expand(B, -1, -1); gi = gi.expand(B, -1, -1)
            bx = torch.arange(B, device=h.device).view(B,1,1).expand(B, N, knn.shape[-1])
        q = self.q_proj(h).view(B, N, self.H, self.D)
        k_node = self.k_proj(h).view(B, N, self.H, self.D)
        v_raw = self.v_proj(torch.cat([h, gk_l, om_l], dim=-1)).view(B, N, self.H, self.D, 2)
        v_node = torch.complex(v_raw[..., 0], v_raw[..., 1])
        K_j = k_node[bx, knn]
        V_j = v_node[bx, knn]
        S_c = torch.einsum('bnhd,bnkhd->bnkh', q, torch.conj(K_j))
        gr_u = gr.unsqueeze(-1); gi_u = gi.unsqueeze(-1)
        S_r = S_c * torch.complex(gr_u, gi_u)
        nrm = torch.sqrt((S_r.abs()**2).sum(dim=2, keepdim=True) + self.eps)
        alpha = S_r / nrm
        out_c = (alpha.unsqueeze(-1)*V_j).sum(dim=2)
        with torch.no_grad():
            rm = out_c.real.abs().mean().item()
            im = out_c.imag.abs().mean().item()
            self.last_im_re_ratio = im/(rm + 1e-8)
        out = self.out_proj(torch.cat([out_c.real.reshape(B, N, C),
                                        out_c.imag.reshape(B, N, C)], dim=-1))
        return h + 0.9*self.norm(out)

class ISNOperator3D(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, num_heads=4,
                 equivariant_output=True, use_velocity_input=True, use_re=True,
                 re_max=2000.0, sigma_z=0.1):
        super().__init__()
        self.equivariant_output = equivariant_output
        self.use_velocity_input = use_velocity_input
        self.use_re = use_re
        self.re_max = re_max
        self.sigma_z = sigma_z
        in_ch = 8 + (3 if use_velocity_input else 0) + (1 if use_re else 0)
        self.in_proj = nn.Linear(in_ch, hidden_dim)
        self.frame_builder = FrameBuilder()
        self.blocks = nn.ModuleList([
            QSA_ISNBlock3D(hidden_dim, 3, num_heads) for _ in range(num_layers)])
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3))
        self.G = 32

    def forward(self, coords, k, gk, gk_m, H, om, u, re_label, knn, G=None):
        B = k.shape[0]; N = k.shape[1]
        if coords.shape[0] == 1 and B > 1:
            coords = coords.expand(B, -1, -1)
        e1, e2, n, pc = self.frame_builder(gk, H, coords, k)
        Fm = torch.stack([e1, e2, n], dim=-1)
        G_eff = G if G is not None else self.G
        gr, gi, bx = precompute_geometry(coords, e1, e2, n, pc, knn, self.sigma_z, G_eff)
        gk_l = torch.einsum('bnji,bnj->bni', Fm, gk)
        om_l = torch.einsum('bnji,bnj->bni', Fm, om)
        feats = [k.unsqueeze(-1), gk_m.unsqueeze(-1), gk_l, om_l]
        if self.use_velocity_input:
            feats.append(torch.einsum('bnji,bnj->bni', Fm, u))
        if self.use_re:
            feats.append(torch.log(re_label/self.re_max).view(B,1,1).expand(-1,N,-1))
        h = self.in_proj(torch.cat(feats, dim=-1))
        for blk in self.blocks:
            h = blk(h, gr, gi, bx, gk_l, om_l, knn)
        v_local = self.out_proj(h)
        if self.equivariant_output:
            return torch.einsum('bnij,bnj->bni', Fm, v_local)
        return v_local

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading checkpoint...")
model = ISNOperator3D(hidden_dim=128, num_layers=4, num_heads=4,
                      equivariant_output=True, use_velocity_input=True,
                      use_re=True, re_max=RE_MAX).to(device)
model.G = G
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  loaded OK")

KX, KY, KZ, K2 = make_wavenumbers(G, device)
x = torch.arange(G, device=device, dtype=torch.float32) * (2*math.pi/G)
X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
coords_single = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).unsqueeze(0)
knn_idx = get_isotropic_knn_3d_periodic(G, device, dilation=max(1, G//16))

# GPU memory probe: this workload needs ~2.5 GB peak at B=1; if the 4 GB card
# cannot fit it, fall back to CPU before generating any data.
try:
    with torch.no_grad():
        _B = 1
        _f = torch.randn(_B, G*G*G, 12, device=device)
        _k = torch.randn(_B, G*G*G, device=device)
        _gv = torch.randn(_B, G*G*G, 3, device=device)
        _H = torch.eye(3, device=device).unsqueeze(0).expand(_B, G*G*G, 3, 3).contiguous()
        _ = model(coords_single, _k, _gv, _k.new_zeros(_B, G*G*G), _H,
                  _gv, _gv, torch.tensor([300.0], device=device), knn_idx, G=G)
    del _
    torch.cuda.empty_cache()
    print("  GPU memory probe: OK")
except torch.cuda.OutOfMemoryError:
    print("  GPU memory probe: OOM -> falling back to CPU")
    device = torch.device('cpu')
    model = model.to(device)
    KX, KY, KZ, K2 = make_wavenumbers(G, device)
    coords_single = coords_single.to(device)
    knn_idx = knn_idx.to(device)
    gc.collect(); torch.cuda.empty_cache()

def build_cache(traj):
    """Cache physics for one trajectory (moved to eval device)."""
    td = traj.to(device)
    Us = rms_velocity(td)
    oa = velocity_to_vorticity(td, KX, KY, KZ)
    return {'u': td, 'Us': Us, 'omega': oa}

@torch.no_grad()
def single_step_pair(cache, t, Re):
    """One (t -> t+1) prediction, per-sample normalized exactly as training."""
    U0 = cache['Us'][t].clamp(min=1e-6)
    om = cache['omega'][t:t+1] / U0
    kf = om.norm(dim=-1)
    k_, gm, gv, Hs = scalar_derivatives(kf, KX, KY, KZ)
    k_s = _s(k_, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
    H_s = _s(Hs, 'H'); om_s = _s(om.reshape(1,-1,3), 'om')
    u_s = _s((cache['u'][t:t+1]/U0).reshape(1,-1,3), 'u')
    tgt = _s((cache['omega'][t+1:t+2]/U0).reshape(1,-1,3), 'tgt')
    re_t = torch.tensor([float(Re)], device=device)
    pred = model(coords_single, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn_idx, G=G)
    return F.mse_loss(pred, tgt).item()

@torch.no_grad()
def single_step_all(cache, Re):
    """Single-step MSE over every pair of the trajectory."""
    mses = []
    T = cache['u'].shape[0] - 1
    for t in range(T):
        mses.append(single_step_pair(cache, t, Re))
        if (t+1) % 20 == 0:
            print(f"      step {t+1}/{T} ...")
    return np.array(mses)

@torch.no_grad()
def rollout_eval(cache, Re, n_steps):
    """Autoregressive rollout from t=0, fixed U0(t=0) — identical protocol to v2."""
    traj = cache['u']
    omega_traj = cache['omega']
    u_curr = traj[0:1].clone()
    omega_curr = omega_traj[0:1].clone()
    U0 = rms_velocity(u_curr).clamp(min=1e-6)
    mses, Es, gEs, Zs, gZs, Ds = [], [], [], [], [], []
    for step in range(n_steps):
        om_nd = omega_curr / U0
        kf = om_nd.norm(dim=-1)
        k_, gm, gv, Hs = scalar_derivatives(kf, KX, KY, KZ)
        u_nd = u_curr / U0
        k_s = _s(k_, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
        H_s = _s(Hs, 'H'); om_s = _s(om_nd.reshape(1,-1,3), 'om')
        u_s = _s(u_nd.reshape(1,-1,3), 'u')
        re_t = torch.tensor([float(Re)], device=device)
        pred_s = model(coords_single, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn_idx, G=G)
        om_next_nd = _u(pred_s.reshape(1,-1,3), 'tgt').reshape(1, G, G, G, 3)
        om_next = om_next_nd * U0
        u_curr = vorticity_to_velocity(om_next, KX, KY, KZ, K2)
        omega_curr = om_next
        if step < traj.shape[0] - 1:
            om_t = omega_traj[step+1:step+2]
            u_t = traj[step+1:step+2]
            mses.append(F.mse_loss(omega_curr, om_t).item())
            Es.append(kinetic_energy(u_curr).item()); gEs.append(kinetic_energy(u_t).item())
            Zs.append(enstrophy(u_curr, KX, KY, KZ).item()); gZs.append(enstrophy(u_t, KX, KY, KZ).item())
            Ds.append(divergence_spectral(u_curr, KX, KY, KZ).abs().mean().item())
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if (step+1) % 20 == 0:
            print(f"      rollout step {step+1}/{n_steps} ...")
    return (np.array(mses), np.array(Es), np.array(gEs),
            np.array(Zs), np.array(gZs), np.array(Ds))

def rollout_report(tag, Re, m, E, gE, Z, gZ, D):
    peak_E = max(abs(gE.max()), abs(gE.min()))
    peak_Z = max(abs(gZ.max()), abs(gZ.min()))
    ed = (np.abs(E - gE).max() / peak_E) * 100
    zd = (np.abs(Z - gZ).max() / peak_Z) * 100
    print(f"  {tag} Re={Re}:")
    print(f"    steps={len(m)}  MSE[1]={m[0]:.6f}  MSE[10]={m[min(10,len(m)-1)]:.6f}  MSE[-1]={m[-1]:.6f}")
    print(f"    MSE growth      = {m[-1]/m[0]:.2f}x")
    print(f"    Energy err (max)    = {ed:+.3f}%")
    print(f"    Enstrophy err (max) = {zd:+.3f}%")
    print(f"    max|div u|      = {D.max():.3e}")
    return {'MSE1': float(m[0]), 'MSE10': float(m[min(10,len(m)-1)]),
            'MSE_final': float(m[-1]), 'MSE_growth': float(m[-1]/m[0]),
            'energy_err_max_pct': float(ed), 'enstrophy_err_max_pct': float(zd),
            'max_div': float(D.max())}

DO_QUICK = ARGS.stage in ('quick', 'all')
DO_ROLLOUT = ARGS.stage in ('rollout', 'all')
DO_ABLATION = ARGS.stage in ('ablation', 'all')

results = {'re_held_out': RE_HELD_OUT, 're_trained': RE_LIST, 'ckpt': os.path.basename(CKPT_PATH)}
if os.path.exists(RESULTS_CACHE):
    results.update(np.load(RESULTS_CACHE, allow_pickle=True).item())

cache900 = None

# ═══════════════════════════════════════════════════════════════════════════
#  [2/5] SANITY — reproduce in-distribution Re=300 val pair
# ═══════════════════════════════════════════════════════════════════════════
if DO_QUICK and 'sanity_pass' not in results:
    print("\n[2/5] SANITY check: in-distribution Re=300 val pair (t=72->73)...")
    print("  generating DNS Re=300 ...")
    traj300 = generate_tgv_dataset(G=G, nu=2*math.pi/300, T=T_FINAL, dt=0.005,
                                   n_samples=N_SNAPSHOTS, device=device)
    cache300 = build_cache(traj300)
    mse300 = single_step_pair(cache300, 72, 300)
    expected = 6.0323800425976515e-05  # from alienx_k5_test_metrics_v2.npy
    ratio = mse300 / expected
    ok = (0.5 < ratio < 2.0)
    print(f"  Re=300 t=72->73 MSE = {mse300:.6e}  (paper: {expected:.6e}, ratio {ratio:.3f})")
    print(f"  SANITY {'PASS' if ok else 'FAIL'} — harness {'verified' if ok else 'BROKEN, do not trust new numbers'}")
    results['sanity_re300_mse'] = float(mse300)
    results['sanity_expected'] = float(expected)
    results['sanity_pass'] = bool(ok)
    if not ok:
        print("  Aborting: fix harness before evaluating held-out Re.")
        np.save(OUT_PATH, results, allow_pickle=True)
        raise SystemExit(1)
    cache300 = None; traj300 = None
    gc.collect(); _ = cache300

# ═══════════════════════════════════════════════════════════════════════════
#  [3/5] Re=900 — generate + single-step over all pairs
# ═══════════════════════════════════════════════════════════════════════════
if DO_QUICK and 're900_single_step' not in results:
    print(f"\n[3/5] Held-out Re={RE_HELD_OUT} (never in training)...")
    if os.path.exists(TRAJ900_CACHE):
        print("  loading cached Re=900 trajectory ...")
        traj900 = torch.load(TRAJ900_CACHE, map_location=device)
    else:
        print("  generating DNS ...")
        t0 = time.time()
        traj900 = generate_tgv_dataset(G=G, nu=2*math.pi/RE_HELD_OUT, T=T_FINAL, dt=0.005,
                                       n_samples=N_SNAPSHOTS, device=device)
        print(f"  DNS done in {time.time()-t0:.0f}s")
        assert traj900 is not None and traj900.shape[0] == N_SNAPSHOTS
        torch.save(traj900.cpu(), TRAJ900_CACHE)
        traj900 = traj900.to(device)
    cache900 = build_cache(traj900)

    print(f"  single-step over all {N_SNAPSHOTS-1} pairs ...")
    m_all = single_step_all(cache900, RE_HELD_OUT)
    rel_all = np.sqrt(m_all) / ST['tgt'][1]
    mse_72 = m_all[-1]
    print(f"\n  Re={RE_HELD_OUT} single-step:")
    print(f"    mean MSE        = {m_all.mean():.6f}")
    print(f"    mean rel RMSE   = {rel_all.mean()*100:.2f}%   (trained Re mean was 0.46%)")
    print(f"    median rel RMSE = {np.median(rel_all)*100:.2f}%")
    print(f"    worst rel RMSE  = {rel_all.max()*100:.2f}%  (t={int(np.argmax(rel_all))+1}->t+1)")
    print(f"    val pair t=72->73  MSE = {mse_72:.6f}  rel = {np.sqrt(mse_72)/ST['tgt'][1]*100:.2f}%"
          f"   <- apples-to-apples with training val")
    results['re900_single_step'] = {
        'mean_mse': float(m_all.mean()),
        'mean_rel_rmse': float(rel_all.mean()),
        'median_rel_rmse': float(np.median(rel_all)),
        'worst_rel_rmse': float(rel_all.max()),
        'val_pair_mse': float(mse_72),
        'val_pair_rel_rmse': float(np.sqrt(mse_72)/ST['tgt'][1]),
        'per_step_mse': m_all.tolist(),
    }
    cache900 = None; traj900 = None
    gc.collect(); _ = cache900, traj900

if cache900 is None and (DO_ROLLOUT or DO_ABLATION):
    if os.path.exists(TRAJ900_CACHE):
        traj900 = torch.load(TRAJ900_CACHE, map_location=device)
    else:
        print(f"\n  generating DNS Re={RE_HELD_OUT} ...")
        traj900 = generate_tgv_dataset(G=G, nu=2*math.pi/RE_HELD_OUT, T=T_FINAL, dt=0.005,
                                       n_samples=N_SNAPSHOTS, device=device)
        assert traj900 is not None and traj900.shape[0] == N_SNAPSHOTS
        torch.save(traj900.cpu(), TRAJ900_CACHE)
        traj900 = traj900.to(device)
    cache900 = build_cache(traj900)

# ═══════════════════════════════════════════════════════════════════════════
#  [4/5] Re=900 rollout
# ═══════════════════════════════════════════════════════════════════════════
if DO_ROLLOUT and 're900_rollout' not in results:
    print(f"\n[4/5] Re={RE_HELD_OUT} rollout ({N_SNAPSHOTS-1} steps, full T={T_FINAL}) ...")
    m, E, gE, Z, gZ, D = rollout_eval(cache900, RE_HELD_OUT, traj900.shape[0]-1)
    results['re900_rollout'] = rollout_report("HELD-OUT", RE_HELD_OUT, m, E, gE, Z, gZ, D)

# ═══════════════════════════════════════════════════════════════════════════
#  [5/5] gi=0 phase ablation at Re=900
# ═══════════════════════════════════════════════════════════════════════════
if DO_ABLATION and 're900_gi_ablation_ratio' not in results:
    print(f"\n[5/5] gi=0 PHASE ABLATION at Re={RE_HELD_OUT} (attention kept, phase killed) ...")
    GI_ZERO = True
    m_no_phase = single_step_all(cache900, RE_HELD_OUT)
    GI_ZERO = False
    ratio_phase = m_no_phase.mean() / results['re900_single_step']['mean_mse']
    print(f"    MSE with phase (gi):   {results['re900_single_step']['mean_mse']:.6f}")
    print(f"    MSE without phase:     {m_no_phase.mean():.6f}")
    print(f"    phase ablation ratio:  {ratio_phase:.2f}x")
    results['re900_gi_ablation_ratio'] = float(ratio_phase)
    results['re900_gi0_mean_mse'] = float(m_no_phase.mean())

# ═══════════════════════════════════════════════════════════════════════════
np.save(OUT_PATH, results, allow_pickle=True)
np.save(RESULTS_CACHE, results, allow_pickle=True)
print(f"\nSaved: {OUT_PATH}")
print(f"\n{'='*70}\n  ZERO-SHOT SUMMARY (Re={RE_HELD_OUT}, trained on {RE_LIST})\n{'='*70}")
print(f"  sanity Re=300 reproduced:      {'PASS' if results.get('sanity_pass') else 'NOT RUN / FAIL'}")
if 're900_single_step' in results:
    print(f"  single-step mean rel RMSE:     {results['re900_single_step']['mean_rel_rmse']*100:.2f}%  (trained: 0.46%)")
    print(f"  val-pair rel RMSE:             {results['re900_single_step']['val_pair_rel_rmse']*100:.2f}%")
if 're900_rollout' in results:
    print(f"  rollout MSE growth:            {results['re900_rollout']['MSE_growth']:.2f}x")
    print(f"  rollout energy err (max):      {results['re900_rollout']['energy_err_max_pct']:.3f}%")
    print(f"  rollout max|div u|:            {results['re900_rollout']['max_div']:.2e}")
if 're900_gi_ablation_ratio' in results:
    print(f"  gi=0 phase ablation ratio:     {results['re900_gi_ablation_ratio']:.2f}x")
print(f"{'='*70}")
