# ═══════════════════════════════════════════════════════════════════════════
#  AlienX 3D — K=5 Fast Run (Colab, Drive-mounted, best.pt only)
#  Fixes: spectral units (mode bins), stencil-offset frame, DC leak
# ═══════════════════════════════════════════════════════════════════════════

import os, gc, math, time, pickle, shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─── Mount Drive first ──────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

DRIVE_DIR = '/content/drive/MyDrive/AlienX_3D_K5'
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f"Drive save path: {DRIVE_DIR}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════
#  1. SPECTRAL PRIMITIVES
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
    kmask = ((torch.abs(KX) <= 2*kmax/3) &
             (torch.abs(KY) <= 2*kmax/3) &
             (torch.abs(KZ) <= 2*kmax/3))
    ax_hat, ay_hat, az_hat = ax_hat*kmask, ay_hat*kmask, az_hat*kmask
    ax_hat, ay_hat, az_hat = projection(ax_hat, ay_hat, az_hat, KX, KY, KZ, K2)
    return -ax_hat, -ay_hat, -az_hat


def tgv_imex_step(u_hat, v_hat, w_hat, KX, KY, KZ, K2, nu, dt):
    nhx, nhy, nhz = nonlinear_term(u_hat, v_hat, w_hat, KX, KY, KZ, K2)
    vf = 1.0 + dt*nu*K2
    return (u_hat + dt*nhx)/vf, (v_hat + dt*nhy)/vf, (w_hat + dt*nhz)/vf


def generate_tgv_dataset(G=32, nu=0.01, T=10.0, dt=0.005, n_samples=74, device=device):
    print(f"    TGV G={G}, ν={nu:.5f} (Re≈{2*math.pi/nu:.0f}), T={T}")
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


# ═══════════════════════════════════════════════════════════════════════════
#  2. PHYSICS OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

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
    B, G, _, _ = k_grid.shape
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


def radial_spectra(v, G, KX, KY, KZ, K2):
    """E(k), Z(k) in integer-mode bins. DC of Z explicitly zeroed."""
    B = v.shape[0]
    uh = torch.fft.fftn(v[...,0], dim=(-3,-2,-1))
    vh = torch.fft.fftn(v[...,1], dim=(-3,-2,-1))
    wh = torch.fft.fftn(v[...,2], dim=(-3,-2,-1))
    E_density = 0.5 * (uh.abs()**2 + vh.abs()**2 + wh.abs()**2) / (G**6)
    Z_density = 0.5 * K2.unsqueeze(0) * (uh.abs()**2 + vh.abs()**2 + wh.abs()**2) / (G**6)
    Z_density[..., 0, 0, 0] = 0

    kmag_modes = torch.sqrt(K2) / (2*math.pi)     # integer mode magnitude
    n_bins = G // 2                                # 16 bins at G=32
    k_bins = torch.arange(0, n_bins + 2, device=v.device).float()

    E_k = torch.zeros(B, n_bins, device=v.device)
    Z_k = torch.zeros_like(E_k)
    for i in range(n_bins):
        mask = (kmag_modes >= k_bins[i]) & (kmag_modes < k_bins[i+1])
        E_k[:, i] = (E_density * mask.unsqueeze(0)).sum(dim=(-3,-2,-1))
        Z_k[:, i] = (Z_density * mask.unsqueeze(0)).sum(dim=(-3,-2,-1))
    return E_k, Z_k


# ═══════════════════════════════════════════════════════════════════════════
#  3. STENCIL
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


KNN_CACHE = {}
def get_cached_iso_knn_3d(grid_size, device):
    key = (grid_size, str(device))
    if key not in KNN_CACHE:
        KNN_CACHE[key] = get_isotropic_knn_3d_periodic(
            grid_size, device, dilation=max(1, grid_size//16))
    return KNN_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════
#  4. FRAME BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class FrameBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('pert_mat', 1e-3 * torch.eye(3).unsqueeze(0))
        self.tau = 1e-3
        self.epsilon_grad = 1e-4

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
                except RuntimeError:
                    continue
            if evec is None:
                evec = torch.eye(3, device=S.device).unsqueeze(0).expand(B, -1, -1)

            fb = evec[..., -1].unsqueeze(1).expand(-1, N, -1)
            gate = torch.sigmoid((mag - self.epsilon_grad) / self.tau)
            n = F.normalize(gate*n_grad + (1-gate)*fb, dim=-1)

            evec_exp = evec.unsqueeze(1).expand(-1, N, -1, -1)
            dots = torch.einsum('bnij,bni->bnj', evec_exp, n)
            abs_dots = dots.abs()
            best_idx = abs_dots.argmin(dim=-1)
            ref = evec_exp.gather(
                3, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 1)
            ).squeeze(-1)

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
            sgn = torch.where(Ht.norm(dim=-1, keepdim=True) > 1e-6, sgn,
                              torch.ones_like(sgn))
            e1 = F.normalize(e1*sgn, dim=-1)
            e2 = torch.cross(n, e1, dim=-1)

            disc = torch.sqrt((a-c)**2 + 4*b**2 + 1e-8)
            conf = torch.sigmoid((disc - 1e-3)/1e-4)

        return e1, e2, n, conf


def precompute_geometry(coords, e1, e2, n, pc, knn_idx, sigma_z, G):
    """Stencil offsets for exact rel vectors. Equivariant by construction."""
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
    return gr, gi, bx


# ═══════════════════════════════════════════════════════════════════════════
#  5. QSA BLOCK
# ═══════════════════════════════════════════════════════════════════════════

class QSA_ISNBlock3D(nn.Module):
    def __init__(self, C, vec_dim=3, heads=4, eps=1e-8, chunk=0):
        super().__init__()
        self.C = C; self.H = heads; self.D = C//heads
        self.eps = eps; self.chunk = chunk
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C + 2*vec_dim, 2*C)
        self.out_proj = nn.Linear(2*C, C)
        self.norm = nn.LayerNorm(C)
        self.last_im_re_ratio = 0.0

    def _chunk_forward(self, h_c, q_c, k_node, v_node, gr_c, gi_c, bx_c, knn_c):
        B, cs, C = h_c.shape
        K_j = k_node[bx_c, knn_c]
        V_j = v_node[bx_c, knn_c]
        S_c = torch.einsum('bnhd,bnkhd->bnkh', q_c, torch.conj(K_j))
        gr_u = gr_c.unsqueeze(-1); gi_u = gi_c.unsqueeze(-1)
        S_r = S_c * torch.complex(gr_u, gi_u)
        nrm = torch.sqrt((S_r.abs()**2).sum(dim=2, keepdim=True) + self.eps)
        alpha = S_r / nrm
        out_c = (alpha.unsqueeze(-1)*V_j).sum(dim=2)
        with torch.no_grad():
            rm = out_c.real.abs().mean().item()
            im = out_c.imag.abs().mean().item()
            self.last_im_re_ratio = im/(rm + 1e-8)
        return self.out_proj(torch.cat([out_c.real.reshape(B, cs, C),
                                        out_c.imag.reshape(B, cs, C)], dim=-1))

    def forward(self, h, gr, gi, bx, gk_l, om_l, knn):
        B, N, C = h.shape
        if knn.shape[0] != B:
            knn = knn.expand(B, -1, -1)
            gr = gr.expand(B, -1, -1)
            gi = gi.expand(B, -1, -1)
            bx = torch.arange(B, device=h.device).view(B,1,1).expand(B, N, knn.shape[-1])
        q = self.q_proj(h).view(B, N, self.H, self.D)
        k_node = self.k_proj(h).view(B, N, self.H, self.D)
        v_raw = self.v_proj(torch.cat([h, gk_l, om_l], dim=-1)).view(B, N, self.H, self.D, 2)
        v_node = torch.complex(v_raw[..., 0], v_raw[..., 1])
        if self.chunk <= 0 or self.chunk >= N:
            out = self._chunk_forward(h, q, k_node, v_node, gr, gi, bx, knn)
            return h + 0.9*self.norm(out)
        out_h = torch.empty_like(h)
        for i0 in range(0, N, self.chunk):
            i1 = min(i0 + self.chunk, N)
            args = (h[:,i0:i1], q[:,i0:i1], k_node, v_node,
                    gr[:,i0:i1], gi[:,i0:i1], bx[:,i0:i1], knn[:,i0:i1])
            if self.training:
                out_c = checkpoint(self._chunk_forward, *args, use_reentrant=False)
            else:
                out_c = self._chunk_forward(*args)
            out_h[:,i0:i1] = h[:,i0:i1] + 0.9*self.norm(out_c)
        return out_h


# ═══════════════════════════════════════════════════════════════════════════
#  6. FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════

class ISNOperator3D(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, num_heads=4,
                 use_checkpointing=True, equivariant_output=True,
                 use_velocity_input=True, use_re=True,
                 chunk=0, re_max=2000.0, sigma_z=0.1):
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
            QSA_ISNBlock3D(hidden_dim, 3, num_heads, chunk=chunk)
            for _ in range(num_layers)])
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3))
        self.use_checkpointing = use_checkpointing
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
            if self.use_checkpointing and self.training and blk.chunk <= 0:
                h = checkpoint(blk, h, gr, gi, bx, gk_l, om_l, knn, use_reentrant=False)
            else:
                h = blk(h, gr, gi, bx, gk_l, om_l, knn)
        v_local = self.out_proj(h)
        if self.equivariant_output:
            return torch.einsum('bnij,bnj->bni', Fm, v_local)
        return v_local


# ═══════════════════════════════════════════════════════════════════════════
#  7. CONFIG — K=5 fast run
# ═══════════════════════════════════════════════════════════════════════════

G = 32
RE_LIST = [300, 628, 1200]
RE_MAX = max(RE_LIST)

T_FINAL = 10.0
N_SNAPSHOTS = 74
DT = 0.005

K_SCHEDULE = [1, 2, 3, 5]
K_SWITCHES = [0, 15, 35, 60]
K_UNROLL_MAX = K_SCHEDULE[-1]

EPOCHS = 120
BATCH_SIZE = 2
GRAD_ACCUM = 2
LR = 3e-4
CHUNK = 0

LAMBDA_MSE  = 1.0
LAMBDA_E    = 2.0
LAMBDA_Z    = 5.0
LAMBDA_DIV  = 0.05
LAMBDA_SPEC = 0.3


# ═══════════════════════════════════════════════════════════════════════════
#  8. DATA PREP
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(0); np.random.seed(0)
print(f"\n{'='*70}")
print(f" AlienX 3D — K=5 Fast Run")
print(f" G={G} | Re={RE_LIST} | K_max={K_UNROLL_MAX} | T={T_FINAL} | Epochs={EPOCHS}")
print(f" B={BATCH_SIZE}×{GRAD_ACCUM} | snapshots/Re={N_SNAPSHOTS}")
print(f" loss: mse={LAMBDA_MSE} E={LAMBDA_E} Z={LAMBDA_Z} div={LAMBDA_DIV} spec={LAMBDA_SPEC}")
print(f" Drive: {DRIVE_DIR}")
print(f"{'='*70}\n")

print("[1/5] Generating DNS (T=10, ~90 sec per Re)...")
trajs, re_list = [], []
for Re in RE_LIST:
    nu = 2*math.pi / Re
    v = generate_tgv_dataset(G=G, nu=nu, T=T_FINAL, dt=DT,
                             n_samples=N_SNAPSHOTS, device=device)
    if v is not None:
        trajs.append(v); re_list.append(Re)
    print(f"    Re={Re}: {tuple(v.shape) if v is not None else 'FAILED'}")

KX, KY, KZ, K2 = make_wavenumbers(G, device)
x = torch.arange(G, device=device, dtype=torch.float32) * (2*math.pi/G)
X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
coords_single = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).unsqueeze(0)
knn_idx = get_cached_iso_knn_3d(G, device)

print("\n[2/5] Caching physics per Re...")
CACHE = {}
for Re, traj in zip(re_list, trajs):
    td = traj.to(device)
    Us = rms_velocity(td)
    oa = velocity_to_vorticity(td, KX, KY, KZ)
    CACHE[Re] = {'u': td, 'Us': Us, 'omega': oa}
    print(f"  Re={Re}: u={tuple(td.shape)}")

gc.collect(); torch.cuda.empty_cache()

print("\n[3/5] Computing normalization stats...")
kl, gml, gvl, Hl, oml, ul, tl = [],[],[],[],[],[],[]
for Re, traj in zip(re_list, trajs):
    td = traj.to(device); Us = rms_velocity(td)
    oa = velocity_to_vorticity(td, KX, KY, KZ)
    for t in range(traj.shape[0]-1):
        U0 = Us[t].clamp(min=1e-6).to(device)
        om = oa[t:t+1] / U0
        kf = om.norm(dim=-1)
        k_, gm, gv, Hs = scalar_derivatives(kf, KX, KY, KZ)
        kl.append(k_.cpu()); gml.append(gm.cpu()); gvl.append(gv.cpu())
        Hl.append(Hs.cpu()); oml.append(om.reshape(1,-1,3).cpu())
        ul.append((td[t:t+1]/U0).reshape(1,-1,3).cpu())
        tl.append((oa[t+1:t+2]/U0).reshape(1,-1,3).cpu())
    del td, oa
    gc.collect(); torch.cuda.empty_cache()

def mkstat(lst):
    x = torch.cat(lst); return (x.mean().item(), x.std().clamp(min=1e-8).item())

ST = {'k':mkstat(kl),'gm':mkstat(gml),'gv':mkstat(gvl),'H':mkstat(Hl),
      'om':mkstat(oml),'u':mkstat(ul),'tgt':mkstat(tl)}
print(f"  tgt mean={ST['tgt'][0]:.4f} std={ST['tgt'][1]:.4f}")

def _s(x, key): mu, sg = ST[key]; return (x - mu) / sg
def _u(x, key): mu, sg = ST[key]; return x * sg + mu


# ═══════════════════════════════════════════════════════════════════════════
#  9. LOSS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4/5] Building model...")
model = ISNOperator3D(hidden_dim=128, num_layers=4, num_heads=4,
                       use_checkpointing=True, equivariant_output=True,
                       use_velocity_input=True, use_re=True,
                       chunk=CHUNK, re_max=RE_MAX).to(device)
model.G = G
n_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {n_params:,}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
n_opt_steps = (N_SNAPSHOTS * len(re_list)) // BATCH_SIZE
n_opt_steps_per_epoch = max(1, n_opt_steps // GRAD_ACCUM)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=EPOCHS * n_opt_steps_per_epoch)


def sample_batch(bs, k_unroll):
    batch = []
    tries = 0
    while len(batch) < bs and tries < bs * 20:
        tries += 1
        Re = np.random.choice(re_list)
        cache = CACHE[Re]
        max_start = cache['u'].shape[0] - 1 - k_unroll - 1
        if max_start < 1: continue
        t0 = np.random.randint(0, max_start)
        batch.append((Re, t0))
    return batch


def rollout_loss_batched(model, batch, k_unroll):
    if len(batch) == 0:
        return None

    B = len(batch)
    Re_list_b = [Re for Re, _ in batch]

    u_curr = torch.cat([CACHE[Re]['u'][t0:t0+1] for Re, t0 in batch], dim=0)
    U0_vec = torch.stack([CACHE[Re]['Us'][t0].clamp(min=1e-6)
                          for Re, t0 in batch])
    U0 = U0_vec.view(B, 1, 1, 1, 1)

    total_mse = 0.0; total_E = 0.0; total_Z = 0.0; total_D = 0.0; total_S = 0.0
    n_valid = 0

    for step in range(k_unroll):
        omega_curr = velocity_to_vorticity(u_curr, KX, KY, KZ)
        omega_nd = omega_curr / U0
        u_nd = u_curr / U0

        k_field = omega_nd.norm(dim=-1)
        k_flat, gm, gv, Hs = scalar_derivatives(k_field, KX, KY, KZ)

        k_s = _s(k_flat, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
        H_s = _s(Hs, 'H')
        om_s = _s(omega_nd.reshape(B, -1, 3), 'om')
        u_s = _s(u_nd.reshape(B, -1, 3), 'u')

        re_t = torch.tensor([float(r) for r in Re_list_b], device=device)

        pred_omega_std = model(coords_single, k_s, gv_s, gm_s, H_s,
                               om_s, u_s, re_t, knn_idx, G=G)

        targets_std = []
        u_true_list = []
        for i, (Re, t0) in enumerate(batch):
            t_tgt = t0 + step + 1
            U_i = U0[i].view(1, 1, 1, 1)
            om_tgt_nd = CACHE[Re]['omega'][t_tgt:t_tgt+1] / U_i
            targets_std.append(_s(om_tgt_nd.reshape(1, -1, 3), 'tgt'))
            u_true_list.append(CACHE[Re]['u'][t_tgt:t_tgt+1])
        targets_std = torch.cat(targets_std, dim=0)
        u_true = torch.cat(u_true_list, dim=0)

        mse = F.mse_loss(pred_omega_std, targets_std)

        pred_omega_nd = _u(pred_omega_std, 'tgt').reshape(B, G, G, G, 3)
        pred_omega_phys = pred_omega_nd * U0
        u_pred = vorticity_to_velocity(pred_omega_phys, KX, KY, KZ, K2)

        E_pred = kinetic_energy(u_pred);  E_true = kinetic_energy(u_true)
        Z_pred = enstrophy(u_pred, KX, KY, KZ); Z_true = enstrophy(u_true, KX, KY, KZ)

        E_p = E_pred.clamp(min=1e-8); E_t = E_true.clamp(min=1e-8)
        Z_p = Z_pred.clamp(min=1e-8); Z_t = Z_true.clamp(min=1e-8)

        L_E = ((torch.log(E_p) - torch.log(E_t))**2).mean()
        L_Z = ((torch.log(Z_p) - torch.log(Z_t))**2).mean()

        E_k_pred, Z_k_pred = radial_spectra(u_pred, G, KX, KY, KZ, K2)
        E_k_true, Z_k_true = radial_spectra(u_true, G, KX, KY, KZ, K2)
        L_S = ((E_k_pred - E_k_true)**2).mean() / ((E_k_true**2).mean() + 1e-8)
        L_S = L_S + ((Z_k_pred - Z_k_true)**2).mean() / ((Z_k_true**2).mean() + 1e-8)

        div = divergence_spectral(u_pred, KX, KY, KZ)
        L_D = (div**2).mean()

        total_mse = total_mse + mse
        total_E = total_E + L_E
        total_Z = total_Z + L_Z
        total_D = total_D + L_D
        total_S = total_S + L_S
        n_valid += 1

        u_curr = u_pred

    if n_valid == 0:
        return None

    mse_avg = total_mse / n_valid
    E_avg = total_E / n_valid
    Z_avg = total_Z / n_valid
    D_avg = total_D / n_valid
    S_avg = total_S / n_valid

    loss = (LAMBDA_MSE * mse_avg + LAMBDA_E * E_avg +
            LAMBDA_Z * Z_avg + LAMBDA_DIV * D_avg + LAMBDA_SPEC * S_avg)

    return loss, {
        'mse': mse_avg.item(),
        'E': E_avg.item(),
        'Z': Z_avg.item(),
        'D': D_avg.item(),
        'S': S_avg.item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  10. TRAINING LOOP — save best.pt to Drive only
# ═══════════════════════════════════════════════════════════════════════════

def current_K(epoch):
    k = K_SCHEDULE[0]
    for i, sw in enumerate(K_SWITCHES):
        if epoch >= sw:
            k = K_SCHEDULE[min(i, len(K_SCHEDULE)-1)]
    return k


best_val = float('inf')
val_idx_per_re = {Re: N_SNAPSHOTS - 2 for Re in re_list}

# Local best path (temp) + Drive best path
local_best = '/content/alienx_k5_best.pt'
drive_best = os.path.join(DRIVE_DIR, 'alienx_k5_best.pt')
drive_stats = os.path.join(DRIVE_DIR, 'alienx_k5_stats.pkl')

# Save stats + config to Drive immediately
with open(drive_stats, 'wb') as f:
    pickle.dump({'ST': ST, 'RE_LIST': RE_LIST, 'RE_MAX': RE_MAX, 'G': G,
                 'K_SCHEDULE': K_SCHEDULE, 'K_SWITCHES': K_SWITCHES,
                 'T_FINAL': T_FINAL, 'N_SNAPSHOTS': N_SNAPSHOTS,
                 'EPOCHS': EPOCHS, 'LR': LR, 'GRAD_ACCUM': GRAD_ACCUM,
                 'BATCH_SIZE': BATCH_SIZE, 'LAMBDA_MSE': LAMBDA_MSE,
                 'LAMBDA_E': LAMBDA_E, 'LAMBDA_Z': LAMBDA_Z,
                 'LAMBDA_DIV': LAMBDA_DIV, 'LAMBDA_SPEC': LAMBDA_SPEC}, f)
print(f"✓ Stats saved to Drive: {drive_stats}")

print("\n[5/5] Training\n")

for ep in range(1, EPOCHS + 1):
    model.train()
    K_ep = current_K(ep)

    tl, te, tz, td, ts = 0.0, 0.0, 0.0, 0.0, 0.0
    ne = 0
    accum_count = 0
    t_start = time.time()

    opt.zero_grad()
    for _ in range(n_opt_steps):
        batch = sample_batch(BATCH_SIZE, K_ep)
        if len(batch) == 0: continue
        out = rollout_loss_batched(model, batch, K_ep)
        if out is None: continue
        loss, parts = out

        (loss / GRAD_ACCUM).backward()
        accum_count += 1

        tl += parts['mse']; te += parts['E']; tz += parts['Z']
        td += parts['D']; ts += parts['S']
        ne += 1

        if accum_count >= GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step()
            accum_count = 0

        del loss
        torch.cuda.empty_cache()

    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(); sched.step()

    model.eval()
    vl, nv = 0.0, 0
    with torch.no_grad():
        for Re in re_list:
            cache = CACHE[Re]
            td_v = cache['u']; Us_v = cache['Us']; oa_v = cache['omega']
            t = val_idx_per_re[Re]
            U0 = Us_v[t].clamp(min=1e-6)
            om = oa_v[t:t+1] / U0
            kf = om.norm(dim=-1)
            k_, gm, gv, Hs = scalar_derivatives(kf, KX, KY, KZ)
            k_s = _s(k_, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
            H_s = _s(Hs, 'H'); om_s = _s(om.reshape(1,-1,3), 'om')
            u_s = _s((td_v[t:t+1]/U0).reshape(1,-1,3), 'u')
            tgt = _s((oa_v[t+1:t+2]/U0).reshape(1,-1,3), 'tgt')
            re_t = torch.tensor([float(Re)], device=device)
            pred = model(coords_single, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn_idx, G=G)
            vl += F.mse_loss(pred, tgt).item(); nv += 1
    avg_v = vl / max(1, nv)

    if avg_v < best_val:
        best_val = avg_v
        # Save to local first (fast), then copy to Drive
        torch.save(model.state_dict(), local_best)
        try:
            shutil.copy(local_best, drive_best)
            drive_status = "→ Drive"
        except Exception as e:
            drive_status = f"→ Drive FAILED ({e})"

    dt_ep = time.time() - t_start
    if ep % 5 == 0 or ep == 1:
        im_re = model.blocks[-1].last_im_re_ratio
        rel = (avg_v**0.5) / ST['tgt'][1]
        marker = "★" if avg_v == best_val else " "
        print(f"{marker} Ep {ep:03d}/{EPOCHS} | K={K_ep:2d} | ValMSE {avg_v:.6f} | Rel {rel*100:.2f}% | "
              f"[mse={tl/max(1,ne):.4f} E={te/max(1,ne):.4f} Z={tz/max(1,ne):.4f} "
              f"S={ts/max(1,ne):.4f} D={td/max(1,ne):.2e}] | "
              f"im/re {im_re:.3f} | {dt_ep:.0f}s")

print(f"\nBest val MSE: {best_val:.6f}")
print(f"Best Rel RMSE: {(best_val**0.5)/ST['tgt'][1]*100:.2f}%")
print(f"Best checkpoint: {drive_best}")


# ═══════════════════════════════════════════════════════════════════════════
#  11. FINAL ROLLOUT
# ═══════════════════════════════════════════════════════════════════════════

model.load_state_dict(torch.load(drive_best, map_location=device))
model.eval()

print(f"\n{'='*70}\n  FINAL ROLLOUT\n{'='*70}\n")


@torch.no_grad()
def rollout_eval(model, coords, knn, KX, KY, KZ, K2, traj, Re, G, n_steps):
    model.eval()
    omega_traj = velocity_to_vorticity(traj.to(device), KX, KY, KZ)
    u_curr = traj[0:1].to(device).clone()
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

        pred_s = model(coords, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn, G=G)
        om_next_nd = _u(pred_s.reshape(1,-1,3), 'tgt').reshape(1, G, G, G, 3)
        om_next = om_next_nd * U0
        u_next = vorticity_to_velocity(om_next, KX, KY, KZ, K2)

        u_curr = u_next; omega_curr = om_next

        if step < traj.shape[0] - 1:
            om_t = omega_traj[step+1:step+2].to(device)
            u_t = traj[step+1:step+2].to(device)
            mses.append(F.mse_loss(omega_curr, om_t).item())
            Es.append(kinetic_energy(u_curr).item()); gEs.append(kinetic_energy(u_t).item())
            Zs.append(enstrophy(u_curr, KX, KY, KZ).item()); gZs.append(enstrophy(u_t, KX, KY, KZ).item())
            Ds.append(divergence_spectral(u_curr, KX, KY, KZ).abs().mean().item())
        torch.cuda.empty_cache()

    return (np.array(mses), np.array(Es), np.array(gEs),
            np.array(Zs), np.array(gZs), np.array(Ds))


rollout_summary = {}
for Re, traj in zip(re_list, trajs):
    m, E, gE, Z, gZ, D = rollout_eval(model, coords_single, knn_idx,
                                      KX, KY, KZ, K2, traj, Re, G,
                                      min(500, traj.shape[0]-1))
    ed = (E[-1]-gE[-1]) / (abs(gE[-1])+1e-12) * 100
    zd = (Z[-1]-gZ[-1]) / (abs(gZ[-1])+1e-12) * 100
    print(f"  Re={Re}:")
    print(f"    steps={len(m)}  MSE[1]={m[0]:.6f}  MSE[10]={m[min(10,len(m)-1)]:.6f}  MSE[-1]={m[-1]:.6f}")
    print(f"    MSE growth = {m[-1]/m[0]:.2f}×")
    print(f"    Energy drift    = {ed:+.2f}%")
    print(f"    Enstrophy drift = {zd:+.2f}%")
    print(f"    max|∇·u|        = {D.max():.3e}\n")
    rollout_summary[Re] = {'MSE1': float(m[0]), 'MSE_final': float(m[-1]),
                           'energy_drift': float(ed), 'enstrophy_drift': float(zd),
                           'max_div': float(D.max())}


# ═══════════════════════════════════════════════════════════════════════════
#  12. QSA ABLATION
# ═══════════════════════════════════════════════════════════════════════════

print(f"{'='*70}\n  QSA ABLATION\n{'='*70}\n")


@torch.no_grad()
def eval_single_step(model):
    model.eval()
    vl, nv = 0.0, 0
    for Re in re_list:
        cache = CACHE[Re]
        td_v = cache['u']; Us_v = cache['Us']; oa_v = cache['omega']
        t = val_idx_per_re[Re]
        U0 = Us_v[t].clamp(min=1e-6)
        om = oa_v[t:t+1] / U0
        kf = om.norm(dim=-1)
        k_, gm, gv, Hs = scalar_derivatives(kf, KX, KY, KZ)
        k_s = _s(k_, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
        H_s = _s(Hs, 'H'); om_s = _s(om.reshape(1,-1,3), 'om')
        u_s = _s((td_v[t:t+1]/U0).reshape(1,-1,3), 'u')
        tgt = _s((oa_v[t+1:t+2]/U0).reshape(1,-1,3), 'tgt')
        re_t = torch.tensor([float(Re)], device=device)
        pred = model(coords_single, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn_idx, G=G)
        vl += F.mse_loss(pred, tgt).item(); nv += 1
    return vl / max(1, nv)


with torch.no_grad():
    mse_with_qsa = eval_single_step(model)
    saved = [(b.out_proj.weight.clone(), b.out_proj.bias.clone()) for b in model.blocks]
    for b in model.blocks:
        b.out_proj.weight.zero_(); b.out_proj.bias.zero_()
    mse_no_qsa = eval_single_step(model)
    for b, (w, bi) in zip(model.blocks, saved):
        b.out_proj.weight.copy_(w); b.out_proj.bias.copy_(bi)

ratio = mse_no_qsa / mse_with_qsa
print(f"  Val MSE with QSA:    {mse_with_qsa:.6f}")
print(f"  Val MSE without QSA: {mse_no_qsa:.6f}")
print(f"  QSA ablation ratio:  {ratio:.2f}×\n")


# ═══════════════════════════════════════════════════════════════════════════
#  13. EQUIVARIANCE TEST
# ═══════════════════════════════════════════════════════════════════════════

print(f"{'='*70}\n  EQUIVARIANCE TEST\n{'='*70}\n")


def random_rotation(device):
    A = torch.randn(3, 3, device=device)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0: Q[:, 0] *= -1
    return Q


@torch.no_grad()
def equivariance_error(model, n_tests=8):
    model.eval()
    errs, scales = [], []
    for _ in range(n_tests):
        Re = np.random.choice(re_list)
        cache = CACHE[Re]
        td_v = cache['u']; Us_v = cache['Us']; oa_v = cache['omega']
        t = np.random.randint(0, td_v.shape[0] - 2)
        U0 = Us_v[t].clamp(min=1e-6)
        om = oa_v[t:t+1] / U0
        kf = om.norm(dim=-1)
        k_, gm, gv, Hs = scalar_derivatives(kf, KX, KY, KZ)
        k_s = _s(k_, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
        H_s = _s(Hs, 'H'); om_s = _s(om.reshape(1,-1,3), 'om')
        u_s = _s((td_v[t:t+1]/U0).reshape(1,-1,3), 'u')
        re_t = torch.tensor([float(Re)], device=device)

        pred_orig = model(coords_single, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn_idx, G=G)

        R = random_rotation(device)
        coords_rot = coords_single @ R.T
        gv_rot = gv_s @ R.T
        om_rot = om_s @ R.T
        u_rot = u_s @ R.T
        H_rot = torch.einsum('ij,bnjk,kl->bnil', R, H_s, R.T)

        pred_rot = model(coords_rot, k_s, gv_rot, gm_s, H_rot, om_rot, u_rot, re_t, knn_idx, G=G)

        expected = pred_orig @ R.T
        err = (pred_rot - expected).abs().mean().item()
        sc = pred_orig.abs().mean().item()
        errs.append(err / (sc + 1e-12))
        scales.append(sc)
    return float(np.mean(errs)), float(np.mean(scales))


rel_err, scale = equivariance_error(model)
print(f"  Relative equivariance error: {rel_err*100:.4f}%")
print(f"  Mean |pred_orig|:            {scale:.6e}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  14. FINAL SAVE — metrics to Drive
# ═══════════════════════════════════════════════════════════════════════════

metrics = {
    'best_val_mse': best_val,
    'best_rel_rmse': (best_val**0.5) / ST['tgt'][1],
    'qsa_ablation_ratio': ratio,
    'equivariance_rel_error': rel_err,
    'n_params': n_params,
    'G': G, 'RE_LIST': RE_LIST,
    'K_SCHEDULE': K_SCHEDULE, 'K_SWITCHES': K_SWITCHES,
    'T_FINAL': T_FINAL, 'N_SNAPSHOTS': N_SNAPSHOTS,
    'EPOCHS': EPOCHS, 'LR': LR,
    'BATCH_SIZE': BATCH_SIZE, 'GRAD_ACCUM': GRAD_ACCUM,
    'LAMBDA_MSE': LAMBDA_MSE, 'LAMBDA_E': LAMBDA_E,
    'LAMBDA_Z': LAMBDA_Z, 'LAMBDA_DIV': LAMBDA_DIV,
    'LAMBDA_SPEC': LAMBDA_SPEC,
    'rollout': rollout_summary,
}
try:
    np.save(os.path.join(DRIVE_DIR, 'alienx_k5_metrics.npy'), metrics, allow_pickle=True)
    print(f"✓ Metrics saved to Drive: {DRIVE_DIR}/alienx_k5_metrics.npy")
except Exception as e:
    print(f"Metrics save failed: {e}")

print(f"\n{'='*70}\n  FINAL SUMMARY\n{'='*70}")
print(f"  Single-step Rel RMSE:     {(best_val**0.5)/ST['tgt'][1]*100:.2f}%")
print(f"  QSA ablation ratio:       {ratio:.2f}×")
print(f"  Equivariance error:       {rel_err*100:.4f}%")
print(f"  Total params:             {n_params:,}")
print(f"  Reynolds:                 {RE_LIST}")
print(f"  K curriculum:             {K_SCHEDULE}")
print(f"  Epochs:                   {EPOCHS}")
print(f"  Drive checkpoint:         {drive_best}")
print(f"{'='*70}")