# AlienX 3D — model architecture (ISNOperator3D)
# Extracted verbatim from the released training/eval code (Colab K=5 run, v2 final test).
# Loads alienx_k5_best.pt directly: ISNOperator3D(hidden_dim=128, num_layers=4,
# num_heads=4, re_max=1200.0); model.G = 32.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Isotropic stencil: all offsets with r^2 <= 4 in the 5x5x5 cube minus center.
# 26 neighbors, rotation-uniform by construction.
# ─────────────────────────────────────────────────────────────────────────────
ISO_OFFSETS_3D = []
for dx_ in range(-2, 3):
    for dy_ in range(-2, 3):
        for dz_ in range(-2, 3):
            if dx_ == 0 and dy_ == 0 and dz_ == 0:
                continue
            if dx_*dx_ + dy_*dy_ + dz_*dz_ <= 4:
                ISO_OFFSETS_3D.append((dx_, dy_, dz_))
K_ISO_3D = len(ISO_OFFSETS_3D)  # 26


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


# ─────────────────────────────────────────────────────────────────────────────
# Frame builder: SO(3) frame from grad/Hessian of the scalar field,
# inertia-tensor fallback (isotropic 1e-3*I perturbation preserves
# equivariance), Hessian-based in-plane orientation, sign lock.
# ─────────────────────────────────────────────────────────────────────────────
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
                    if j > 0:
                        S_try = 0.5*(S_try + S_try.transpose(-1, -2))
                    _, ev = torch.linalg.eigh(S_try)
                    evec = ev
                    break
                except RuntimeError:
                    continue
            if evec is None:
                evec = torch.eye(3, device=S.device).unsqueeze(0).expand(B, -1, -1)

            fb = evec[..., -1].unsqueeze(1).expand(-1, N, -1)
            gate = torch.sigmoid((mag - self.epsilon_grad) / self.tau)
            n = F.normalize(gate*n_grad + (1-gate)*fb, dim=-1)

            # Tangent reference: inertia eigenvector least aligned with n
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
            conf = torch.sigmoid((disc - 1e-3) / 1e-4)

        return e1, e2, n, conf


def precompute_geometry(coords, e1, e2, n, pc, knn_idx, sigma_z, G):
    """Stencil offsets give exact small-displacement vectors (no min-image wrap).
    Equivariant by construction."""
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
        wo = torch.exp(-(dzl**2) / (sigma_z**2))
        gr = wo * torch.cos(th)
        gi = wo * torch.sin(th) * pc.unsqueeze(-1)
    return gr, gi, bx


# ─────────────────────────────────────────────────────────────────────────────
# QSA block: complex projective attention with geometric phase.
# S = q·conj(k), modulated by (gr + i·gi), ℓ2-normalized over the neighbor
# axis (preserves phase — no softmax), coherent complex superposition.
# ─────────────────────────────────────────────────────────────────────────────
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
        out_c = (alpha.unsqueeze(-1) * V_j).sum(dim=2)
        with torch.no_grad():
            rm = out_c.real.abs().mean().item()
            im = out_c.imag.abs().mean().item()
            self.last_im_re_ratio = im / (rm + 1e-8)
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
            return h + 0.9 * self.norm(out)
        out_h = torch.empty_like(h)
        for i0 in range(0, N, self.chunk):
            i1 = min(i0 + self.chunk, N)
            args = (h[:, i0:i1], q[:, i0:i1], k_node, v_node,
                    gr[:, i0:i1], gi[:, i0:i1], bx[:, i0:i1], knn[:, i0:i1])
            out_c = self._chunk_forward(*args)
            out_h[:, i0:i1] = h[:, i0:i1] + 0.9 * self.norm(out_c)
        return out_h


# ─────────────────────────────────────────────────────────────────────────────
# Full operator
# ─────────────────────────────────────────────────────────────────────────────
class ISNOperator3D(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, num_heads=4,
                 equivariant_output=True, use_velocity_input=True, use_re=True,
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
