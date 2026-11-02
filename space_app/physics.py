# AlienX 3D — spectral physics utilities (Taylor–Green vortex)
# Extracted verbatim from the released training/eval code.
import math
import torch
import torch.nn.functional as F


def make_wavenumbers(G, device):
    kx = 2*math.pi*torch.fft.fftfreq(G, d=1.0/G, device=device)
    ky = 2*math.pi*torch.fft.fftfreq(G, d=1.0/G, device=device)
    kz = 2*math.pi*torch.fft.fftfreq(G, d=1.0/G, device=device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    return KX, KY, KZ, K2


def projection(u_hat, v_hat, w_hat, KX, KY, KZ, K2):
    kd = KX*u_hat + KY*v_hat + KZ*w_hat
    return (u_hat - KX*kd/K2, v_hat - KY*kd/K2, w_hat - KZ*kd/K2)


def nonlinear_term(u_hat, v_hat, w_hat, KX, KY, KZ, K2):
    u = torch.fft.ifftn(u_hat, dim=(-3, -2, -1)).real
    v = torch.fft.ifftn(v_hat, dim=(-3, -2, -1)).real
    w = torch.fft.ifftn(w_hat, dim=(-3, -2, -1)).real
    du_dx = torch.fft.ifftn(1j*KX*u_hat, dim=(-3, -2, -1)).real
    du_dy = torch.fft.ifftn(1j*KY*u_hat, dim=(-3, -2, -1)).real
    du_dz = torch.fft.ifftn(1j*KZ*u_hat, dim=(-3, -2, -1)).real
    dv_dx = torch.fft.ifftn(1j*KX*v_hat, dim=(-3, -2, -1)).real
    dv_dy = torch.fft.ifftn(1j*KY*v_hat, dim=(-3, -2, -1)).real
    dv_dz = torch.fft.ifftn(1j*KZ*v_hat, dim=(-3, -2, -1)).real
    dw_dx = torch.fft.ifftn(1j*KX*w_hat, dim=(-3, -2, -1)).real
    dw_dy = torch.fft.ifftn(1j*KY*w_hat, dim=(-3, -2, -1)).real
    dw_dz = torch.fft.ifftn(1j*KZ*w_hat, dim=(-3, -2, -1)).real
    ax = u*du_dx + v*du_dy + w*du_dz
    ay = u*dv_dx + v*dv_dy + w*dv_dz
    az = u*dw_dx + v*dw_dy + w*dw_dz
    ax_hat = torch.fft.fftn(ax, dim=(-3, -2, -1))
    ay_hat = torch.fft.fftn(ay, dim=(-3, -2, -1))
    az_hat = torch.fft.fftn(az, dim=(-3, -2, -1))
    G = u_hat.shape[-1]
    kmax = G // 2
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


def generate_tgv_dataset(G=32, nu=0.01, T=10.0, dt=0.005, n_samples=74, device='cpu'):
    """Pseudo-spectral DNS of the Taylor-Green vortex.
    IMEX-Euler, Orszag 2/3 de-aliasing, Leray projection every step.
    Returns [n_samples, G, G, G, 3] velocity snapshots."""
    x = torch.arange(G, device=device, dtype=torch.float32) * (2*math.pi/G)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    u = torch.sin(X)*torch.cos(Y)*torch.cos(Z)
    v = -torch.cos(X)*torch.sin(Y)*torch.cos(Z)
    w = torch.zeros_like(u)
    u_hat = torch.fft.fftn(u, dim=(-3, -2, -1))
    v_hat = torch.fft.fftn(v, dim=(-3, -2, -1))
    w_hat = torch.fft.fftn(w, dim=(-3, -2, -1))
    KX, KY, KZ, K2 = make_wavenumbers(G, device)
    snaps = []
    total = int(T/dt)
    save_every = max(1, total//n_samples)
    for step in range(total):
        u_hat, v_hat, w_hat = tgv_imex_step(u_hat, v_hat, w_hat, KX, KY, KZ, K2, nu, dt)
        u_hat, v_hat, w_hat = projection(u_hat, v_hat, w_hat, KX, KY, KZ, K2)
        if torch.isnan(u_hat).any():
            return None
        if step % save_every == 0 and len(snaps) < n_samples:
            snaps.append(torch.stack([
                torch.fft.ifftn(u_hat, dim=(-3, -2, -1)).real.cpu(),
                torch.fft.ifftn(v_hat, dim=(-3, -2, -1)).real.cpu(),
                torch.fft.ifftn(w_hat, dim=(-3, -2, -1)).real.cpu(),
            ], dim=-1))
    return torch.stack(snaps, dim=0)


def velocity_to_vorticity(v, KX, KY, KZ):
    u = torch.fft.fftn(v[..., 0], dim=(-3, -2, -1))
    vv = torch.fft.fftn(v[..., 1], dim=(-3, -2, -1))
    w = torch.fft.fftn(v[..., 2], dim=(-3, -2, -1))
    wx = torch.fft.ifftn(1j*(KY*w - KZ*vv), dim=(-3, -2, -1)).real
    wy = torch.fft.ifftn(1j*(KZ*u - KX*w), dim=(-3, -2, -1)).real
    wz = torch.fft.ifftn(1j*(KX*vv - KY*u), dim=(-3, -2, -1)).real
    return torch.stack([wx, wy, wz], dim=-1)


def vorticity_to_velocity(omega, KX, KY, KZ, K2):
    ox = torch.fft.fftn(omega[..., 0], dim=(-3, -2, -1))
    oy = torch.fft.fftn(omega[..., 1], dim=(-3, -2, -1))
    oz = torch.fft.fftn(omega[..., 2], dim=(-3, -2, -1))
    ux_hat = 1j*(KY*oz - KZ*oy)/K2
    uy_hat = 1j*(KZ*ox - KX*oz)/K2
    uz_hat = 1j*(KX*oy - KY*ox)/K2
    ux_hat[..., 0, 0, 0] = 0
    uy_hat[..., 0, 0, 0] = 0
    uz_hat[..., 0, 0, 0] = 0
    return torch.stack([
        torch.fft.ifftn(ux_hat, dim=(-3, -2, -1)).real,
        torch.fft.ifftn(uy_hat, dim=(-3, -2, -1)).real,
        torch.fft.ifftn(uz_hat, dim=(-3, -2, -1)).real,
    ], dim=-1)


def scalar_derivatives(k_grid, KX, KY, KZ):
    """One FFT call -> gradient + 6 Hessian components of the scalar field."""
    B, G_, _, _ = k_grid.shape
    k_hat = torch.fft.fftn(k_grid, dim=(-3, -2, -1))
    ops = torch.stack([1j*KX, 1j*KY, 1j*KZ, -(KX**2), -(KY**2), -(KZ**2),
                       -(KX*KY), -(KX*KZ), -(KY*KZ)], dim=0)
    d = torch.fft.ifftn(ops.unsqueeze(1)*k_hat.unsqueeze(0), dim=(-3, -2, -1)).real
    d = d.permute(1, 2, 3, 4, 0).reshape(B, -1, 9)
    gv = d[..., :3]
    gm = gv.norm(dim=-1)
    k_flat = k_grid.reshape(B, -1)
    H_xx, H_yy, H_zz, H_xy, H_xz, H_yz = d[..., 3:].unbind(-1)
    row0 = torch.stack([H_xx, H_xy, H_xz], dim=-1)
    row1 = torch.stack([H_xy, H_yy, H_yz], dim=-1)
    row2 = torch.stack([H_xz, H_yz, H_zz], dim=-1)
    H = torch.stack([row0, row1, row2], dim=-2)
    return k_flat, gm, gv, H


def kinetic_energy(v):
    return 0.5 * (v**2).sum(dim=-1).mean(dim=(-3, -2, -1))


def enstrophy(v, KX, KY, KZ):
    om = velocity_to_vorticity(v, KX, KY, KZ)
    return 0.5 * (om**2).sum(dim=-1).mean(dim=(-3, -2, -1))


def divergence_spectral(v, KX, KY, KZ):
    u = torch.fft.fftn(v[..., 0], dim=(-3, -2, -1))
    vv = torch.fft.fftn(v[..., 1], dim=(-3, -2, -1))
    w = torch.fft.fftn(v[..., 2], dim=(-3, -2, -1))
    return torch.fft.ifftn(1j*(KX*u + KY*vv + KZ*w), dim=(-3, -2, -1)).real


def rms_velocity(v):
    return torch.sqrt((v**2).sum(dim=-1).mean(dim=(1, 2, 3)))
