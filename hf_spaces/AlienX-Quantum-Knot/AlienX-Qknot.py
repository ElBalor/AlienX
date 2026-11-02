import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import plotly.graph_objects as go
import math

# Enforce exact float32 precision for topological SO(3) frames
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
KNOT_EPOCHS = int(os.environ.get("ALIENX_KNOT_EPOCHS", "30"))
KNOT_PARTICLES = int(os.environ.get("ALIENX_KNOT_PARTICLES", "4096"))
KNOT_TIME_STEPS = int(os.environ.get("ALIENX_KNOT_TIME_STEPS", "6"))

# ═══════════════════════════════════════════════════════════════════
# 1. THE TREFOIL QUANTUM VORTEX GENERATOR
# ═══════════════════════════════════════════════════════════════════

def _trefoil_curve(device, n=400):
    t = torch.linspace(0, 2*math.pi, n, device=device)
    return torch.stack([torch.sin(t) + 2*torch.sin(2*t),
                        torch.cos(t) - 2*torch.cos(2*t),
                        -torch.sin(3*t)], dim=-1) * 0.8

def _density_from_coords(coords, curve):
    """Evaluate the trefoil tube field at arbitrary (possibly perturbed) points."""
    chunks = []
    for i in range(0, coords.shape[0], 1024):
        chunks.append(torch.cdist(coords[i:i+1024], curve).amin(dim=-1))
    return torch.tanh(torch.cat(chunks) / 0.5)

def generate_trefoil_knot(n_particles=4096, grid_size=32, box_size=6.0,
                          perturb=0.0, seed=0):
    cpu_device = torch.device('cpu')
    
    gen = torch.Generator(device=cpu_device).manual_seed(int(seed))
    curve = _trefoil_curve(cpu_device)
    
    x = torch.linspace(-box_size/2, box_size/2, grid_size, device=cpu_device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    
    rho = _density_from_coords(grid_points, curve).reshape(grid_size, grid_size, grid_size)
    theta = torch.atan2(Y, X)
    
    prob = (1.0 - rho).clamp(min=0.01).flatten()
    prob = prob / prob.sum()
    idx = torch.multinomial(prob, n_particles, replacement=True, generator=gen)
    
    coords = grid_points[idx]
    if perturb:
        noise = torch.randn(coords.shape, generator=gen, device=cpu_device) * float(perturb) * 0.5
        coords = coords + noise
    densities = _density_from_coords(coords, curve).unsqueeze(-1)
    phases = theta.flatten()[idx].unsqueeze(-1)
    phases = torch.atan2(coords[:, 1], coords[:, 0]).unsqueeze(-1)
    target_density = densities * torch.exp(-0.2 * torch.sum(coords**2, dim=-1, keepdim=True))
    
    return coords.unsqueeze(0), densities.unsqueeze(0), target_density.unsqueeze(0), phases.unsqueeze(0)

def apply_fixed_perturbation(base, noise, perturbation_strength, curve):
    """Apply a deterministic perturbation to one cached specimen."""
    coords0, _, _, _ = base
    coords = coords0[0] + noise * float(perturbation_strength) * 0.5
    densities = _density_from_coords(coords, curve).unsqueeze(-1)
    phases = torch.atan2(coords[:, 1], coords[:, 0]).unsqueeze(-1)
    target = densities * torch.exp(-0.2 * coords.square().sum(-1, keepdim=True))
    return coords.unsqueeze(0), densities.unsqueeze(0), target.unsqueeze(0), phases.unsqueeze(0)

def apply_temporal_perturbation(base, noise, perturbation_strength, time_value, curve):
    """Generate one frame on a deterministic perturbation trajectory t∈[0,1]."""
    coords0, _, _, _ = base
    coords = coords0[0] + noise * float(perturbation_strength) * float(time_value) * 0.5
    densities = _density_from_coords(coords, curve).unsqueeze(-1)
    phases = torch.atan2(coords[:, 1], coords[:, 0]).unsqueeze(-1)
    target = densities * torch.exp(-0.2 * coords.square().sum(-1, keepdim=True))
    return coords.unsqueeze(0), densities.unsqueeze(0), target.unsqueeze(0), phases.unsqueeze(0)

# ═══════════════════════════════════════════════════════════════════
# 2. CHUNKED kNN ATTENTION
# ═══════════════════════════════════════════════════════════════════

def knn_chunked(coords, k=32, chunk=512):
    B, N, _ = coords.shape
    knn_idx = torch.zeros(B, N, k, dtype=torch.long, device=coords.device)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        dist = torch.cdist(coords[:, i:j], coords)
        _, idx = dist.topk(k, dim=-1, largest=False)
        knn_idx[:, i:j] = idx
    return knn_idx

# ═══════════════════════════════════════════════════════════════════
# 3. SO(3) FRAME & QSA
# ═══════════════════════════════════════════════════════════════════

class SO3_FrameBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('pert_mat', 1e-3 * torch.eye(3).unsqueeze(0))

    def forward(self, coords, density, knn_idx):
        B, N, _ = coords.shape
        grad_proxy = coords - coords.mean(dim=1, keepdim=True)
        mag = grad_proxy.norm(dim=-1, keepdim=True) + 1e-8
        n_grad = grad_proxy / mag

        w = torch.softmax(density.squeeze(-1), dim=1).unsqueeze(-1)
        S = torch.einsum('bnd,bnc->bdc', coords * w, coords)
        S = S / (S.norm(dim=(-2, -1), keepdim=True) + 1e-8) + self.pert_mat
        _, evec = torch.linalg.eigh(S)
        fb = evec[..., -1].unsqueeze(1).expand(-1, N, -1)

        gate = torch.sigmoid((mag - 1e-4) / 1e-3)
        n = F.normalize(gate * n_grad + (1 - gate) * fb, dim=-1)

        evec_exp = evec.unsqueeze(1).expand(-1, N, -1, -1)
        dots = torch.einsum('bnij,bni->bnj', evec_exp, n).abs()
        best_idx = dots.argmin(dim=-1)
        ref = evec_exp.gather(2, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 1)).squeeze(-1)
        
        u1 = F.normalize(torch.cross(n, ref, dim=-1), dim=-1)
        u2 = torch.cross(n, u1, dim=-1)
        
        # Local neighbor covariance: this is a genuine curvature/tangent
        # estimate, rather than the rank-1 global outer product of position.
        b_idx = torch.arange(B, device=coords.device).view(B, 1, 1).expand(B, N, knn_idx.shape[-1])
        nbr = coords[b_idx, knn_idx]
        rel = nbr - coords.unsqueeze(2)
        scale = rel.square().sum(-1).mean(dim=-1, keepdim=True).clamp_min(1e-6)
        w = torch.exp(-rel.square().sum(-1) / scale).unsqueeze(-1)
        H = (rel.unsqueeze(-1) * rel.unsqueeze(-2) * w.unsqueeze(-1)).sum(2) / (w.sum(2).unsqueeze(-1) + 1e-8)
        
        Hu1 = torch.einsum('bnij,bnj->bni', H, u1)
        Hu2 = torch.einsum('bnij,bnj->bni', H, u2)
        a = (u1 * Hu1).sum(-1)
        b = (u1 * Hu2).sum(-1)
        c = (u2 * Hu2).sum(-1)
        th = 0.5 * torch.atan2(2*b, a - c + 1e-8)
        e1 = torch.cos(th).unsqueeze(-1) * u1 + torch.sin(th).unsqueeze(-1) * u2

        Hn = torch.einsum('bnij,bnj->bni', H, grad_proxy)
        Ht = Hn - (Hn * n).sum(-1, keepdim=True) * n
        sgn = torch.sign((e1 * Ht).sum(-1, keepdim=True))
        sgn = torch.where(Ht.norm(dim=-1, keepdim=True) > 1e-6, sgn, torch.ones_like(sgn))
        e1 = F.normalize(e1 * sgn, dim=-1)
        e2 = torch.cross(n, e1, dim=-1)
        
        return e1, e2, n

class QSA_TopologicalBlock(nn.Module):
    def __init__(self, dim, heads=4, k=32):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.k = k
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, coords, e1, e2, n, knn_idx):
        B, N, C = h.shape
        H, D, K = self.heads, self.head_dim, self.k
        q = self.q_proj(h).view(B, N, H, D)
        k_all = self.k_proj(h).view(B, N, H, D)
        v_all = self.v_proj(h).view(B, N, H, D)

        b_idx = torch.arange(B, device=h.device).view(B, 1, 1).expand(B, N, K)
        k_nbr = k_all[b_idx, knn_idx]
        v_nbr = v_all[b_idx, knn_idx]
        c_nbr = coords[b_idx, knn_idx]

        rel = c_nbr - coords.unsqueeze(2)
        dx = (rel * e1.unsqueeze(2)).sum(-1)
        dy = (rel * e2.unsqueeze(2)).sum(-1)
        dz = (rel * n.unsqueeze(2)).sum(-1)

        theta = torch.atan2(dy, dx)
        w_off = torch.exp(-dz**2 / 0.1)
        g_ij = w_off * (torch.cos(theta) + 1j * torch.sin(theta))

        q_c = torch.complex(q, torch.zeros_like(q))
        k_c = torch.complex(k_nbr, torch.zeros_like(k_nbr))
        S = (q_c.unsqueeze(2) * k_c.conj()).sum(-1) * g_ij.unsqueeze(-1)

        norm = torch.sqrt((S.abs()**2).sum(dim=2, keepdim=True) + 1e-8)
        alpha = S / norm

        v_c = torch.complex(v_nbr, torch.zeros_like(v_nbr))
        out = (alpha.unsqueeze(-1) * v_c).sum(2).real.reshape(B, N, C)
        return h + 0.9 * self.norm(self.out_proj(out))

class AlienX_KnotsEngine(nn.Module):
    def __init__(self, hidden_dim=64, num_blocks=2):
        super().__init__()
        self.frame_builder = SO3_FrameBuilder()
        # coords(3), density(1), perturbation(1), time(1), phase cos/sin(2)
        self.in_proj = nn.Linear(8, hidden_dim)
        self.blocks = nn.ModuleList([QSA_TopologicalBlock(hidden_dim) for _ in range(num_blocks)])
        # complex density: real and imaginary channels make phase supervision
        # observable instead of an inert scalar penalty.
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, coords, density, perturbation=0.0, time_value=0.0,
                phases=None, knn_idx=None):
        B, N, _ = coords.shape
        if knn_idx is None:
            knn_idx = knn_chunked(coords, k=32)

        e1, e2, n = self.frame_builder(coords, density, knn_idx)
        if phases is None:
            phases = torch.atan2(coords[..., 1:2], coords[..., 0:1])
        if not torch.is_tensor(perturbation):
            perturbation = torch.full((B, N, 1), float(perturbation), device=coords.device, dtype=coords.dtype)
        elif perturbation.ndim == 0:
            perturbation = torch.full((B, N, 1), float(perturbation), device=coords.device, dtype=coords.dtype)
        elif perturbation.ndim == 1:
            perturbation = perturbation.view(B, 1, 1).expand(B, N, 1)
        else:
            perturbation = perturbation.expand(B, N, 1)
        if not torch.is_tensor(time_value):
            time_value = torch.full((B, N, 1), float(time_value), device=coords.device, dtype=coords.dtype)
        elif time_value.ndim == 0:
            time_value = torch.full((B, N, 1), float(time_value), device=coords.device, dtype=coords.dtype)
        elif time_value.ndim == 1:
            time_value = time_value.view(B, 1, 1).expand(B, N, 1)
        else:
            time_value = time_value.expand(B, N, 1)
        h = self.in_proj(torch.cat([coords, density, perturbation, time_value,
                                    torch.cos(phases), torch.sin(phases)], dim=-1))

        for blk in self.blocks:
            h = blk(h, coords, e1, e2, n, knn_idx)
        return self.out_proj(h)

# ═══════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP WITH LIVE LOGS
# ═══════════════════════════════════════════════════════════════════

def topological_loss(pred, true, phases):
    """Complex-density reconstruction with an active phase gradient."""
    target_complex = torch.cat([true * torch.cos(phases), true * torch.sin(phases)], dim=-1)
    l2 = F.mse_loss(pred, target_complex)
    pred_amp = pred.square().sum(-1, keepdim=True).sqrt()
    core_penalty = torch.mean(torch.abs(pred_amp[true < 0.2]))
    pred_phase = torch.atan2(pred[..., 1:2], pred[..., 0:1])
    phase_weight = true.detach().clamp_min(0.05)
    phase_pen = ((1.0 - torch.cos(pred_phase - phases)) * phase_weight).mean()
    return l2 + 0.5 * core_penalty + 0.02 * phase_pen

print("[Train] Generating perturbed Trefoil Knot curriculum...")
model = AlienX_KnotsEngine().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

print(f"[Train] Starting {KNOT_EPOCHS} epochs with random perturbation conditioning...")
loss_history = []
for epoch in range(KNOT_EPOCHS):
    strength = float(torch.rand((), device='cpu') * 0.9)
    time_value = float(torch.randint(0, KNOT_TIME_STEPS, (), device='cpu') / max(1, KNOT_TIME_STEPS - 1))
    coords, density, target, phases = generate_trefoil_knot(
        n_particles=KNOT_PARTICLES, grid_size=32, perturb=strength, seed=1000 + epoch)
    coords, density, target, phases = (x.to(device) for x in (coords, density, target, phases))
    optimizer.zero_grad()
    pred = model(coords, density, perturbation=strength, time_value=time_value, phases=phases)
    loss = topological_loss(pred, target, phases)
    loss.backward()
    optimizer.step()
    loss_history.append({"epoch": epoch, "loss": float(loss.item()),
                         "perturbation": strength, "time": time_value})
    if epoch % 5 == 0:
        print(f"  -> Epoch {epoch:02d} | Loss: {loss.item():.5f} | perturb={strength:.3f}")

if KNOT_EPOCHS > 0:
    torch.save({"model": model.state_dict(), "loss_history": loss_history,
                "architecture": "AlienX_KnotsEngine complex-density v2",
                "epochs": KNOT_EPOCHS}, "alienx_quantum_knot_checkpoint.pt")
    np.save("alienx_quantum_knot_loss.npy", np.asarray([x["loss"] for x in loss_history], dtype=np.float32))
        
print("[Train] Training complete. Launching Gradio app...")

# Fixed inference specimen and fixed perturbation basis: moving the slider now
# changes one physical perturbation amplitude, rather than resampling a new
# particle cloud on every click.
INFERENCE_BASE = generate_trefoil_knot(n_particles=KNOT_PARTICLES, grid_size=32, seed=4242)
INFERENCE_CURVE = _trefoil_curve(torch.device("cpu"))
INFERENCE_NOISE = torch.randn(INFERENCE_BASE[0][0].shape, generator=torch.Generator().manual_seed(4243))

# ═══════════════════════════════════════════════════════════════════
# 5. GRADIO APP (Fixed for Gradio 6.0)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_reconnection(perturbation_strength):
    coords_cpu, density_cpu, _, phases_cpu = apply_fixed_perturbation(
        INFERENCE_BASE, INFERENCE_NOISE, perturbation_strength, INFERENCE_CURVE)
    coords_gpu = coords_cpu.to(device)
    density_gpu = density_cpu.to(device)
    phases_gpu = phases_cpu.to(device)
    pred_complex = model(coords_gpu, density_gpu, perturbation=perturbation_strength,
                         time_value=1.0, phases=phases_gpu)
    
    c = coords_cpu[0].cpu().numpy()
    d = pred_complex[0].square().sum(-1).sqrt().cpu().numpy()
    
    fig = go.Figure(data=[go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode='markers',
        marker=dict(
            size=2.5,
            color=d,
            colorscale='Plasma',
            colorbar=dict(title="Predicted Density", thickness=15),
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=f"AlienX-Knots: Trefoil Reconnection (Perturbation={perturbation_strength:.2f})",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            bgcolor='rgb(5, 5, 10)'
        ),
        paper_bgcolor='rgb(5, 5, 10)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

with gr.Blocks(title="AlienX-Quantum-Knots") as demo:
    gr.Markdown(
        """
        # 🕸️ AlienX-Quantum-Knots
        ### First Real-Time SO(3) Topological Vortex Reconnection on a Single T4
        
        Predicting the density cascade of a **trefoil knot** singularity after an external perturbation.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            perturb_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                label="Reconnection Perturbation Strength"
            )
            run_btn = gr.Button("Run AlienX-Knots", variant="primary")
        with gr.Column(scale=3):
            plot_output = gr.Plot(label="3D Quantum Vortex")
            
    run_btn.click(fn=predict_reconnection, inputs=perturb_slider, outputs=plot_output)
    demo.load(fn=predict_reconnection, inputs=perturb_slider, outputs=plot_output)

if os.environ.get("ALIENX_KNOT_NO_LAUNCH", "0") != "1":
    demo.launch(theme=gr.themes.Monochrome(), share=os.environ.get("ALIENX_KNOT_SHARE", "0") == "1")
