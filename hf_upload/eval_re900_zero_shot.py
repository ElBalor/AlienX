# AlienX 3D — zero-shot held-out Reynolds evaluation (HF kit version)
#
# Verifies the harness against the in-distribution Re=300 paper number, then
# evaluates the released checkpoint on Re=900 (never in training):
#   [1] sanity   : reproduce Re=300 val pair (must match within 2x)
#   [2] quick    : Re=900 single-step over all 73 pairs
#   [3] rollout  : Re=900 73-step autoregressive rollout + invariants
#   [4] ablation : gi=0 phase ablation at Re=900 (QSA phase mechanism)
#
# Usage:
#   python eval_re900_zero_shot.py --stage all      # everything (~8 min CPU)
#   python eval_re900_zero_shot.py --stage quick    # sanity + single-step
#
# Expected results (released checkpoint, epoch-55 K=3):
#   sanity Re=300 val pair MSE  ~6.0e-05
#   Re=900 mean rel RMSE        0.53%   (in-distribution: 0.46%)
#   Re=900 rollout MSE growth   5.30x   (628: 2.17x, 1200: 10.86x)
#   Re=900 energy err (max)     4.31%
#   gi=0 phase ablation ratio   4.74x
import os, math, pickle, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

from model import ISNOperator3D, get_isotropic_knn_3d_periodic
import physics as ph

parser = argparse.ArgumentParser()
parser.add_argument('--stage', default='all', choices=['sanity', 'quick', 'rollout', 'ablation', 'all'])
parser.add_argument('--re', type=float, default=900.0, help='held-out Reynolds number')
ARGS = parser.parse_args()

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE, 'alienx_k5_best.pt')
STATS_PATH = os.path.join(BASE, 'alienx_k5_stats.pkl')
OUT_PATH = os.path.join(BASE, 're900_zero_shot_results.npy')
GI_ZERO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

with open(STATS_PATH, 'rb') as f:
    SD = pickle.load(f)
ST = SD['ST']; RE_LIST = SD['RE_LIST']; RE_MAX = SD['RE_MAX']
G = SD['G']; T_FINAL = SD['T_FINAL']; N_SNAPSHOTS = SD['N_SNAPSHOTS']
print(f"G={G}  trained Re={RE_LIST}  held-out Re={ARGS.re}")

def _s(x, key):
    mu, sg = ST[key]; return (x - mu) / sg
def _u(x, key):
    mu, sg = ST[key]; return x * sg + mu

# ── model ────────────────────────────────────────────────────────────────────
print("Loading checkpoint...")
model = ISNOperator3D(hidden_dim=128, num_layers=4, num_heads=4,
                      equivariant_output=True, use_velocity_input=True,
                      use_re=True, re_max=RE_MAX).to(device)
model.G = G
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  OK")

KX, KY, KZ, K2 = ph.make_wavenumbers(G, device)
x = torch.arange(G, device=device, dtype=torch.float32) * (2*math.pi/G)
X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
coords_single = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).unsqueeze(0)
knn_idx = get_isotropic_knn_3d_periodic(G, device, dilation=max(1, G//16))

def build_cache(traj):
    td = traj.to(device)
    return {'u': td,
            'Us': ph.rms_velocity(td),
            'omega': ph.velocity_to_vorticity(td, KX, KY, KZ)}

def gen(Re):
    print(f"  generating DNS Re={Re} (~5 s) ...")
    v = ph.generate_tgv_dataset(G=G, nu=2*math.pi/Re, T=T_FINAL, dt=0.005,
                                n_samples=N_SNAPSHOTS, device=device)
    assert v is not None and v.shape[0] == N_SNAPSHOTS
    return v

@torch.no_grad()
def single_step_pair(cache, t, Re):
    U0 = cache['Us'][t].clamp(min=1e-6)
    om = cache['omega'][t:t+1] / U0
    kf = om.norm(dim=-1)
    k_, gm, gv, Hs = ph.scalar_derivatives(kf, KX, KY, KZ)
    k_s = _s(k_, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
    H_s = _s(Hs, 'H'); om_s = _s(om.reshape(1, -1, 3), 'om')
    u_s = _s((cache['u'][t:t+1]/U0).reshape(1, -1, 3), 'u')
    tgt = _s((cache['omega'][t+1:t+2]/U0).reshape(1, -1, 3), 'tgt')
    re_t = torch.tensor([float(Re)], device=device)
    pred = model(coords_single, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn_idx, G=G)
    return F.mse_loss(pred, tgt).item()

@torch.no_grad()
def single_step_all(cache, Re):
    mses = []
    for t in range(cache['u'].shape[0] - 1):
        mses.append(single_step_pair(cache, t, Re))
    return np.array(mses)

@torch.no_grad()
def rollout_eval(cache, Re, n_steps):
    traj = cache['u']; omega_traj = cache['omega']
    u_curr = traj[0:1].clone(); omega_curr = omega_traj[0:1].clone()
    U0 = ph.rms_velocity(u_curr).clamp(min=1e-6)
    mses, Es, gEs, Zs, gZs, Ds = [], [], [], [], [], []
    for step in range(n_steps):
        om_nd = omega_curr / U0
        kf = om_nd.norm(dim=-1)
        k_, gm, gv, Hs = ph.scalar_derivatives(kf, KX, KY, KZ)
        u_nd = u_curr / U0
        k_s = _s(k_, 'k'); gm_s = _s(gm, 'gm'); gv_s = _s(gv, 'gv')
        H_s = _s(Hs, 'H'); om_s = _s(om_nd.reshape(1, -1, 3), 'om')
        u_s = _s(u_nd.reshape(1, -1, 3), 'u')
        re_t = torch.tensor([float(Re)], device=device)
        pred_s = model(coords_single, k_s, gv_s, gm_s, H_s, om_s, u_s, re_t, knn_idx, G=G)
        om_next = _u(pred_s.reshape(1, -1, 3), 'tgt').reshape(1, G, G, G, 3) * U0
        u_curr = ph.vorticity_to_velocity(om_next, KX, KY, KZ, K2)
        omega_curr = om_next
        if step < traj.shape[0] - 1:
            om_t = omega_traj[step+1:step+2]; u_t = traj[step+1:step+2]
            mses.append(F.mse_loss(omega_curr, om_t).item())
            Es.append(ph.kinetic_energy(u_curr).item()); gEs.append(ph.kinetic_energy(u_t).item())
            Zs.append(ph.enstrophy(u_curr, KX, KY, KZ).item()); gZs.append(ph.enstrophy(u_t, KX, KY, KZ).item())
            Ds.append(ph.divergence_spectral(u_curr, KX, KY, KZ).abs().mean().item())
    return (np.array(mses), np.array(Es), np.array(gEs),
            np.array(Zs), np.array(gZs), np.array(Ds))

def rollout_report(tag, Re, m, E, gE, Z, gZ, D):
    peak_E = max(abs(gE.max()), abs(gE.min()))
    peak_Z = max(abs(gZ.max()), abs(gZ.min()))
    ed = (np.abs(E - gE).max() / peak_E) * 100
    zd = (np.abs(Z - gZ).max() / peak_Z) * 100
    print(f"  {tag} Re={Re}:")
    print(f"    steps={len(m)}  MSE[1]={m[0]:.6f}  MSE[10]={m[min(10, len(m)-1)]:.6f}  MSE[-1]={m[-1]:.6f}")
    print(f"    MSE growth      = {m[-1]/m[0]:.2f}x")
    print(f"    Energy err (max)    = {ed:+.3f}%")
    print(f"    Enstrophy err (max) = {zd:+.3f}%")
    print(f"    max|div u|      = {D.max():.3e}")
    return {'MSE1': float(m[0]), 'MSE10': float(m[min(10, len(m)-1)]),
            'MSE_final': float(m[-1]), 'MSE_growth': float(m[-1]/m[0]),
            'energy_err_max_pct': float(ed), 'enstrophy_err_max_pct': float(zd),
            'max_div': float(D.max())}

results = {'re_held_out': ARGS.re, 're_trained': RE_LIST}

# ── [1] sanity ───────────────────────────────────────────────────────────────
if ARGS.stage in ('sanity', 'quick', 'all'):
    print("\n[1] SANITY: in-distribution Re=300 val pair (t=72->73)")
    cache300 = build_cache(gen(300))
    mse300 = single_step_pair(cache300, 72, 300)
    expected = 6.0323800425976515e-05  # released in-distribution metric
    ratio = mse300 / expected
    ok = (0.5 < ratio < 2.0)
    print(f"  Re=300 t=72->73 MSE = {mse300:.6e}  (paper: {expected:.6e}, ratio {ratio:.3f})")
    print(f"  SANITY {'PASS' if ok else 'FAIL'}")
    results['sanity_re300_mse'] = float(mse300)
    results['sanity_pass'] = bool(ok)
    if not ok:
        print("  Harness broken — do not trust new numbers. Aborting.")
        np.save(OUT_PATH, results, allow_pickle=True)
        raise SystemExit(1)

# ── [2] single-step ──────────────────────────────────────────────────────────
if ARGS.stage in ('quick', 'all'):
    print(f"\n[2] {ARGS.re:g} single-step over all {N_SNAPSHOTS-1} pairs")
    cache900 = build_cache(gen(ARGS.re))
    m_all = single_step_all(cache900, ARGS.re)
    rel_all = np.sqrt(m_all) / ST['tgt'][1]
    print(f"  mean rel RMSE   = {rel_all.mean()*100:.2f}%   (trained Re mean: 0.46%)")
    print(f"  median rel RMSE = {np.median(rel_all)*100:.2f}%")
    print(f"  worst rel RMSE  = {rel_all.max()*100:.2f}%")
    results['single_step'] = {
        'mean_mse': float(m_all.mean()),
        'mean_rel_rmse': float(rel_all.mean()),
        'median_rel_rmse': float(np.median(rel_all)),
        'worst_rel_rmse': float(rel_all.max()),
        'per_step_mse': m_all.tolist(),
    }

# ── [3] rollout ──────────────────────────────────────────────────────────────
if ARGS.stage in ('rollout', 'all'):
    print(f"\n[3] {ARGS.re:g} rollout ({N_SNAPSHOTS-1} steps, full T={T_FINAL})")
    cache900 = build_cache(gen(ARGS.re))
    m, E, gE, Z, gZ, D = rollout_eval(cache900, ARGS.re, traj_len := N_SNAPSHOTS - 1)
    results['rollout'] = rollout_report("HELD-OUT", ARGS.re, m, E, gE, Z, gZ, D)

# ── [4] gi=0 phase ablation ──────────────────────────────────────────────────
if ARGS.stage in ('ablation', 'all'):
    print(f"\n[4] gi=0 PHASE ABLATION at Re={ARGS.re:g} (attention intact, phase killed)")
    import model as M
    cache900 = build_cache(gen(ARGS.re))
    m_all = single_step_all(cache900, ARGS.re)
    # zero the imaginary geometry channel (attention intact)
    orig_forward = M.precompute_geometry
    def gi_zeroed(*a, **kw):
        gr, gi, bx = orig_forward(*a, **kw)
        return gr, torch.zeros_like(gi), bx
    M.precompute_geometry = gi_zeroed
    m_no_phase = single_step_all(cache900, ARGS.re)
    M.precompute_geometry = orig_forward
    ratio_phase = m_no_phase.mean() / m_all.mean()
    print(f"  MSE with phase (gi):   {m_all.mean():.6f}")
    print(f"  MSE without phase:     {m_no_phase.mean():.6f}")
    print(f"  phase ablation ratio:  {ratio_phase:.2f}x")
    results['gi_ablation_ratio'] = float(ratio_phase)
    results['gi0_mean_mse'] = float(m_no_phase.mean())
    results['gi_mean_mse'] = float(m_all.mean())

np.save(OUT_PATH, results, allow_pickle=True)
print(f"\nSaved: {OUT_PATH}")
print(f"\n{'='*60}\n  SUMMARY (Re={ARGS.re:g}, trained on {RE_LIST})\n{'='*60}")
if 'sanity_pass' in results:
    print(f"  sanity Re=300 reproduced:  {'PASS' if results['sanity_pass'] else 'FAIL'}")
if 'single_step' in results:
    print(f"  single-step mean rel RMSE: {results['single_step']['mean_rel_rmse']*100:.2f}%  (trained: 0.46%)")
if 'rollout' in results:
    print(f"  rollout MSE growth:        {results['rollout']['MSE_growth']:.2f}x")
    print(f"  rollout energy err (max):  {results['rollout']['energy_err_max_pct']:.3f}%")
if 'gi_ablation_ratio' in results:
    print(f"  gi=0 phase ablation:       {results['gi_ablation_ratio']:.2f}x")
print(f"{'='*60}")
