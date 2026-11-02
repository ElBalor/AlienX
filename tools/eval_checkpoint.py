"""Evaluate an AlienX MHD checkpoint without changing or resuming training.

This deliberately reuses the canonical implementation.  It reports the same
in-distribution, hard-regime, ablation, rollout, and gate evidence that the
training script writes after a completed run, while labeling a partial
checkpoint honestly in the output metadata.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "hf_upload_mhd" / "AlienX-MHD-65ep-v3" / "training" / "latest_resume" / "run"
OUT = ROOT / "hf_upload_mhd" / "AlienX-MHD-65ep-v3" / "evaluation" / "partial_checkpoint"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("ALIENX_OUT_DIR", str(OUT))
os.environ.setdefault("ALIENX_RESUME", "0")
os.environ.setdefault("ALIENX_ARCHIVE", "0")
# A GTX 1650 cannot hold the full 736-trajectory protocol in host RAM
# (the canonical run reports ~6.47 GiB before Python overhead).  The local
# pass is therefore a smoke/diagnostic evaluation; the grant benchmark remains
# the full Kaggle run with 32/8/32 samples per regime.
os.environ.setdefault("ALIENX_TRAIN_PER_REGIME", "1")
os.environ.setdefault("ALIENX_VAL_PER_REGIME", "1")
os.environ.setdefault("ALIENX_HELDOUT", "2")

spec = importlib.util.spec_from_file_location("alienx_mhd", ROOT / "AlienX-mhd.py")
ax = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ax)

import torch
import numpy as np

def main() -> None:
    ckpt = RUN / "last_training.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ax.AlienX_MHD(hidden=ax.HIDDEN, blocks=ax.LAYERS, heads=ax.HEADS,
                          chunk=ax.CHUNK, re_max=max(ax.RE_VALUES),
                          rm_max=max(ax.RM_VALUES), sigma_z=ax.SIGMA_Z).to(ax.DEVICE)
    model.load_state_dict(state["model"])
    train, val, held = ax.build_datasets()
    abl = ax.run_ablations(model, val, held)
    rollout_rows = [ax.rollout_eval(model, sample) for sample in held]
    u = np.asarray([x[0] for x in rollout_rows], dtype=np.float64)
    b = np.asarray([x[1] for x in rollout_rows], dtype=np.float64)
    rollout = {
        "n_trajectories": len(held),
        "held_out_regimes": [list(x) for x in ax.HELD_OUT_REGIMES],
        "rel_u_mean_by_step": u.mean(axis=0).tolist(),
        "rel_b_mean_by_step": b.mean(axis=0).tolist(),
        "rel_u_final_mean": float(u[:, -1].mean()),
        "rel_b_final_mean": float(b[:, -1].mean()),
    }
    result = {
        "status": "partial_checkpoint_evaluation",
        "warning": "Training stopped at the saved checkpoint before the 65-epoch K=8 release phase; this is not the final trained benchmark.",
        "checkpoint": str(ckpt),
        "checkpoint_next_epoch": int(state.get("next_epoch", -1)),
        "best_val_at_checkpoint": float(state.get("best_val", float("inf"))),
        "training_log_entries": len(state.get("log", [])),
        "gates": json.loads((RUN / "gate_results.json").read_text(encoding="utf-8")),
        "ablations": abl,
        "rollout": rollout,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    shutil.copy2(RUN / "train_log.json", OUT / "train_log.json")
    shutil.copy2(RUN / "run_metadata.json", OUT / "run_metadata.json")
    print(json.dumps(result, indent=2, default=float))
    print(f"Saved evaluation: {OUT / 'results.json'}")

if __name__ == "__main__":
    main()
