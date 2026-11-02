import torch, os, gc
src = open('AlienX-mhd.py').read().replace('if __name__ == "__main__":\n    main()', 'if False:\n    main()')
exec(compile(src, 'AlienX-mhd.py', 'exec'), globals())

free, total = torch.cuda.mem_get_info()
print(f"cuda free before: {free/2**30:.2f} GiB / total {total/2**30:.2f} GiB")

torch.manual_seed(0)
u_tr, b_tr = generate_trajectory(1200.0, 1200.0, 2, DT, seed=SEED)
u_t, b_t = u_tr[0].to(DEVICE), b_tr[0].to(DEVICE)
su = u_t.square().mean().sqrt(); sb = b_t.square().mean().sqrt()
u_c = (u_t/su).permute(1,2,3,0).reshape(1, GRID**3, 3)
b_c = (b_t/sb).permute(1,2,3,0).reshape(1, GRID**3, 3)
u_t1, b_t1 = u_tr[1].to(DEVICE), b_tr[1].to(DEVICE)
u_tgt = u_t1/su; b_tgt = b_t1/sb
re_t = torch.tensor([1200.0], device=DEVICE); rm_t = torch.tensor([1200.0], device=DEVICE)

model = AlienX_MHD(hidden=HIDDEN, blocks=LAYERS, heads=HEADS, chunk=CHUNK,
                   re_max=1200.0, rm_max=1200.0, sigma_z=SIGMA_Z).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

try:
    opt.zero_grad()
    B_pred, u_pred, _, _ = model_step(model, b_c, u_c, re_t, rm_t)
    up_sp = u_pred.reshape(1, GRID, GRID, GRID, 3).permute(0,4,1,2,3)[0]
    bp_sp = B_pred.reshape(1, GRID, GRID, GRID, 3).permute(0,4,1,2,3)[0]
    loss, _ = loss_state(up_sp, bp_sp, u_tgt, b_tgt)
    print(f"forward ok, loss={float(loss):.4e}, peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    loss.backward()
    print(f"backward ok, peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"step ok, peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    # second iteration to check stability
    opt.zero_grad()
    B_pred, u_pred, _, _ = model_step(model, b_c, u_c, re_t, rm_t)
    up_sp = u_pred.reshape(1, GRID, GRID, GRID, 3).permute(0,4,1,2,3)[0]
    bp_sp = B_pred.reshape(1, GRID, GRID, GRID, 3).permute(0,4,1,2,3)[0]
    loss, _ = loss_state(up_sp, bp_sp, u_tgt, b_tgt)
    loss.backward()
    opt.step()
    print(f"iter2 ok, loss={float(loss):.4e}, peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1)
print("MEM PROBE PASS")