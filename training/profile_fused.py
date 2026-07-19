"""Find the hot spots in the fused step's backward (scratch tool, not a gate)."""
import torch
from torch.profiler import profile, ProfilerActivity

from test_fused_parity import CH, HIDDEN, FIRE_RATE, EagerNCA, _aux
from fused_step import fused_nca_step

device = torch.device("cuda")
model = EagerNCA(2, 0, False, None).to(device)
x0 = torch.rand(8, CH, 64, 64, device=device) * 0.4
x0[:, 3] += 0.3
T = 80


def one_iter(fused):
    x = x0.clone().requires_grad_(True)
    y = x
    for s in range(T):
        if fused:
            y = fused_nca_step(y, model.w1, model.b1, model.w2, model.b2,
                               seed=1, step=s, fire_rate=FIRE_RATE, clamp=None)
        else:
            fm = (torch.rand(8, 1, 64, 64, device=device) <= FIRE_RATE).float()
            y = model(y, fm)
    loss = y.square().mean()
    loss.backward()


for mode in (True, False):
    one_iter(mode)  # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        one_iter(mode)
        torch.cuda.synchronize()
    print(f"\n===== {'FUSED' if mode else 'EAGER'} one iter (T={T}) =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14))
