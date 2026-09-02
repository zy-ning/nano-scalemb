"""Loss-free balancing (DeepSeek; Wang et al. 2024).

Adapted from the sibling harness ``nanochat-moe`` (``moe/balancers/loss_free.py``).

A per-expert bias is added to the gate score for SELECTION only; the routing
weights use the unbiased score, so the task-loss gradient is never touched. The
bias is hand-updated each step from the observed load: overloaded experts get
their bias pushed down so they are selected less often next step.

Per step, with f the normalized load and Q = 1/n:

    d = f - Q
    bias -= gamma * (sign(d)      if update == 'sign'
                     d / rms(d)   if update == 'rms')
    bias -= bias.mean()           # drop the constant-shift DoF

Order matters (no info leak): the Router selects with the CURRENT bias, then calls
update() with the resulting counts. Under DDP the counts are summed across ranks
first so every rank holds an identical bias -- the model weights are synced, so
routing state must be too.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from nano_scalemb.moe.balancers.base import Balancer


class LossFreeBalancer(Balancer):
    def __init__(
        self,
        n_experts: int,
        top_k: int | None = None,
        gamma: float = 0.01,
        update: str = "rms",
    ):
        super().__init__(n_experts, top_k)
        if update not in ("sign", "rms"):
            raise ValueError(f"loss_free update must be 'sign'|'rms', got {update!r}")
        self.gamma = float(gamma)
        self.update_rule = update
        # Selection state, not an optimizer parameter (no autograd). Persistent so
        # it round-trips through checkpoints. fp32 to match the fp32 routing math.
        self.register_buffer("bias_buf", torch.zeros(n_experts, dtype=torch.float32))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.bias_buf.zero_()

    def bias(self, s: torch.Tensor) -> torch.Tensor:
        return self.bias_buf.to(s.dtype)

    @torch.no_grad()
    def update(self, counts: torch.Tensor, s: torch.Tensor, k_idx: torch.Tensor) -> None:
        c = counts.to(self.bias_buf.dtype)  # precision follows the state buffer
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c, op=dist.ReduceOp.SUM)  # global load -> identical bias per rank
        tot = c.sum().clamp_min(1.0)
        f = c / tot  # normalized load, sums to 1
        d = f - 1.0 / self.n_experts  # deviation from uniform
        if self.update_rule == "sign":
            step = torch.sign(d)
        else:
            rms = torch.sqrt((d * d).mean()) + 1e-12
            step = d / rms
        self.bias_buf -= self.gamma * step.to(self.bias_buf.dtype)
        self.bias_buf -= self.bias_buf.mean()  # remove the redundant DoF
