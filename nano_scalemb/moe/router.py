"""MoE router: linear gate + balancer bias + hard top-k.

Adapted from the sibling harness ``nanochat-moe`` (``moe/router.py``), trimmed to
the linear gate (the STAR / SoftMoE / ReMoE / ProbMoE / DirMoE router families are
not ported).

    logits = x @ Wg                       gate (Wg is 2D -> Muon)
    s      = sigmoid(logits)              gate score (softmax is a knob)
    b      = balancer.bias(s)             per-expert additive bias, BEFORE top-k (no grad)
    k_idx  = topk(s + b, k).indices       SELECTION uses the biased score
    k_w    = normalize(gather(s, k_idx))  WEIGHTS use the UNBIASED s

That last split is the defining property of loss-free balancing: the bias steers
*which* experts fire but never touches the value multiplied into their output, so
the task-loss gradient stays uncontaminated.

Routing math runs in fp32 (stable top-k). Weights are returned fp32; the block
casts them to the activation dtype.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_scalemb.moe.metrics import expert_counts, router_stats


class Router(nn.Module):
    def __init__(self, d_model, n_experts, top_k, balancer, gate="sigmoid"):
        super().__init__()
        if gate not in ("sigmoid", "softmax"):
            raise ValueError(f"unknown gate {gate!r}; have 'sigmoid'|'softmax'")
        assert 1 <= top_k <= n_experts, f"need 1 <= top_k <= n_experts, got {top_k}/{n_experts}"
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = gate
        self.Wg = nn.Linear(d_model, n_experts, bias=False)  # fp32 master, 2D -> Muon
        self.balancer = balancer
        # The train loop flips this on log steps only: router_stats() syncs to host,
        # and an every-step sync serializes the pipeline and breaks compile tracing.
        self.collect_stats = True

    @torch.no_grad()
    def reset_parameters(self):
        s = 3**0.5 * self.d_model**-0.5
        nn.init.uniform_(self.Wg.weight, -s, s)
        # Buffers come back uninitialized after meta -> to_empty, so stateful
        # balancers MUST zero theirs here.
        self.balancer.reset_state()

    def _score(self, logits):
        if self.gate == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def forward(self, x):
        """x: (T, d_model).

        Returns (k_w fp32 (T,k), k_idx long (T,k), s fp32 (T,n), stats|None).
        """
        logits = F.linear(x.float(), self.Wg.weight.float())  # (T, n) fp32
        s = self._score(logits)

        b = self.balancer.bias(s)  # (n,) additive bias, or 0.0
        sel = s if isinstance(b, float) else s + b
        _, k_idx = torch.topk(sel, self.top_k, dim=-1)  # selection on the BIASED score

        k_w = s.gather(-1, k_idx)  # weights from the UNBIASED s
        k_w = k_w / k_w.sum(-1, keepdim=True).clamp_min(1e-9)  # normalized over selected

        stats = router_stats(s, k_idx, self.n_experts) if self.collect_stats else None
        counts = expert_counts(k_idx, self.n_experts)
        self.balancer.update(counts, s, k_idx)  # post-step state update

        return k_w, k_idx, s, stats
