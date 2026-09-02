"""Expert bank with a swappable kernel.

Adapted from the sibling harness ``nanochat-moe`` (``moe/experts.py``).

Backends:
  - ``scattermoe`` : Triton dropless grouped-GEMM (https://github.com/shawntan/scattermoe,
    Apache-2.0). The default; installed via the ``gpu`` extra.
  - ``fallback``   : correctness-equivalent pure-torch per-expert loop. Runs on
    CPU/MPS and is what the test suite uses.

The router is EXTERNAL: ``forward`` takes the per-token top-k weights and indices
and only runs the experts.

This module owns the canonical expert weights and calls scattermoe's *functional*
path on them (rather than wrapping ``scattermoe.mlp.MLP``). That matters: the GPT
is built on meta then ``to_empty()``-ed, and if the same Parameter were registered
in two submodules, ``to_empty`` would hand each registration a fresh tensor and
silently break the sharing. One registration here == one tensor, always. The same
reasoning is why Mobius's shared pools are referenced, not re-registered (see
``MoEBlock``).

Weight layout matches ``scattermoe.ParallelExperts``:
    w_in  : (E, d_ff,    d_model)
    w_out : (E, d_model, d_ff)
No bias. Activation applied between the two grouped-GEMMs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {
    "gelu": lambda: nn.GELU(),
    "relu": lambda: nn.ReLU(),
    "silu": lambda: nn.SiLU(),
}


class Experts(nn.Module):
    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_ff: int,
        top_k: int,
        activation: str = "gelu",
        backend: str = "scattermoe",
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.activation_name = activation
        if activation not in _ACT:
            raise ValueError(f"unknown activation {activation!r}; have {sorted(_ACT)}")
        self.act = _ACT[activation]()
        if backend not in ("scattermoe", "fallback"):
            raise ValueError(f"unknown backend {backend!r}; have 'scattermoe'|'fallback'")
        self.backend = backend

        self.w_in = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        self.w_out = nn.Parameter(torch.empty(n_experts, d_model, d_ff))

    @torch.no_grad()
    def reset_parameters(self):
        # w_in: uniform std 1/sqrt(d_model) (matches the GPT matrix init in
        # gpt.py:init_weights); w_out: zeros so the MoE block contributes nothing
        # at init, the same convention as the dense MLP's c_proj.
        s = 3**0.5 * self.d_model**-0.5
        nn.init.uniform_(self.w_in, -s, s)
        nn.init.zeros_(self.w_out)

    def forward(
        self, x: torch.Tensor, k_weights: torch.Tensor, k_idxs: torch.Tensor
    ) -> torch.Tensor:
        """x: (T, d_model); k_weights, k_idxs: (T, top_k). Returns (T, d_model)."""
        if self.backend == "scattermoe":
            return self._forward_scattermoe(x, k_weights, k_idxs)
        return self._forward_fallback(x, k_weights, k_idxs)

    def _autocast_dtype(self, x):
        """The dtype the experts should run in.

        Under autocast we pick the autocast dtype ourselves rather than letting
        autocast rewrite ops inside the kernel -- see _forward_scattermoe.
        """
        if torch.is_autocast_enabled(x.device.type):
            return torch.get_autocast_dtype(x.device.type)
        return x.dtype

    def _forward_scattermoe(self, x, k_weights, k_idxs):
        try:
            from scattermoe.parallel_experts import flatten_sort_count, parallel_linear
        except ImportError as e:  # pragma: no cover - depends on the gpu extra
            raise ImportError(
                "backend='scattermoe' needs the scattermoe package (Triton). "
                "Install it with `uv sync --extra gpu`, or pass "
                "backend='fallback' for the pure-torch path."
            ) from e

        sorted_idx, sorted_scattered, offsets = flatten_sort_count(
            k_idxs, num_experts=self.n_experts
        )
        # scattermoe's autograd.Function is not autocast-aware: under autocast the
        # grouped GEMM emits the autocast dtype while `gates` stays fp32, and its
        # backward computes `output_expanded @ grad_out` across the two, raising
        # "expected scalar type Float but found BFloat16". So we disable autocast
        # for the kernel region and cast every operand to one dtype by hand.
        # Master weights stay fp32 and their grads still come back fp32.
        dtype = self._autocast_dtype(x)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x.to(dtype)
            w_in = self.w_in.permute(0, 2, 1).to(dtype)  # (E, d_in, d_out) for the kernel
            w_out = self.w_out.permute(0, 2, 1).to(dtype)
            h = parallel_linear(
                xf,
                w_in,
                self.top_k,
                sorted_idx,
                sorted_scattered,
                offsets,
                grouped_out=True,
            )
            h = self.act(h)
            y = parallel_linear(
                h,
                w_out,
                1,
                sorted_idx,
                sorted_scattered,
                offsets,
                grouped_in=True,
                gates=k_weights.to(dtype),
            )
        return y

    def _forward_fallback(self, x, k_weights, k_idxs):
        T, d = x.shape
        k = k_idxs.shape[1]
        # Match the scattermoe path's dtype handling so the two backends produce
        # the same thing under autocast, not just in plain fp32.
        dtype = self._autocast_dtype(x)
        # Disable autocast for the same reason as the scattermoe path, plus one
        # of its own: Tensor.sum is an autocast fp32-promotion op, so the final
        # top-k reduction would silently return fp32 here while the kernel path
        # returns bf16. Both backends must have the same dtype contract.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xr = x.to(dtype).repeat_interleave(k, dim=0)  # (T*k, d)
            idx = k_idxs.reshape(-1)  # (T*k,)
            w = k_weights.reshape(-1).to(dtype)  # (T*k,)
            y = torch.zeros(T * k, d, dtype=dtype, device=x.device)
            for e in range(self.n_experts):
                m = idx == e
                if m.any():
                    h = self.act(xr[m] @ self.w_in[e].t().to(dtype))
                    y[m] = h @ self.w_out[e].t().to(dtype)
            y = y * w.unsqueeze(1)
            return y.view(T, k, d).sum(dim=1)
