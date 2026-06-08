from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MHCConfig:
    """Configuration for multi-stream residual routing.

    The final stream collapse always uses the learned MHCHead (the faithful
    reproduction). The earlier fixed mean/sum reductions have been removed.
    """

    num_streams: int = 4
    sinkhorn_iters: int = 20


def sinkhorn_knopps(log_alpha: torch.Tensor, iters: int = 20) -> torch.Tensor:
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
    return log_alpha.exp()


class StreamExpand(nn.Module):
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = num_streams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        return x.repeat_interleave(self.num_streams, dim=0)


class MHCHead(nn.Module):
    """Learned final stream collapse (the faithful reproduction).

    Computes a per-token, per-stream sigmoid mix from the flattened multi-stream
    residual, then sums the weighted streams into a single residual for the LM
    head. The earlier fixed mean/sum reductions have been removed.
    """

    def __init__(
        self,
        num_residual_streams: int,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_residual_streams = num_residual_streams
        self.dim = dim
        self.eps = eps

        self.norm: Optional[nn.RMSNorm]
        self.dynamic_head_fn: Optional[nn.Parameter]
        self.head_scale: Optional[nn.Parameter]
        self.head_base: Optional[nn.Parameter]

        if num_residual_streams > 1:
            self.norm = nn.RMSNorm(dim * num_residual_streams, eps=eps)
            self.dynamic_head_fn = nn.Parameter(
                torch.zeros(dim * num_residual_streams, num_residual_streams)
            )
            self.head_scale = nn.Parameter(torch.ones(()) * 1e-2)
            self.head_base = nn.Parameter(torch.zeros(num_residual_streams))
        else:
            self.norm = None
            self.dynamic_head_fn = None
            self.head_scale = None
            self.head_base = None

    @torch.no_grad()
    def init_weights(self):
        if self.num_residual_streams == 1:
            return
        assert self.norm is not None
        assert self.dynamic_head_fn is not None
        assert self.head_scale is not None
        assert self.head_base is not None
        self.norm.weight.fill_(1.0)
        self.dynamic_head_fn.zero_()
        self.head_scale.fill_(1e-2)
        self.head_base.zero_()

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        streams = self.num_residual_streams
        if streams == 1:
            return residuals

        batch_streams, seq_len, dim = residuals.shape
        assert dim == self.dim
        assert batch_streams % streams == 0
        batch = batch_streams // streams

        residuals = residuals.view(batch, streams, seq_len, dim).transpose(1, 2)

        assert self.norm is not None
        assert self.dynamic_head_fn is not None
        assert self.head_scale is not None
        assert self.head_base is not None

        normed = residuals.reshape(batch, seq_len, streams * dim)
        normed = F.rms_norm(
            normed,
            self.norm.normalized_shape,
            self.norm.weight.to(normed.dtype),
            self.norm.eps,
        )
        logits = normed @ self.dynamic_head_fn.to(normed.dtype)
        mix = torch.sigmoid(
            logits * self.head_scale.to(logits.dtype) + self.head_base.to(logits.dtype)
        ) + self.eps
        mix = mix.to(residuals.dtype)
        return (residuals * mix.unsqueeze(-1)).sum(dim=2)


class ManifoldConstrainedHyperConnections(nn.Module):
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        layer_index: int | None = None,
        sinkhorn_iters: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_residual_streams > 0
        self.num_residual_streams = num_residual_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.init_residual_index = (layer_index or 0) % num_residual_streams

        self.norm = nn.RMSNorm(dim * num_residual_streams)

        init_alpha_pre = torch.ones((num_residual_streams, 1)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.0
        init_alpha_res = torch.ones((num_residual_streams, num_residual_streams)) * -8
        init_alpha_res.fill_diagonal_(0.0)
        self.static_alpha = nn.Parameter(
            torch.cat((init_alpha_pre, init_alpha_res), dim=1)
        )
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim * num_residual_streams,
                num_residual_streams * (num_residual_streams + 1),
            )
        )

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        beta_init = torch.ones(num_residual_streams) * -1.0
        beta_init[self.init_residual_index] = 1.0
        self.static_beta = nn.Parameter(beta_init)
        self.dynamic_beta_fn = nn.Parameter(
            torch.zeros(dim * num_residual_streams, num_residual_streams)
        )
        self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def init_weights(self):
        streams = self.num_residual_streams
        device = self.static_alpha.device
        dtype = self.static_alpha.dtype

        self.norm.weight.fill_(1.0)

        init_alpha_pre = torch.ones((streams, 1), device=device, dtype=dtype) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.0
        init_alpha_res = torch.ones((streams, streams), device=device, dtype=dtype) * -8
        init_alpha_res.fill_diagonal_(0.0)
        self.static_alpha.copy_(torch.cat((init_alpha_pre, init_alpha_res), dim=1))
        self.dynamic_alpha_fn.zero_()

        self.pre_branch_scale.fill_(1e-2)
        self.residual_scale.fill_(1e-2)

        beta_init = torch.ones(streams, device=device, dtype=dtype) * -1.0
        beta_init[self.init_residual_index] = 1.0
        self.static_beta.copy_(beta_init)
        self.dynamic_beta_fn.zero_()
        self.h_post_scale.fill_(1e-2)

    def width_connection(self, residuals: torch.Tensor):
        streams = self.num_residual_streams
        batch_streams, seq_len, dim = residuals.shape
        assert batch_streams % streams == 0
        batch = batch_streams // streams

        residuals = residuals.view(batch, streams, seq_len, dim).transpose(1, 2)
        normed = residuals.reshape(batch, seq_len, streams * dim)
        normed = F.rms_norm(
            normed,
            self.norm.normalized_shape,
            self.norm.weight.to(normed.dtype),
            self.norm.eps,
        )

        alpha_weight = self.dynamic_alpha_fn.to(normed.dtype)
        wc_weight = normed @ alpha_weight
        wc_weight = wc_weight.view(batch, seq_len, streams, streams + 1)

        alpha_scale = torch.cat(
            (
                self.pre_branch_scale.expand(1),
                self.residual_scale.expand(streams),
            )
        )
        dynamic_alpha = wc_weight * alpha_scale.to(wc_weight.dtype).view(
            1, 1, 1, streams + 1
        )
        alpha = dynamic_alpha + self.static_alpha.to(dynamic_alpha.dtype).view(
            1, 1, streams, streams + 1
        )

        alpha_pre = alpha[..., :1].float().sigmoid()
        alpha_residual = sinkhorn_knopps(alpha[..., 1:].float(), self.sinkhorn_iters)
        alpha = torch.cat((alpha_pre, alpha_residual), dim=-1)
        alpha = alpha.to(residuals.dtype)

        mixed = torch.einsum("btsm,btsd->btmd", alpha, residuals)
        branch_input = mixed[:, :, 0, :]
        next_residuals = (
            mixed[:, :, 1:, :].transpose(1, 2).reshape(batch_streams, seq_len, dim)
        )

        beta = normed @ self.dynamic_beta_fn.to(normed.dtype)
        beta = beta.float() * self.h_post_scale.float() + self.static_beta.to(
            beta.dtype
        ).view(1, 1, streams)
        beta = beta.sigmoid() * 2.0
        return branch_input, next_residuals, beta

    def depth_connection(
        self,
        branch_output: torch.Tensor,
        residuals: torch.Tensor,
        *,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, dim = branch_output.shape
        streams = self.num_residual_streams
        output = branch_output.unsqueeze(2) * beta.to(branch_output.dtype).unsqueeze(-1)
        output = output.transpose(1, 2).reshape(batch * streams, seq_len, dim)
        return self.dropout(output + residuals)

    def depth_connection_streams(
        self,
        branch_outputs: torch.Tensor,
        residuals: torch.Tensor,
        *,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, streams, dim = branch_outputs.shape
        assert streams == self.num_residual_streams
        output = branch_outputs * beta.to(branch_outputs.dtype).unsqueeze(-1)
        output = output.transpose(1, 2).reshape(batch * streams, seq_len, dim)
        return self.dropout(output + residuals)

    def forward(
        self, residuals: torch.Tensor, branch, *branch_args, **branch_kwargs
    ) -> torch.Tensor:
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)
        branch_output = branch(branch_input, *branch_args, **branch_kwargs)
        return self.depth_connection(branch_output, residuals, beta=residual_kwargs)
