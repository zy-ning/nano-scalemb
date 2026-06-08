"""Tests for common distributed initialization helpers."""

from datetime import timedelta

import nano_scalemb.common as common


def test_compute_init_passes_dist_timeout_to_process_group(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")

    init_calls = []

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(common.torch.cuda, "manual_seed", lambda seed: None)
    monkeypatch.setattr(common.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        common.torch, "set_float32_matmul_precision", lambda precision: None
    )
    monkeypatch.setattr(common.dist, "init_process_group", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(common.dist, "barrier", lambda: None)

    timeout = timedelta(minutes=120)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = common.compute_init(
        "cuda", dist_timeout=timeout
    )

    assert ddp is True
    assert ddp_rank == 0
    assert ddp_local_rank == 0
    assert ddp_world_size == 8
    assert init_calls == [
        {
            "backend": "nccl",
            "device_id": device,
            "timeout": timeout,
        }
    ]
