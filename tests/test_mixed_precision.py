from pathlib import Path

import pytest
import torch

from finetune.args import LoraArgs
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.wrapped_model import load_model
from tests.test_utils import MODEL_PATH, get_dataloader, setup_mp_test_dist

from .test_utils import spawn_for_all_world_sizes


@pytest.mark.parametrize(
    ("world_size", "enable_lora"), [(1, False), (1, True), (2, False), (2, True)]
)
def test_mixed_precision(world_size, enable_lora):
    spawn_for_all_world_sizes(
        _check_mixed_precision,
        world_sizes=[world_size],
        args=[enable_lora],
        deterministic=True,
    )


def _check_mixed_precision(
    rank: int, world_size: int, filename: str, filename_rpc: str, enable_lora: bool
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)
    seq_len = 100

    folder = Path(MODEL_PATH)
    # mixed precision
    param_dtype = torch.bfloat16
    optim_dtype = torch.float32

    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=enable_lora),
        checkpoint=True,
        param_dtype=param_dtype,
    )

    optimizer = torch.optim.AdamW(model.parameters())

    # initialize mixed precision training for TP
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    data_loader = get_dataloader(seq_len=seq_len)

    # ensure every parameter that requires a grad has a _mp_param of optim_dtype precision
    for param in model.parameters():
        assert param.dtype == param_dtype
        if param.requires_grad:
            assert param._mp_param.dtype == optim_dtype
            assert (
                param._mp_param.tolist() == param.data.to(optim_dtype).tolist()
            ), "mp param has to match param in optim dtype precision"
        else:
            assert not hasattr(param, "_mp_param")

    # test three train steps
    for _ in range(3):

        optimizer.zero_grad()

        # micro-batching
        for _ in range(2):
            batch = next(data_loader)

            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )

            output = model(
                input_ids=x,
                seqlens=batch.sizes,
            )

            mb_loss = compute_loss_with_mask(output, y, y_mask)
            mb_loss.backward()

        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # ensure all params are upcasted correctly and mp param equals param
        param_sum = 0
        for param in model.parameters():
            if param.requires_grad:
                assert param.dtype == optim_dtype, param.dtype
                assert (
                    param._mp_param.tolist() == param.data.tolist()
                ), "mp param and param should point to the same data"
                assert param.grad.dtype == optim_dtype
                assert param._temp.dtype == param_dtype
                param_sum += param.data.float().abs().sum()
            else:
                assert param.dtype == param_dtype

        optimizer.step()

        # ensure that after optimizer step params are still in optim dtype precision
        new_param_sum = 0
        for param in model.parameters():
            if param.requires_grad:
                assert param.dtype == optim_dtype
                assert param._mp_param.dtype == optim_dtype
                assert param.grad.dtype == optim_dtype
                new_param_sum += param.data.float().abs().sum()
            else:
                assert param.dtype == param_dtype

        assert new_param_sum != param_sum, "Make sure parameters are updated"

        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        # ensure that before new forward pass params are downcast to param dtype
        for param in model.parameters():
            assert param.dtype == param_dtype
            if param.requires_grad:
                assert param._mp_param.dtype == optim_dtype
                assert param.grad.dtype == param_dtype
                assert (
                    param._mp_param.to(param_dtype).tolist() == param.data.tolist()
                ), "mp param has to match param in optim dtype precision"
