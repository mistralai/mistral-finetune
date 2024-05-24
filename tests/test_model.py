import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch

from finetune.args import LoraArgs
from finetune.checkpointing import Checkpointer
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.utils import TrainState
from finetune.wrapped_model import load_model
from model.transformer import (
    LoRALinear,
)
from tests.test_utils import (
    MODEL_PATH,
    get_dataloader,
    is_float_equal,
    setup_mp_test_dist,
)

from .test_utils import spawn_for_all_world_sizes

torch.backends.cudnn.deterministic = True  # use deterministic algorithms
torch.backends.cudnn.benchmark = False  # disable cuDNN benchmark


@pytest.mark.parametrize(
    ("world_size", "enable_lora", "dtype"),
    [
        (1, False, torch.float32),
        (1, True, torch.float32),
        (2, False, torch.float32),
        (2, True, torch.float32),
        (1, False, torch.bfloat16),
        (1, True, torch.bfloat16),
        (2, False, torch.bfloat16),
        (2, True, torch.bfloat16),
    ],
)
def test_weights_loading(world_size, enable_lora, dtype):
    spawn_for_all_world_sizes(
        _check_weights_loading,
        world_sizes=[world_size],
        args=[enable_lora, dtype],
        deterministic=True,
    )


def _check_weights_loading(
    rank: int,
    world_size: int,
    filename: str,
    filename_rpc: str,
    enable_lora: bool,
    dtype: torch.dtype,
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)

    folder = Path(MODEL_PATH)
    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=enable_lora),
        checkpoint=True,
        param_dtype=dtype,
    )

    # add hook so that LoRA weights are automatically merged:
    def register_merge_lora_hook(m: torch.nn.Module):
        def merge_lora(
            m: torch.nn.Module, destination: Dict[str, torch.Tensor], prefix: str, *args
        ):
            weight = m.merge_weight()
            destination[prefix + "weight"] = weight

        if isinstance(m, LoRALinear):
            m._merge_lora_handle = m._register_state_dict_hook(merge_lora)

    model.apply(register_merge_lora_hook)

    if world_size > 1:
        with model.summon_full_params(model, writeback=True):
            states = {
                k: v
                for k, v in model.state_dict().items()
                if "lora" not in k and "frozen" not in k
            }
    else:
        states = {
            k: v
            for k, v in model.state_dict().items()
            if "lora" not in k and "frozen" not in k
        }

    EXP_PARAM_SUM = 308.9932 if dtype == torch.float32 else 308.0
    params = sum([v.sum() for v in states.values()]).item()

    # LoRA is equal to no LoRA as LoRA weights should be init to 0
    assert is_float_equal(params, EXP_PARAM_SUM), params

    if enable_lora:
        lora_B_params = [
            v.float().abs().sum() for k, v in model.named_parameters() if "lora_B" in k
        ]

        assert len(lora_B_params) > 0
        assert sum(lora_B_params) == 0, "Lora_B should always be zero init"

        lora_A_params = [
            v.float().abs().sum() for k, v in model.named_parameters() if "lora_A" in k
        ]

        assert len(lora_A_params) > 0
        assert sum(lora_A_params) > 0, "Lora_A should init to non-zero values"


@pytest.mark.parametrize(
    ("world_size", "enable_lora"), [(1, False), (1, True), (2, False), (2, True)]
)
def test_fsdp_logits_and_loss(world_size, enable_lora):
    spawn_for_all_world_sizes(
        _check_fsdp_logits_and_loss,
        world_sizes=[world_size],
        args=[enable_lora],
        deterministic=True,
    )


def _check_fsdp_logits_and_loss(
    rank: int, world_size: int, filename: str, filename_rpc: str, enable_lora: bool
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)
    seq_len = 100

    folder = Path(MODEL_PATH)
    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=enable_lora),
        checkpoint=True,
        param_dtype=torch.bfloat16,
    )
    # By seting equal rank and world_size we can assure that both ranks see the same data and hence the average
    data_loader = get_dataloader(seq_len=seq_len, rank=0, world_size=2)

    batch = next(data_loader)

    x = torch.from_numpy(batch.x).cuda(non_blocking=True)
    y = torch.from_numpy(batch.y).cuda(non_blocking=True)
    y_mask = torch.from_numpy(batch.y_mask).cuda(non_blocking=True)

    # forward / backward
    output = model(
        input_ids=x,
        seqlens=batch.sizes,
    )

    # check logits
    # logits should be the same for LoRA and non-LoRA
    assert output.shape == (seq_len, model.args.vocab_size)
    output_sum = output.abs().float().sum().item()

    EXP_OUTPUT_WORLD_1 = 162617.625

    assert is_float_equal(output_sum, EXP_OUTPUT_WORLD_1, precision=1e1), output_sum

    # check loss is the same for all
    # loss should be the same for LoRA and non-LoRA
    mb_loss = compute_loss_with_mask(output, y, y_mask)

    EXPECTED_LOSS = 10.408413887023926

    assert is_float_equal(mb_loss.item(), EXPECTED_LOSS), mb_loss.item()


@pytest.mark.parametrize(
    ("world_size", "dtype"),
    [(1, torch.bfloat16), (2, torch.bfloat16), (1, torch.float32), (2, torch.float32)],
)
def test_fsdp_grads_non_lora(world_size, dtype):
    spawn_for_all_world_sizes(
        _check_fsdp_grads_non_lora,
        world_sizes=[world_size],
        deterministic=True,
        args=[dtype],
    )


def _check_fsdp_grads_non_lora(
    rank: int, world_size: int, filename: str, filename_rpc: str, dtype: torch.dtype
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)
    seq_len = 2048

    folder = Path(MODEL_PATH)
    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=False),
        checkpoint=True,
        param_dtype=dtype,
    )
    # same world_size to check for equality
    data_loader = get_dataloader(seq_len=seq_len, rank=0, world_size=2)

    batch = next(data_loader)

    x = torch.from_numpy(batch.x).cuda(non_blocking=True)
    y = torch.from_numpy(batch.y).cuda(non_blocking=True)
    y_mask = torch.from_numpy(batch.y_mask).cuda(non_blocking=True)

    # forward / backward
    output = model(
        input_ids=x,
        seqlens=batch.sizes,
    )

    mb_loss = compute_loss_with_mask(output, y, y_mask)
    mb_loss.backward()

    num_grad_params = sum([p.grad.numel() for p in model.parameters()])

    assert (4301120 // world_size) == num_grad_params, num_grad_params

    torch.distributed.barrier()

    sharded_flat_grads = sum(
        [p.grad.float().abs().sum().item() for p in model.parameters()]
    )

    print(f"{rank}: {world_size}: {dtype} = {sharded_flat_grads}")

    EXP_GRAD_WORLD_2_RANK_0 = 95.45827150344849
    EXP_GRAD_WORLD_2_RANK_1 = 86.09188461303711
    EXP_GRAD_WORLD_1 = EXP_GRAD_WORLD_2_RANK_0 + EXP_GRAD_WORLD_2_RANK_1

    if world_size == 1:
        assert is_float_equal(
            sharded_flat_grads, EXP_GRAD_WORLD_1, 2.0e-1
        ), sharded_flat_grads
    elif world_size == 2 and rank == 0:
        assert is_float_equal(
            sharded_flat_grads, EXP_GRAD_WORLD_2_RANK_0, 2.0e-1
        ), sharded_flat_grads
    elif world_size == 2 and rank == 1:
        assert is_float_equal(
            sharded_flat_grads, EXP_GRAD_WORLD_2_RANK_1, 2.0e-1
        ), sharded_flat_grads


@pytest.mark.parametrize(
    ("world_size", "dtype"),
    [(1, torch.bfloat16), (2, torch.bfloat16), (1, torch.float32), (2, torch.float32)],
)
def test_fsdp_grads_lora(world_size, dtype):
    spawn_for_all_world_sizes(
        _check_fsdp_grads_lora,
        world_sizes=[world_size],
        deterministic=True,
        args=[dtype],
    )


def _check_fsdp_grads_lora(
    rank: int, world_size: int, filename: str, filename_rpc: str, dtype: torch.dtype
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)
    seq_len = 2048

    folder = Path(MODEL_PATH)
    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=True),
        checkpoint=True,
        param_dtype=dtype,
    )
    # same world_size to check for equality
    data_loader = get_dataloader(seq_len=seq_len, rank=0, world_size=2)

    batch = next(data_loader)

    x = torch.from_numpy(batch.x).cuda(non_blocking=True)
    y = torch.from_numpy(batch.y).cuda(non_blocking=True)
    y_mask = torch.from_numpy(batch.y_mask).cuda(non_blocking=True)

    # forward / backward
    output = model(
        input_ids=x,
        seqlens=batch.sizes,
    )

    mb_loss = compute_loss_with_mask(output, y, y_mask)
    mb_loss.backward()

    num_grad_params = sum(
        [p.grad.numel() for p in model.parameters() if p.grad is not None]
    )

    assert (40960 // world_size) == num_grad_params, num_grad_params

    torch.distributed.barrier()

    sharded_flat_grads = sum(
        [
            p.grad.float().abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        ]
    )

    print(f"{rank}: {world_size}: {dtype} = {sharded_flat_grads}")

    EXP_GRAD_WORLD_2_RANK_0 = 3.0742580661177635
    EXP_GRAD_WORLD_2_RANK_1 = 3.074301045779139
    EXP_GRAD_WORLD_1 = EXP_GRAD_WORLD_2_RANK_0 + EXP_GRAD_WORLD_2_RANK_1

    if world_size == 1:
        assert is_float_equal(
            sharded_flat_grads, EXP_GRAD_WORLD_1, 2.0e-1
        ), sharded_flat_grads
    elif world_size == 2 and rank == 0:
        assert is_float_equal(
            sharded_flat_grads, EXP_GRAD_WORLD_2_RANK_0, 2.0e-1
        ), sharded_flat_grads
    elif world_size == 2 and rank == 1:
        assert is_float_equal(
            sharded_flat_grads, EXP_GRAD_WORLD_2_RANK_1, 2.0e-1
        ), sharded_flat_grads


@pytest.mark.parametrize(
    ("world_size", "dtype"),
    [(1, torch.bfloat16), (2, torch.bfloat16), (1, torch.float32), (2, torch.float32)],
)
def test_grad_update_lora(world_size, dtype):
    spawn_for_all_world_sizes(
        _check_grad_update_lora,
        world_sizes=[world_size],
        args=[dtype],
        deterministic=True,
    )


def _check_grad_update_lora(
    rank: int, world_size: int, filename: str, filename_rpc: str, dtype: torch.dtype
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)
    seq_len = 1000

    folder = Path(MODEL_PATH)
    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=True),
        checkpoint=True,
        param_dtype=dtype,
    )
    optimizer = torch.optim.AdamW(model.parameters())

    data_loader = get_dataloader(seq_len=seq_len)

    batch = next(data_loader)

    x = torch.from_numpy(batch.x).cuda(non_blocking=True)
    y = torch.from_numpy(batch.y).cuda(non_blocking=True)
    y_mask = (
        torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
        if batch.y_mask is not None
        else None
    )

    # forward / backward
    output = model(
        input_ids=x,
        seqlens=batch.sizes,
    )

    mb_loss = compute_loss_with_mask(output, y, y_mask)
    mb_loss.backward()

    lora_weight_sum = 0
    non_lora_weight_sum = 0
    for name, param in model.named_parameters():
        if "lora" in name or "norm" in name:
            assert param.grad is not None, name
            lora_weight_sum += param.data.float().abs().sum()
        else:
            assert param.grad is None, name
            non_lora_weight_sum += param.data.float().abs().sum()

    # update weights
    optimizer.step()

    new_lora_weight_sum = 0
    new_non_lora_weight_sum = 0
    for name, param in model.named_parameters():
        if "lora" in name or "norm" in name:
            assert param.grad is not None, name
            new_lora_weight_sum += param.data.float().abs().sum()
        else:
            assert param.grad is None, name
            new_non_lora_weight_sum += param.data.float().abs().sum()

    # make sure that LoRA weights changed, but non-LoRA weights stayed the same
    assert not is_float_equal(
        new_lora_weight_sum, lora_weight_sum, 1e-4
    ), f"New: {new_lora_weight_sum}, Old: {lora_weight_sum}"
    assert is_float_equal(
        new_non_lora_weight_sum, non_lora_weight_sum, 1e-4
    ), f"New: {new_non_lora_weight_sum}, Old: {non_lora_weight_sum}"


@pytest.mark.parametrize(
    ("enable_lora", "param_dtype"),
    [
        (False, torch.float32),
        (True, torch.float32),
        (False, torch.bfloat16),
        (True, torch.bfloat16),
    ],
)
def test_grads_fsdp_mp(enable_lora, param_dtype):
    with tempfile.TemporaryDirectory() as tmpdirname:
        for world_size in [1, 2]:
            spawn_for_all_world_sizes(
                _check_grads_fsdp_mp,
                world_sizes=[world_size],
                deterministic=True,
                args=[tmpdirname, enable_lora, param_dtype],
            )

        w1_sd = torch.load(Path(tmpdirname) / Path("params_w1.pt"), map_location="cpu")
        w2_sd = torch.load(Path(tmpdirname) / Path("params_w2.pt"), map_location="cpu")

        for k in w1_sd.keys():
            assert w1_sd[k].shape == w2_sd[k].shape, k
            atol = 10 if param_dtype == torch.float32 else 100
            assert (w1_sd[k] - w2_sd[k]).sum().abs().item() < atol


def _check_grads_fsdp_mp(
    rank: int,
    world_size: int,
    filename: str,
    filename_rpc: str,
    tmpdirname: str,
    enable_lora: bool,
    param_dtype: torch.dtype,
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)
    seq_len = 4096

    optim_dtype = torch.float32

    folder = Path(MODEL_PATH)
    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=enable_lora),
        checkpoint=True,
        param_dtype=param_dtype,
    )

    # high learning rate to show differences
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    # mock a train state that has done three steps
    steps = 4
    state = TrainState(max_steps=steps)

    # mock run_dir as we won't save anything in this test
    run_dir = Path(tmpdirname)

    checkpointer = Checkpointer(model, state, run_dir=run_dir, num_ckpt_keep=None)

    # make sure the same data is seen
    dataloaders = [
        get_dataloader(seq_len=seq_len, rank=rank + i, world_size=2)
        for i in range(2 - world_size + 1)
    ]

    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    for _ in range(steps):
        state.start_step()
        optimizer.zero_grad()

        for data_loader in dataloaders:
            torch.manual_seed(0)
            batch = next(data_loader)

            x = torch.from_numpy(batch.x).cuda()
            y = torch.from_numpy(batch.y).cuda()
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )

            # forward / backward
            output = model(
                input_ids=x,
                seqlens=batch.sizes,
            )

            mb_loss = compute_loss_with_mask(output, y, y_mask)
            mb_loss.backward()

            assert model.params[0].dtype == param_dtype

            print(f"rank: {rank}, world_size: {world_size}, x: {x.abs().sum()}")
            print(f"rank: {rank}, world_size: {world_size}, y: {y.abs().sum()}")
            print(f"rank: {rank}, world_size: {world_size}, x shape: {x.shape}")

            if y_mask is not None:
                print(
                    f"rank: {rank}, world_size: {world_size}, y_mask: {y_mask.abs().sum()}"
                )
            print(f"rank: {rank}, world_size: {world_size}, loss: {mb_loss}")

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                p.grad.div_(len(dataloaders))

        max_norm = 1.0
        model.clip_grad_norm_(max_norm=max_norm)

        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        optimizer.step()

        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

    save_dict = checkpointer.retrieve_save_states(
        save_only_lora=enable_lora, save_dtype=torch.float32
    )

    path = "params_w1.pt" if world_size == 1 else "params_w2.pt"
    torch.save(save_dict, Path(tmpdirname) / Path(path))
