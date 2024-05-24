from pathlib import Path

import pytest
import torch

from finetune.args import LoraArgs
from finetune.checkpointing import Checkpointer
from finetune.utils import TrainState
from finetune.wrapped_model import load_model
from tests.test_utils import MODEL_PATH, is_float_equal, setup_mp_test_dist
from utils.merge_lora import merge_checkpoints

from .test_utils import spawn_for_all_world_sizes

# fmt: off
EXPECTED_NON_LORA_KEYS = sorted(['layers.0.attention.wk.weight', 'layers.0.attention.wo.weight', 'layers.0.attention.wq.weight', 'layers.0.attention.wv.weight', 'layers.0.attention_norm.weight', 'layers.0.feed_forward.w1.weight', 'layers.0.feed_forward.w2.weight', 'layers.0.feed_forward.w3.weight', 'layers.0.ffn_norm.weight', 'layers.1.attention.wk.weight', 'layers.1.attention.wo.weight', 'layers.1.attention.wq.weight', 'layers.1.attention.wv.weight', 'layers.1.attention_norm.weight', 'layers.1.feed_forward.w1.weight', 'layers.1.feed_forward.w2.weight', 'layers.1.feed_forward.w3.weight', 'layers.1.ffn_norm.weight', 'norm.weight', 'output.weight', 'tok_embeddings.weight'])
EXPECTED_LORA_KEYS = sorted(['layers.0.attention.wq.lora_A.weight', 'layers.0.attention.wq.lora_B.weight', 'layers.0.attention.wk.lora_A.weight', 'layers.0.attention.wk.lora_B.weight', 'layers.0.attention.wv.lora_A.weight', 'layers.0.attention.wv.lora_B.weight', 'layers.0.attention.wo.lora_A.weight', 'layers.0.attention.wo.lora_B.weight', 'layers.0.feed_forward.w1.lora_A.weight', 'layers.0.feed_forward.w1.lora_B.weight', 'layers.0.feed_forward.w2.lora_A.weight', 'layers.0.feed_forward.w2.lora_B.weight', 'layers.0.feed_forward.w3.lora_A.weight', 'layers.0.feed_forward.w3.lora_B.weight', 'layers.1.attention.wq.lora_A.weight', 'layers.1.attention.wq.lora_B.weight', 'layers.1.attention.wk.lora_A.weight', 'layers.1.attention.wk.lora_B.weight', 'layers.1.attention.wv.lora_A.weight', 'layers.1.attention.wv.lora_B.weight', 'layers.1.attention.wo.lora_A.weight', 'layers.1.attention.wo.lora_B.weight', 'layers.1.feed_forward.w1.lora_A.weight', 'layers.1.feed_forward.w1.lora_B.weight', 'layers.1.feed_forward.w2.lora_A.weight', 'layers.1.feed_forward.w2.lora_B.weight', 'layers.1.feed_forward.w3.lora_A.weight', 'layers.1.feed_forward.w3.lora_B.weight'])
# fmt: on


@pytest.mark.parametrize(
    ("world_size", "save_only_lora", "enable_lora"),
    [
        (1, False, False),
        (2, False, False),
        (1, False, True),
        (2, False, True),
        (1, True, True),
        (2, True, True),  # this is the most important test! - FSDP only LORA
    ],
)
def test_states_retrieval(world_size, enable_lora, save_only_lora):
    spawn_for_all_world_sizes(
        _check_states_retrieval,
        world_sizes=[world_size],
        args=[enable_lora, save_only_lora],
        deterministic=True,
    )


def _check_states_retrieval(
    rank: int,
    world_size: int,
    filename: str,
    filename_rpc: str,
    enable_lora: bool,
    save_only_lora: bool,
):
    model_parallel = 1
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)

    folder = Path(MODEL_PATH)
    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=enable_lora),
        checkpoint=True,
        param_dtype=torch.bfloat16,
    )

    # mock a train state that has done three steps
    step = 3
    state = TrainState(max_steps=10, step=step)  # 10 is just a dummy value here

    # mock run_dir as we won't save anything in this test
    run_dir = Path("dir")
    use_sf = True

    checkpointer = Checkpointer(model, state, run_dir=run_dir, num_ckpt_keep=None)
    prefix = "lora" if enable_lora else "consolidated"

    assert checkpointer.dst_dir == Path(
        f"dir/checkpoints/checkpoint_00000{step}/consolidated"
    ), checkpointer.dst_dir
    assert checkpointer.consolidated_path(
        checkpointer.dst_dir, use_sf, save_only_lora=enable_lora
    ) == Path(
        f"dir/checkpoints/checkpoint_00000{step}/consolidated/{prefix}.safetensors"
    ), checkpointer.consolidated_path(
        checkpointer.dst_dir, use_sf, save_only_lora=enable_lora
    )

    # increase step by one
    state.start_step()

    assert checkpointer.dst_dir == Path(
        f"dir/checkpoints/checkpoint_00000{step + 1}/consolidated"
    ), checkpointer.dst_dir
    assert checkpointer.consolidated_path(
        checkpointer.dst_dir, use_sf, save_only_lora=enable_lora
    ) == Path(
        f"dir/checkpoints/checkpoint_00000{step + 1}/consolidated/{prefix}.safetensors"
    ), checkpointer.consolidated_path(
        checkpointer.dst_dir, use_sf, save_only_lora=enable_lora
    )

    assert all("lora" in k for k in EXPECTED_LORA_KEYS), EXPECTED_LORA_KEYS

    for save_dtype in [torch.float16, torch.bfloat16, torch.float32]:

        save_dict = checkpointer.retrieve_save_states(
            save_only_lora=save_only_lora, save_dtype=save_dtype
        )

        for k, v in save_dict.items():
            assert v.dtype == save_dtype, f"{k}: v.dtype"

        if save_only_lora:
            assert sorted(save_dict.keys()) == EXPECTED_LORA_KEYS, save_dict.keys()
        else:
            assert sorted(save_dict.keys()) == EXPECTED_NON_LORA_KEYS, save_dict.keys()

        EXPECTED_NON_LORA_VALUES = 34909.7500

        EXPECTED_LORA_VALUES = 984.4179840087891

        values_sum = sum(v.abs().float().sum().item() for v in save_dict.values())

        if save_only_lora:
            assert is_float_equal(
                values_sum, EXPECTED_LORA_VALUES, 5e-1
            ), f"{values_sum} for {save_dtype}"
        else:
            assert is_float_equal(
                values_sum, EXPECTED_NON_LORA_VALUES, 1e-1
            ), f"{values_sum} for {save_dtype}"


@pytest.mark.parametrize("world_size", [1, 2])
def test_lora_merge_equal(world_size):
    spawn_for_all_world_sizes(
        _check_lora_merge_equal,
        world_sizes=[world_size],
        deterministic=True,
    )


def _check_lora_merge_equal(
    rank: int, world_size: int, filename: str, filename_rpc: str
):
    model_parallel = 1
    enable_lora = True
    setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0)

    world_size // model_parallel

    folder = Path(MODEL_PATH)

    step = 3
    state = TrainState(max_steps=10, step=step)  # 10 is just a dummy value here
    run_dir = Path("dir")

    non_lora_model = load_model(
        folder=folder,
        lora=LoraArgs(enable=False),
        checkpoint=True,
        param_dtype=torch.bfloat16,
    )

    non_lora_checkpointer = Checkpointer(
        non_lora_model, state, run_dir=run_dir, num_ckpt_keep=None
    )
    orig_model = non_lora_checkpointer.retrieve_save_states(
        save_only_lora=False, save_dtype=torch.float32
    )

    scaling = 2.0

    model = load_model(
        folder=folder,
        lora=LoraArgs(enable=enable_lora, scaling=scaling),
        checkpoint=True,
        param_dtype=torch.bfloat16,
    )

    state_dict = model.state_dict()
    state_dict = {k: v + 0.01 if "lora" in k else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # mock a train state that has done three steps
    checkpointer = Checkpointer(model, state, run_dir=run_dir, num_ckpt_keep=None)

    for save_dtype in [torch.float16, torch.bfloat16, torch.float32]:
        model_dict = {
            k: torch.empty_like(v).copy_(v).to(save_dtype)
            for k, v in orig_model.items()
        }
        merged_save_dict = checkpointer.retrieve_save_states(
            save_only_lora=False, save_dtype=save_dtype
        )

        lora_save_dict = checkpointer.retrieve_save_states(
            save_only_lora=True, save_dtype=save_dtype
        )

        merge_checkpoints(
            model_dict, lora_save_dict, scaling=scaling, save_dtype=save_dtype
        )

        for k in model_dict.keys():
            torch.allclose(
                model_dict[k].cpu(), merged_save_dict[k].cpu(), atol=1e-3, rtol=1e-3
            )

        for k in model_dict.keys():
            # make sure that merged model differs from orig model
            if "attention" in k or "feed_forward" in k:
                not torch.allclose(
                    orig_model[k].to(save_dtype).cpu(),
                    merged_save_dict[k].cpu(),
                    atol=1e-3,
                    rtol=1e-3,
                )
