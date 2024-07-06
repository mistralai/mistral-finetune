import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pytest
import safetensors
import torch

from finetune.args import LoraArgs, OptimArgs, TrainArgs
from finetune.data.args import DataArgs, InstructArgs
from tests.test_utils import DATA_PATH, EVAL_DATA_PATH, MODEL_PATH, setup_mp_test_dist
from train import _train
from .test_utils import spawn_for_all_world_sizes

def file_size_and_md5(file_path):
    """
    Calculate the file size and an MD5-like hash of a safetensors file.

    Args:
        file_path (str): The path to the file.

    Returns:
        tuple: File size in bytes and a sum of absolute values of the state_dict items.
    """
    if not os.path.isfile(file_path):
        return "Error: File not found"

    file_size = os.path.getsize(file_path)
    state_dict = safetensors.torch.load_file(file_path)
    md5_sum = sum(v.abs().sum().item() for v in state_dict.values())

    return file_size, md5_sum

@pytest.mark.parametrize("enable_lora", [False, True])
def test_integration(enable_lora):
    """
    Integration test for the training process with and without LoRA enabled.

    Args:
        enable_lora (bool): Whether to enable LoRA (Low-Rank Adaptation).
    """
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    instruct_args = InstructArgs(shuffle=False, dynamic_chunk_fn_call=False)
    data_args = DataArgs(
        data="",
        instruct_data=DATA_PATH,
        eval_instruct_data=EVAL_DATA_PATH,
        instruct=instruct_args,
    )
    optim_args = OptimArgs(lr=0.01, weight_decay=0.1, pct_start=0.0)
    model_path = MODEL_PATH

    with tempfile.TemporaryDirectory() as tmpdirname:
        train_args = TrainArgs(
            data=data_args,
            model_id_or_path=model_path,
            run_dir=tmpdirname,
            seed=0,
            optim=optim_args,
                        max_steps=4,
            num_microbatches=1,
            lora=LoraArgs(:enable=enable_lora),
            save_adapters=enable_lora,
            checkpoint=True,
            no_eval=False,
        )
        spawn_for_all_world_sizes(
            _run_dummy_train,
            world_sizes=[2],
            deterministic=True,
            args=[train_args],
        )
        prefix = "lora" if enable_lora else "consolidated"
        ckpt_path = Path(tmpdirname) / f"checkpoints/checkpoint_00000{train_args.max_steps}/consolidated/{prefix}.safetensors"
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

        file_size, md5_hash = file_size_and_md5(ckpt_path)

        EXPECTED_FILE_SIZE = [8604200, 84760]
        EXPECTED_HASH = [50515.5, 1296.875]

        assert file_size == EXPECTED_FILE_SIZE[enable_lora], f"Expected file size {EXPECTED_FILE_SIZE[enable_lora]}, got {file_size}"
        assert abs(md5_hash - EXPECTED_HASH[enable_lora]) < 1e-2, f"Expected hash {EXPECTED_HASH[enable_lora]}, got {md5_hash}"


def _run_dummy_train(rank, world_size, filename, filename_rpc, args):
    """
    Run dummy training process for distributed training.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        filename (str): Filename for process communication.
        filename_rpc (str): RPC filename for process communication.
        args (TrainArgs): Training arguments.
    """
    setup_mp_test_dist(rank, world_size, filename, 1, seed=0)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)

