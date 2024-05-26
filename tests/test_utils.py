import os
import tempfile
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from finetune.data.args import DataArgs, InstructArgs
from finetune.data.data_loader import build_data_loader
from finetune.distributed import get_rank, get_world_size
from finetune.utils import set_random_seed


def is_float_equal(a, b, precision=5e-3):
    return abs(a - b) < precision


MODEL_PATH = os.getenv("DUMMY_MODEL")
assert MODEL_PATH != "", "Provide a path to a dummy model"
DATA_PATH = "tests/fixtures/sample_instruct.jsonl:.1,tests/fixtures/sample_instruct_2.jsonl:.1,tests/fixtures/sample_instruct_3.jsonl:.1"
EVAL_DATA_PATH = "tests/fixtures/sample_instruct.jsonl,tests/fixtures/sample_instruct_2.jsonl,tests/fixtures/sample_instruct_3.jsonl"


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Pipeline parallel group that the current rank belongs to.
_PIPELINE_PARALLEL_GROUP = None

_PIPELINE_PARALLEL_RANKS = None


def rmf(filename: str) -> None:
    """Remove a file like rm -f."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def runner(
    rank: int, test_func: Callable, deterministic: bool = False, *args: List[Any], **kwargs: Dict[str, Any]
) -> None:
    # At this point we're in a new process, torch options need to be set again
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(1357)

    test_func(rank, *args, **kwargs)


def spawn_for_all_world_sizes(
    test_func: Callable, world_sizes: List[int] = [], args: Any = [], deterministic: bool = False
) -> None:
    for world_size in world_sizes:
        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()

        try:
            torch.multiprocessing.spawn(
                runner,
                args=(test_func, deterministic, world_size, filename, filename_rpc, *args),
                nprocs=world_size,
                join=True,
            )
        finally:
            rmf(filename)
            rmf(filename_rpc)

def initialize_model_parallel(
    model_parallel_size_: int,
    pipeline_length: int = 1,
    *,
    model_parallel_backend: Optional[str] = None,
    pipeline_backend: Optional[str] = None,
    ddp_backend: Optional[str] = None
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = int(min(model_parallel_size_, world_size))
    rank = torch.distributed.get_rank()

    data_parallel_size = int(world_size / (model_parallel_size * pipeline_length))

    if torch.distributed.get_rank() == 0:
        print("> initializing model parallel with size {}".format(model_parallel_size_))
        print("> initializing ddp with size {}".format(data_parallel_size))
        print("> initializing pipeline with size {}".format(pipeline_length))

    groups = torch.LongTensor(range(world_size)).reshape(data_parallel_size, pipeline_length, model_parallel_size)

    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(pipeline_length):
        for k in range(model_parallel_size):
            group = torch.distributed.new_group(groups[:, j, k].tolist(), backend=ddp_backend)
            if j == found[1] and k == found[2]:
                _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        for j in range(pipeline_length):
            group = torch.distributed.new_group(groups[i, j, :].tolist(), backend=model_parallel_backend)
            if i == found[0] and j == found[1]:
                _MODEL_PARALLEL_GROUP = group

    global _PIPELINE_PARALLEL_GROUP
    assert _PIPELINE_PARALLEL_GROUP is None, "model parallel group is already initialized"
    global _PIPELINE_PARALLEL_RANKS
    assert _PIPELINE_PARALLEL_RANKS is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        for k in range(model_parallel_size):
            ranks = groups[i, :, k].tolist()
            group = torch.distributed.new_group(ranks, backend=pipeline_backend)
            if i == found[0] and k == found[2]:
                _PIPELINE_PARALLEL_GROUP = group
                _PIPELINE_PARALLEL_RANKS = ranks


def setup_mp_test_dist(rank, world_size, filename, model_parallel, seed=0):
    dist_init_for_testing(rank, world_size, filename)
    torch.cuda.set_device(rank)

    # Init NCCL
    backend = "nccl"
    initialize_model_parallel(
        model_parallel,
        model_parallel_backend=backend,
        pipeline_backend=backend,
        ddp_backend=backend,
    )

    set_random_seed(seed)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore


def dist_init_for_testing(
    rank: int, world_size: int, filename: str, filename_rpc: str = "", timeout: int = 30
):
    """
    Same than fairscale testing.dist_init but without rpc

    filename_rpc is here to keep same signature than fairscale init
    """

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    url = "file://" + filename

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if backend == "nccl" and torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"Requested world size {world_size} cannot be reached on this machine, not enough GPUs {torch.cuda.device_count()}"
        )

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=url,
        timeout=timedelta(seconds=timeout),
    )


def get_dataloader(
    seed: int = 0,
    seq_len: int = 10000,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    batch_size = 1
    rank = rank if rank is not None else get_rank()
    world_size = world_size if world_size is not None else get_world_size()

    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    instruct = InstructArgs(shuffle=False, dynamic_chunk_fn_call=False)

    data_args = DataArgs(
        data="",
        instruct_data="tests/fixtures/sample_instruct.jsonl:.1,tests/fixtures/sample_instruct_2.jsonl:.1,tests/fixtures/sample_instruct_3.jsonl:.1",
        instruct=instruct,
    )
    data_loader = build_data_loader(
        instruct_tokenizer,
        data_args,
        batch_size,
        seq_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=False,
    )
    return data_loader
