import logging
from typing import List

import numpy as np
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_with_mask
from .utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    batches: List[Batch],
    state: TrainState,
):
    # Create fake samples to make FSDP happy for unbalanced data
    num_samples = torch.tensor([len(batches)], device="cuda", dtype=torch.long)
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

    torch.distributed.all_gather(all_num_samples, num_samples)

    total_num_samples = int(torch.tensor(all_num_samples).sum().item())
    max_num_samples = int(torch.tensor(all_num_samples).max().item())

    for _ in range(max_num_samples - int(num_samples.item())):
        pad_x = np.zeros_like(batches[-1].x)
        pad_y = np.zeros_like(batches[-1].y)
        pad_sizes = batches[-1].sizes.copy()

        pad_batch = Batch(pad_x, pad_y, pad_sizes, is_pad_only=True)
        batches.append(pad_batch)

    # eval mode!
    model.eval()

    eval_loss = torch.tensor(0.0).cuda()
    main_logger_info("Start eval...")
    for batch in batches:
        x = torch.from_numpy(batch.x).cuda()
        y = torch.from_numpy(batch.y).cuda()
        y_mask = (
            torch.from_numpy(batch.y_mask).cuda() if batch.y_mask is not None else None
        )

        with torch.no_grad():
            output = model(
                input_ids=x,
                seqlens=batch.sizes,
            )

            if y_mask.sum() > 0:
                eval_loss += compute_loss_with_mask(output, y, y_mask)

            assert batch.is_pad_only or y.abs().sum() != 0, "Pad sample is used to compute loss."

    # sum loss
    main_logger_info("Eval finished!")

    dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    eval_loss /= total_num_samples

    state.this_eval_loss = eval_loss.item()
    state.this_eval_perplexity = (2**eval_loss).item()

    # train mode!
    model.train()
