from typing import Optional

import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor, target: torch.Tensor, target_mask: Optional[torch.Tensor]
):
    if target_mask is None:
        return F.cross_entropy(logits, target, reduction="mean")

    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.sum(mb_loss * target_mask) / torch.sum(target_mask)

    return mb_loss
