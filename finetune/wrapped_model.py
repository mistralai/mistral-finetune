import functools
import json
import logging
import math
from pathlib import Path
from typing import Callable, Union

import safetensors
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from model.args import ModelArgs, MoeArgs
from model.transformer import Transformer, TransformerBlock

from .args import LoraArgs
from .checkpointing import Checkpointer
from .distributed import (
    get_rank,
    get_world_size,
)

logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy(is_lora: bool) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes it's own FSDP group so that only a single Transformer block is sharded at a time
    - If LoRA is enabled, we additionally create separate FSDP sub-groups for every trainable and non-trainable parameter group
      since this is a requirement for mixed requires_grad=True/False training. See: https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(TransformerBlock,),
    )

    if not is_lora:
        return transformer_block_wrap_policy

    def fsdp_lora_policy_fn(module):
        return all(p.requires_grad for p in module.parameters())

    # For LoRA training, trainable and non-trainable parameters need to be put into
    # different FSDP groups
    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_lora_policy_fn
    )

    policies = [fsdp_lora_policy, transformer_block_wrap_policy]

    return functools.partial(torch_wrap._or_policy, policies=policies)


def log_train_params(model: Union[torch.nn.Module, FullyShardedDataParallel]):
    world_size = get_world_size()

    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameter are finetuned ({num_train_params / num_params * 100:.2f}%)."
    )


def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
        Initialize LoRA layers with Kaiming uniform and zeros.
        See original paper for more info: https://arxiv.org/abs/2106.09685 and
        original github repo: https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L122
    """
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]

                if m_name.split(".")[-1] == "lora_A":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "lora_B":
                    torch.nn.init.zeros_(param)
                else:
                    raise ValueError(
                        "Only Lora layers should be randomly initialized."
                    )


def load_model(
    folder: Path,
    lora: LoraArgs,
    checkpoint: bool,
    param_dtype: torch.dtype,
) -> FullyShardedDataParallel:
    with open(folder / "params.json", "r") as f:
        args = json.loads(f.read())

    model_args = ModelArgs(
        lora=lora,
        dim=args["dim"],
        n_layers=args["n_layers"],
        head_dim=args["head_dim"],
        hidden_dim=args["hidden_dim"],
        n_heads=args["n_heads"],
        n_kv_heads=args["n_kv_heads"],
        norm_eps=args["norm_eps"],
        vocab_size=args["vocab_size"],
    )

    if model_args.vocab_size == 32000:
        raise ValueError(
            f"Fine-tuning is not supported for older model versions with vocab_size 32000. Make sure to extend your model to vocab_size=32768 using `python -m utils.extend_model_vocab --original_model_ckpt {folder} --extended_model_ckpt {folder}_extended`."
        )

    assert (
        model_args.vocab_size >= 32768
    ), "Make sure to use a model with a vocab size of at least 32768"

    if args.get("rope_theta") is not None:
        model_args.rope_theta = args["rope_theta"]

    if args.get("moe") is not None:
        model_args.moe = MoeArgs(**args["moe"])

    with torch.device("meta"):
        model = Transformer(args=model_args, checkpoint=checkpoint)

    if get_rank() == 0:
        state_dict = load_state_dict(folder, dtype=param_dtype)

        model.load_state_dict(state_dict, assign=True)  # type: ignore
        logger.info("Loaded model on cpu!")

        if lora.enable:
            logger.info("Initializing lora layers ...")
            # initialize LoRA layers
            initialize_lora_parameters(model, param_dtype)

        assert not any(
            p.is_meta for p in model.parameters()
        ), "All parameters should be initialized by now"
        assert all(
            p.dtype == param_dtype for p in model.parameters()
        ), f"All parameters should be on {param_dtype}"

        logger.info("Finished initialization!")
        param_init_fn = None
    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(
            p.is_meta for p in model.parameters()
        ), "All parameters should be on meta"

    torch.distributed.barrier()

    # only finetune LoRA parameters and freeze before wrapping
    if lora.enable:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    auto_wrap_policy = get_fsdp_policy(model_args.lora.enable)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
    )
    main_logger_info("Model sharded!")

    log_train_params(wrapped_model)

    return wrapped_model


@torch.no_grad()
def load_state_dict(path: Path, dtype: torch.dtype):
    assert path.is_dir(), path

    this_safetensors_path = Checkpointer.consolidated_path(path, use_safetensors=True)
    this_torch_path = Checkpointer.consolidated_path(path, use_safetensors=False)

    assert (
        this_safetensors_path.exists() or this_torch_path.exists()
    ), f"Either {this_safetensors_path} or {this_torch_path} must exist."
    assert not (
        this_safetensors_path.exists() and this_torch_path.exists()
    ), f"Only one of {this_safetensors_path} or {this_torch_path} should exist."

    if this_safetensors_path.exists():
        logger.info(f"Reloading model from {this_safetensors_path} ...")
        model_state_dict = safetensors.torch.load_file(this_safetensors_path)
    else:
        logger.info(f"Reloading model from {this_torch_path} ...")
        model_state_dict = torch.load(this_torch_path)

    logger.info(f"Converting model to dtype {dtype} ...")

    for k, v in model_state_dict.items():
        model_state_dict[k] = v.to(dtype)

    return model_state_dict
