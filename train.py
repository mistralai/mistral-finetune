import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import ExitStack

import fire
import torch
import torch.cuda
import torch.distributed as dist
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader, DataLoader
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import MetricsLogger
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import load_model

logger = logging.getLogger("train")

def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)

def train(config: str) -> None:
    try:
        args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
        print(f"args: {args}")
        set_logger(logging.INFO)

        with ExitStack() as exit_stack:
            _train(args, exit_stack)
        logger.info("Closed everything!")
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise

def _train(
    args: TrainArgs,
    exit_stack: ExitStack,
) -> None:
    try:
        set_random_seed(args.seed)

        if "LOCAL_RANK" in os.environ:
            set_device()
            logger.info("Going to init comms...")
            try:
                dist.init_process_group(backend=BACKEND)
            except Exception as e:
                logger.error(f"Failed to initialize process group: {str(e)}")
                raise
        else:
            logger.error(
                "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
            )

        run_dir = Path(args.run_dir)
        _setup_run_directory(run_dir, args)

        metrics_logger: MetricsLogger = _setup_logger(run_dir, "train", args, exit_stack)
        eval_logger: MetricsLogger = _setup_logger(run_dir, "eval", args, exit_stack)

        model_folder = _get_model_folder(args.model_id_or_path)
        instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

        data_loader = _setup_data_loader(args, instruct_tokenizer, is_eval=False)
        eval_batches = _setup_eval_data(args, instruct_tokenizer) if not args.no_eval else None

        model = _setup_model(model_folder, args)
        optimizer = _setup_optimizer(model, args)
        scheduler = _setup_scheduler(optimizer, args)
        state = TrainState(args.max_steps)

        checkpointer = Checkpointer(
            model=model,
            state=state,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
        )

        prepare_mixed_precision(
            model.parameters(), param_dtype=torch.bfloat16, optim_dtype=torch.float32
        )

        _train_loop(args, model, optimizer, scheduler, state, data_loader, eval_batches, 
                    checkpointer, metrics_logger, eval_logger, instruct_tokenizer)

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise

def _setup_run_directory(run_dir: Path, args: TrainArgs) -> None:
    main_logger_info(f"Run dir: {run_dir}")
    if is_torchrun() and run_dir.exists():
        raise RuntimeError(f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}.")

    try:
        dist.barrier()
        run_dir.mkdir(exist_ok=True, parents=True)

        args_path = run_dir / "args.yaml"
        if not args_path.exists():
            args.save(args_path)
    except Exception as e:
        logger.error(f"Failed to set up run directory: {str(e)}")
        raise

def _setup_logger(run_dir: Path, tag: str, args: TrainArgs, exit_stack: ExitStack) -> MetricsLogger:
    logger = MetricsLogger(
        run_dir,
        tag=tag,
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        mlflow_args=args.mlflow,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(logger, f"{tag}_logger"))
    return logger

def _get_model_folder(model_id_or_path: str) -> Path:
    model_path = Path(model_id_or_path)
    if not model_path.is_dir():
        raise ValueError(f"Invalid folder path: {model_id_or_path}. Please set `args.initial_model` to a valid folder path.")
    return model_path

def _setup_data_loader(args: TrainArgs, instruct_tokenizer: Any, is_eval: bool) -> DataLoader:
    return build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed if not is_eval else None,
        rank=get_rank(),
        world_size=get_world_size(),
        is_eval=is_eval,
    )

def _setup_eval_data(args: TrainArgs, instruct_tokenizer: Any) -> List[Any]:
    if args.data.eval_instruct_data == "":
        raise ValueError("Either set `no_eval` to True or provide evaluation samples under `data.eval_instruct_data`")
    
    eval_data_loader = _setup_data_loader(args, instruct_tokenizer, is_eval=True)
    return list(eval_data_loader)

def _setup_model(model_folder: Path, args: TrainArgs) -> Union[torch.nn.Module, DDP]:
    if args.lora is None:
        raise ValueError("`args.lora` should be set to a valid value.")
    
    model = load_model(
        folder=model_folder,
        lora=args.lora,
        checkpoint=args.checkpoint,
        param_dtype=torch.bfloat16,
    )
    
    if torch.cuda.device_count() > 1:
        model = DDP(model)
    
    return model

def _setup_optimizer(model: torch.nn.Module, args: TrainArgs) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

def _setup_scheduler(optimizer: AdamW, args: TrainArgs) -> lr_scheduler.OneCycleLR:
    return lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

def _train_loop(
    args: TrainArgs,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: lr_scheduler.OneCycleLR,
    state: TrainState,
    data_loader: DataLoader,
    eval_batches: Optional[List[Any]],
    checkpointer: Checkpointer,
    metrics_logger: MetricsLogger,
    eval_logger: MetricsLogger,
    instruct_tokenizer: Any,
) -> None:
    model.train()
    torch.cuda.empty_cache()

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = torch.from_numpy(batch.y_mask).cuda(non_blocking=True) if batch.y_mask is not None else None

            output = model(input_ids=x, seqlens=batch.sizes)
            mb_loss = compute_loss_with_mask(output, y, y_mask)

            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += x.numel()

            if i < args.num_microbatches - 1:
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        upcast_mixed_precision(model.parameters(), optim_dtype=torch.float32)
        model.clip_grad_norm_(max_norm=args.max_norm)
        optimizer.step()
        downcast_mixed_precision(model.parameters(), param_dtype=torch.bfloat16)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if not args.no_eval and ((args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step):
            _perform_evaluation(model, eval_batches, state, avg_loss, eval_logger)

        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            _log_training_progress(state, avg_loss, last_lr, metrics_logger, args)

        if not args.no_ckpt and ((args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step):
            checkpointer.save_checkpoint(
                save_only_lora=args.save_adapters,
                dtype=torch.bfloat16,
                instruct_tokenizer=instruct_tokenizer,
            )

    main_logger_info("Training completed successfully!")

def _perform_evaluation(
    model: torch.nn.Module,
    eval_batches: List[Any],
    state: TrainState,
    avg_loss: float,
    eval_logger: MetricsLogger,
) -> None:
    evaluate(model, eval_batches, state)
    eval_logs = {
        "step": state.step,
        "train_loss": avg_loss,
        "eval_perplexity": state.this_eval_perplexity,
        "eval_loss": state.this_eval_loss,
    }
    main_logger_info(f"Evaluation: {eval_logs}")
    eval_logger.log(eval_logs, step=state.step)

def _log_training_progress(
    state: TrainState,
    avg_loss: float,
    last_lr: float,
    metrics_logger: MetricsLogger,
    args: TrainArgs,
) -> None:
    train_logs = {
        "step": state.step,
        "loss": avg_loss,
        "lr": last_lr,
        "tokens_per_sec": state.tokens_per_sec,
        "cuda_max_memory": torch.cuda.max_memory_allocated(),
        "cuda_allocated_memory": torch.cuda.memory_allocated(),
    }
    main_logger_info(f"Training progress: {train_logs}")
    metrics_logger.log(train_logs, step=state.step)

if __name__ == "__main__":
    fire.Fire(train)
