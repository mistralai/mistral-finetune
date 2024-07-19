import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import safetensors.torch
import torch
from mistral_common.tokens.tokenizers.sentencepiece import (
    InstructTokenizerBase,
    SentencePieceTokenizer,
)
from torch.distributed import barrier
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from model.transformer import LoRALinear

from .distributed import get_rank, get_world_size
from .utils import TrainState

logger = logging.getLogger("checkpointing")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


class Checkpointer:
    """A class to save PyTorch model and optimizer states"""

    def __init__(
        self,
        model: FullyShardedDataParallel,
        state: TrainState,
        run_dir: Union[Path, str],
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_ckpt_keep: Optional[int] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> Path:
        return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "consolidated"

    @staticmethod
    def consolidated_path(
        ckpt_dir: Path, use_safetensors: bool, save_only_lora: Optional[bool] = False
    ) -> Path:
        suffix = "safetensors" if use_safetensors else "00.pth"
        prefix = "lora" if save_only_lora else "consolidated"

        return ckpt_dir / f"{prefix}.{suffix}"

    @staticmethod
    def _tmp(ckpt_dir: Path) -> Path:
        return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")

    def write_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "params.json"
        with open(params_path, "w") as f:
            model_args = self.model.args.to_dict()

            f.write(json.dumps(model_args, indent=4))

    def delete_old_ckpts(self) -> List[Path]:
        all_saved_ckpts = [d for d in self.ckpt_dir.iterdir() if d.is_dir()]

        # Sort directories by creation time (oldest to newest)
        all_saved_ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        ckpts_to_delete = all_saved_ckpts[self.num_ckpt_keep :]

        for ckpt_to_delete in ckpts_to_delete:
            try:
                shutil.rmtree(ckpt_to_delete)
                main_logger_info(f"Deleted ckpt: {ckpt_to_delete}")
            except OSError as e:
                main_logger_info(f"Error deleting directory {ckpt_to_delete}: {e}")

        return ckpts_to_delete

    @staticmethod
    def get_lora_states(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in state_dict.items() if "lora" in k}

    @staticmethod
    def get_non_lora_states(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in state_dict.items()
            if not any(l_key in k for l_key in ["lora", "frozen"])
        }

    @torch.no_grad()
    def retrieve_save_states(
        self, save_only_lora: bool, save_dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        if save_only_lora:
            assert (
                self.model.args.lora.enable
            ), "Cannot save LoRA checkpoint as LoRA training is not enabled."

        # remove all potential hooks
        for module in self.model.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                module._merge_lora_handle.remove()  # type: ignore

        # merge weights if we don't just save LoRA
        if not save_only_lora:

            def merge_lora(
                m: torch.nn.Module,
                destination: Dict[str, torch.Tensor],
                prefix: str,
                *args,
            ):
                weight = m.merge_weight()  # type: ignore
                destination[prefix + "weight"] = weight

            for module in self.model.modules():
                if isinstance(module, LoRALinear):
                    module._merge_lora_handle = module._register_state_dict_hook(
                        merge_lora
                    )

        offload_to_cpu = get_world_size() > 1
        if save_only_lora:

            def is_trainable_fsdp(
                module: Union[torch.nn.Module, FullyShardedDataParallel],
            ):
                is_fsdp = isinstance(module, FullyShardedDataParallel)
                all_params_have_grads = is_fsdp and all(
                    p.requires_grad is True for p in module.parameters()
                )

                # need to make sure only lowest fsdp wrap is used
                is_leaf_node = is_fsdp and len(list(module.module.children())) == 0  # type: ignore

                return is_fsdp and all_params_have_grads and is_leaf_node

            # extract all modules with only trainable weights
            modules = {
                k: m for k, m in self.model.named_modules() if is_trainable_fsdp(m)
            }

            states = {}
            for key, module in modules.items():
                assert isinstance(
                    module, FullyShardedDataParallel
                ), "`module` should be an instance of `FullyShardedDataParallel`"
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                with module.summon_full_params(
                    module, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    states.update(
                        {
                            f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                            for k, v in module.state_dict().items()
                        }
                    )
        else:
            # make sure you have enough CPU RAM available to save the full model
            assert isinstance(
                self.model, FullyShardedDataParallel
            ), "`self.model` should be an instance of `FullyShardedDataParallel`"
            with self.model.summon_full_params(
                self.model, writeback=True, offload_to_cpu=offload_to_cpu
            ):
                states = self.get_non_lora_states(self.model.state_dict())
                states = {k: v.to(dtype=save_dtype) for k, v in states.items()}

        states = dict(sorted(states.items()))
        return states

    @staticmethod
    def save_tokenizer(instruct_tokenizer: InstructTokenizerBase, tmp_dst: Path):
        if isinstance(instruct_tokenizer.tokenizer, SentencePieceTokenizer):
            serialized_spm = (
                instruct_tokenizer.tokenizer._model.serialized_model_proto()
            )  # type: ignore

            tokenizer_path = tmp_dst / "tokenizer.model.v3"

            with open(tokenizer_path, "wb") as f:
                f.write(serialized_spm)
        else:
            path = instruct_tokenizer.tokenizer._path
            assert path is not None
            shutil.copy(path, tmp_dst / "tekken.json")

    @torch.no_grad()
    def save_checkpoint(
        self,
        save_only_lora: bool,
        dtype: torch.dtype = torch.float16,
        instruct_tokenizer: Optional[InstructTokenizerBase] = None,
    ):
        tmp_dst = self._tmp(self.dst_dir)
        main_logger_info(
            f"Dumping checkpoint in {self.dst_dir} using tmp name: {tmp_dst.name}"
        )

        assert not self.dst_dir.exists(), f"dst exists {self.dst_dir}"
        tmp_dst.mkdir(parents=True, exist_ok=True)

        states: Dict[str, torch.Tensor] = self.retrieve_save_states(
            save_only_lora, dtype
        )

        barrier()

        if self.rank == 0:
            # save checkpoint in tmp path
            safetensors.torch.save_file(
                states,
                self.consolidated_path(
                    tmp_dst, use_safetensors=True, save_only_lora=save_only_lora
                ),  # always use safetensors for checkpointing
            )

            self.write_params_info(tmp_dst)

            # save tokenizer
            if instruct_tokenizer is not None:
                self.save_tokenizer(instruct_tokenizer, tmp_dst)

            assert not self.dst_dir.exists(), f"should not happen! {self.dst_dir}"
            tmp_dst.rename(self.dst_dir)

            logger.info(
                f"Done dumping checkpoint in {self.dst_dir} for step: {self.state.step}"
            )

            # delete last n checkpoints
            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
