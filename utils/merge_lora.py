import argparse
from typing import Dict, Optional

import safetensors.torch
import torch
import tqdm


def merge_checkpoints(
    model_checkpoint: Dict[str, torch.Tensor],
    lora_checkpoint: Dict[str, torch.Tensor],
    scaling: float,
    save_dtype: Optional[torch.dtype] = None,
):
    save_dtype = save_dtype or next(iter(lora_checkpoint.values())).dtype
    print(f"Merging to {save_dtype} precision...")

    keys_to_update = [
        key for key in lora_checkpoint.keys() if "norm" in key or "lora_A" in key
    ]
    assert any(
        "lora_A" in k or "lora_B" in k for k in keys_to_update
    ), "No `lora` keys found in your checkpoint. Check that `lora_ckpt` is correct."

    for key in tqdm.tqdm(keys_to_update):
        if "norm" in key:
            model_checkpoint[key] = lora_checkpoint[key].to("cpu")
        else:
            weight_name = key.replace("lora_A.weight", "weight")

            lora_A_weight = lora_checkpoint[key].to("cuda")
            lora_B_weight = lora_checkpoint[key.replace("lora_A", "lora_B")].to("cuda")

            weight = lora_B_weight.mm(lora_A_weight) * scaling
            weight += model_checkpoint[weight_name].to("cuda")
            weight = weight.to(save_dtype)

            model_checkpoint[weight_name] = weight.to("cpu")

    # cast all tensors to save dtype
    for key in tqdm.tqdm(model_checkpoint.keys()):
        model_checkpoint[key] = model_checkpoint[key].to(save_dtype)


def load(filename: str):
    if filename.endswith(".safetensors"):
        return safetensors.torch.load_file(filename)
    else:
        return torch.load(filename)


def main(args):
    model_checkpoint = load(args.initial_model_ckpt)
    lora_checkpoint = load(args.lora_ckpt)

    merge_checkpoints(model_checkpoint, lora_checkpoint, args.scaling)

    safetensors.torch.save_file(model_checkpoint, args.dump_ckpt)

    print(f"Merged checkpoint saved to {args.dump_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a LoRA checkpoint into a model checkpoint."
    )
    parser.add_argument(
        "--initial_model_ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--lora_ckpt", type=str, required=True, help="Path to the LoRA checkpoint."
    )
    parser.add_argument(
        "--dump_ckpt",
        type=str,
        required=True,
        help="Path to save the merged checkpoint.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=2.0,
        help="Scaling factor for the LoRA checkpoint. Default is 2.0.",
    )

    args = parser.parse_args()
    main(args)
