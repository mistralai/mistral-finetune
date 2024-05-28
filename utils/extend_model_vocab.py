import argparse
import json
import math
import os
from pathlib import Path

import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer

from model.args import ModelArgs

FIRST_PIECE_ID = 3
OLD_VOCAB_SIZE = 32000
NEW_VOCAB_SIZE = 32768


def extend_model(original_model: Path, extended_model: Path):
    original_ckpt = torch.load(str(original_model / "consolidated.00.pth"), mmap=True)
    model_args = ModelArgs.load(str(original_model / "params.json"))

    original_vocab_size = model_args.vocab_size
    assert (
        original_vocab_size == OLD_VOCAB_SIZE
    ), f"Original vocab size {original_vocab_size} is not equal to 32000. Can only extend models with vocab_size of 32000"

    if not extended_model.exists():
        os.makedirs(extended_model, exist_ok=True)
        print(f"Created empty directory {extended_model}.")

    assert not list(
        extended_model.iterdir()
    ), f"Make sure {extended_model} is empty"

    # Load and check tokenizers
    mistral_tokenizer = MistralTokenizer.v3()
    tokenizer: SentencePieceTokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    new_vocab_size = tokenizer.n_words
    assert (
        new_vocab_size == 32768
    ), f"New Tokenizer has vocab_size: {new_vocab_size} but has to be equal to 32768. Make sure to pass a v2 or v3 tokenizer file"

    vocabulary_delta = new_vocab_size - original_vocab_size

    # Check that 0...FIRST_PIECE_ID-1 are UNK + control characters and FIRST_PIECE_ID is the first piece
    assert tokenizer._model.id_to_piece(vocabulary_delta + FIRST_PIECE_ID) == "<0x00>"
    assert tokenizer._model.id_to_piece(FIRST_PIECE_ID - 1) == "</s>"

    assert isinstance(tokenizer, SentencePieceTokenizer)

    original_embeddings = original_ckpt["tok_embeddings.weight"]

    assert (
        original_vocab_size == original_embeddings.shape[0]
    ), f"Original vocab size {original_vocab_size} is not equal to original embeddings shape {original_embeddings.shape[0]}."

    dim = original_embeddings.shape[1]

    # Extend embeddings
    extended_embeddings = torch.zeros(
        tokenizer.n_words, dim, dtype=original_embeddings.dtype
    )
    extended_embeddings[:original_vocab_size] = original_embeddings
    extended_embeddings[:FIRST_PIECE_ID] = original_embeddings[:FIRST_PIECE_ID]
    extended_embeddings[FIRST_PIECE_ID + vocabulary_delta :] = original_embeddings[
        FIRST_PIECE_ID:
    ]

    # randomly initialize new tokens
    extended_tokens = torch.empty(
        vocabulary_delta, dim, dtype=original_embeddings.dtype
    )
    torch.nn.init.normal_(extended_tokens, std=1 / math.sqrt(dim))

    extended_embeddings[FIRST_PIECE_ID : FIRST_PIECE_ID + vocabulary_delta] = (
        extended_tokens
    )

    # Extend output
    original_output = original_ckpt["output.weight"]
    assert (
        original_output.shape[0] == original_vocab_size
    ), f"Original output shape {original_output.shape[0]} is not equal to {original_vocab_size}."
    assert (
        original_output.shape[1] == dim
    ), f"Original output dim {original_output.shape[1]} is not equal to embedding dim {dim}."

    assert (
        original_output.dtype == original_embeddings.dtype
    ), f"Original output and embeddings have different dtypes: {original_output.dtype} vs {original_embeddings.dtype}."

    extended_output = torch.zeros(tokenizer.n_words, dim, dtype=original_output.dtype)
    extended_output[:FIRST_PIECE_ID] = original_output[:FIRST_PIECE_ID]
    extended_output[FIRST_PIECE_ID + vocabulary_delta :] = original_output[
        FIRST_PIECE_ID:
    ]

    # randomly initialize new tokens
    extended_tokens = torch.empty(vocabulary_delta, dim, dtype=original_output.dtype)
    torch.nn.init.normal_(extended_tokens, std=1 / math.sqrt(dim))

    extended_output[FIRST_PIECE_ID : FIRST_PIECE_ID + vocabulary_delta] = (
        extended_tokens
    )

    original_ckpt["tok_embeddings.weight"] = extended_embeddings
    original_ckpt["output.weight"] = extended_output

    new_ckpt_path = extended_model / "consolidated.00.pth"
    print(f"Exporting extended model to {extended_model} ...")
    torch.save(original_ckpt, new_ckpt_path)

    params_path = extended_model / "params.json"
    with open(params_path, "w") as f:
        model_dict = model_args.to_dict()
        del model_dict["lora"]
        if model_dict["moe"] is None:
            del model_dict["moe"]
        model_dict["vocab_size"] = new_vocab_size

        f.write(json.dumps(model_dict, indent=4))


def main():
    parser = argparse.ArgumentParser(
        description="Extend a model using the specified original model, extended model, and tokenizer paths."
    )
    parser.add_argument(
        "--original_model_ckpt", type=Path, help="Path to the original model folder."
    )
    parser.add_argument(
        "--extended_model_ckpt", type=Path, help="Path to the extended model file."
    )
    args = parser.parse_args()

    extend_model(
        original_model=args.original_model_ckpt,
        extended_model=args.extended_model_ckpt,
    )


if __name__ == "__main__":
    main()
