import argparse
import json
from pathlib import Path
from typing import Dict

from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidFunctionCallException,
    InvalidMessageStructureException,
    InvalidToolSchemaException,
    TokenizerException,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tqdm import tqdm

from finetune.args import TrainArgs
from finetune.data.dataset import parse_data_sources
from finetune.data.tokenize import (
    ConversationFormatError,
    FunctionFormatError,
    MessageFormatError,
    SampleType,
    ToolCallFormatError,
    UnrecognizedRoleError,
    build_instruct_sample,
    get_pretrain_sample,
    tokenize,
)

NUM_GPUS = 8

# EXPECTED WPS for batch_size = 32768 per GPU on H100
EXPECTED_WPS = {
    "open-mistral-7b": 5720,
    "open-mixtral-8x7b": 2966,
    "open-mixtral-8x22b": 1007,
}

MIN_NUM_JSONL_LINES = 10
MAX_NUM_JSONL_LINES = 10_000_000

MIN_BYTES = 1_000
MAX_BYTES = 10_000_000_000  # rougly 10 GB


def convert_seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60

    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def verify_size(jsonl_file: Path):
    print(f"Verifying {jsonl_file} ...")
    with jsonl_file.open() as f:
        num_lines = 0
        num_bytes = 0
        for line in f:
            num_lines += 1
            num_bytes += len(line)

    if num_lines < MIN_NUM_JSONL_LINES:
        raise ValueError(
            f"{jsonl_file} has only {num_lines} conversation which is less than the minimum amount of conversations required per dataset file: {MIN_NUM_JSONL_LINES}. Please make sure that each dataset has at least {MIN_NUM_JSONL_LINES} conversations."
        )
    elif num_bytes < MIN_BYTES:
        raise ValueError(
            f"{jsonl_file} has only {num_bytes} bytes which is less than the minimum amount of bytes required per dataset file: {MIN_BYTES}. Please make sure that each dataset has at least {MIN_BYTES} bytes."
        )
    elif num_lines > MAX_NUM_JSONL_LINES:
        raise ValueError(
            f"{jsonl_file} has {num_lines} conversation which is more than the maximum amount of allowed per dataset file: {MAX_NUM_JSONL_LINES}. Please make sure that each dataset has no more than {MAX_NUM_JSONL_LINES} conversations."
        )
    elif num_bytes > MAX_BYTES:
        raise ValueError(
            f"{jsonl_file} has {num_bytes} bytes which is more than the maximum amount of bytes allowed per dataset file: {MAX_BYTES}. Please make sure that each dataset has no more than {MAX_BYTES} bytes."
        )

    print(
        f"Dataset {jsonl_file} is valid. Dataset has {num_lines} conversations amounting to a total of {num_bytes} bytes."
    )


def get_train_stats(
    num_tokens: Dict[str, int],
    datasets_proportion: Dict[str, float],
    train_args: TrainArgs,
    return_type: str,
):
    dataset_tokens = sum(num_tokens.values())
    batch_size = train_args.batch_size * train_args.seq_len * NUM_GPUS

    if Path(train_args.model_id_or_path).is_dir():
        params_config = json.load(
            (Path(train_args.model_id_or_path) / "params.json").open()
        )

        if params_config["dim"] == 4096 and params_config.get("moe") is None:
            model_id = "open-mistral-7b"
        elif params_config["dim"] == 4096 and params_config.get("moe") is not None:
            model_id = "open-mixtral-8x7b"
        elif params_config["dim"] == 6144:
            model_id = "open-mixtral-8x22b"
        else:
            raise ValueError("Provided model folder seems incorrect.")
    else:
        model_id = train_args.model_id_or_path

    wps = EXPECTED_WPS[model_id]

    if return_type == "expected":
        train_tokens = train_args.max_steps * batch_size
        max_steps = train_args.max_steps
        num_epochs = train_tokens / dataset_tokens
    elif return_type == "recommended":
        num_epochs = 3
        max_steps = int(sum(num_tokens.values()) / batch_size * num_epochs)
        train_tokens = max_steps * batch_size
    else:
        raise ValueError(
            f"`return_type` is {return_type}, but has to be one of ['expected', 'recommended']"
        )

    expected_time_in_sec = train_tokens / NUM_GPUS / wps

    # Add 5min buffer for loading/init/ckpt/eval
    expected_time_in_sec += 300

    train_tokens_per_dataset = {
        k: (train_tokens * v) for k, v in datasets_proportion.items()
    }

    return {
        "eta": convert_seconds_to_hms(expected_time_in_sec),
        "data_tokens": dataset_tokens,
        "train_tokens": train_tokens,
        "epochs": f"{num_epochs:.2f}",
        "max_steps": max_steps,
        "data_tokens_per_dataset": {k: f"{v:.1f}" for k, v in num_tokens.items()},
        "train_tokens_per_dataset": {
            k: f"{v:.1f}" for k, v in train_tokens_per_dataset.items()
        },
        "epochs_per_dataset": {
            k: f"{(train_tokens_per_dataset[k] / num_tokens[k]):.1f}"
            for k in num_tokens.keys()
        },
    }


def main(args):
    train_args = TrainArgs.load(args.train_yaml)

    yaml_data_errors = []
    conversation_format_errors = []
    message_format_errors = []
    tokenization_errors = []

    # Check if pretrain can be loaded
    # train_pretrain_data = train_args.data.data
    data = [("train", train_args.data.data, train_args.data.instruct_data)]

    if train_args.data.eval_instruct_data != "":
        data.append(("eval", "", train_args.data.eval_instruct_data))

    EXPECTED_WPS.keys()

    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    for name, pretrain_file, instruct_file in data:
        datasets, weights = parse_data_sources(pretrain_file, instruct_file)
        data_types = [d.sample_type for d in datasets]
        datasets = [str(d.path) for d in datasets]

        datasets_proportion = dict(zip(datasets, weights))
        num_tokens = {k: 0 for k in datasets_proportion}

        for data_type, dataset in tqdm(zip(data_types, datasets)):
            # verify_size(Path(dataset))
            print(f"Validating {dataset} ...")

            corrected_dataset = dataset + ".corrected"
            correct_lines = []

            sub_yaml_data_errors = []
            sub_conversation_format_errors = []
            sub_message_format_errors = []
            sub_tokenization_errors = []

            # Load the dataset
            with open(dataset, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for idx, line in tqdm(enumerate(lines), total=len(lines)):
                    try:
                        data = json.loads(line)
                    except ValueError as e:
                        prefix = f"The data in line {idx + 1} of dataset {dataset} is incorrectly formatted."
                        sub_yaml_data_errors.append(prefix + str(e))
                        continue

                    if data_type == SampleType.PRETRAIN:
                        # TODO(Patrick) - Get good error message
                        sample = get_pretrain_sample(data)
                    else:
                        try:
                            sample = build_instruct_sample(data)

                        except (
                            ConversationFormatError,
                            UnrecognizedRoleError,
                            MessageFormatError,
                            ToolCallFormatError,
                            FunctionFormatError,
                            InvalidAssistantMessageException,
                            InvalidFunctionCallException,
                            InvalidMessageStructureException,
                            InvalidToolSchemaException,
                        ) as e:
                            prefix = f"The data in line {idx + 1} of dataset {dataset} is incorrectly formatted."
                            if isinstance(
                                e, (ConversationFormatError, FunctionFormatError)
                            ):
                                sub_conversation_format_errors.append(prefix + str(e))
                            elif isinstance(
                                e,
                                (
                                    MessageFormatError,
                                    UnrecognizedRoleError,
                                    ToolCallFormatError,
                                ),
                            ):
                                sub_message_format_errors.append(prefix + str(e))
                            if isinstance(
                                e,
                                (
                                    InvalidFunctionCallException,
                                    InvalidMessageStructureException,
                                    InvalidAssistantMessageException,
                                    InvalidToolSchemaException,
                                ),
                            ):
                                sub_conversation_format_errors.append(prefix + str(e))

                            continue
                    try:
                        tokens = tokenize(sample, instruct_tokenizer).tokens
                    except TokenizerException as e:
                        error_message = (
                            f"The data in line {idx + 1} of dataset {dataset} could not be tokenized. "
                            + str(e)
                        )
                        sub_tokenization_errors.append(error_message)

                    correct_lines.append(line)
                    num_tokens[dataset] += len(tokens)

            is_sub_error = (
                len(
                    sub_yaml_data_errors
                    + sub_conversation_format_errors
                    + sub_message_format_errors
                    + sub_tokenization_errors
                )
                > 0
            )
            if is_sub_error and args.create_corrected:
                with open(corrected_dataset, "w", encoding="utf-8") as f:
                    for line in correct_lines:
                        f.write(line)

                print(f"Saved {corrected_dataset}.")
            elif args.create_corrected:
                print(f"No error in {dataset} - no need to create a corrected version.")

        yaml_data_errors.extend(sub_yaml_data_errors)
        conversation_format_errors.extend(sub_conversation_format_errors)
        message_format_errors.extend(sub_message_format_errors)
        tokenization_errors.extend(sub_tokenization_errors)

        is_error = (
            len(
                yaml_data_errors
                + conversation_format_errors
                + message_format_errors
                + tokenization_errors
            )
            > 0
        )
        if is_error:
            all_yaml_data_errors = "\n".join(yaml_data_errors)
            all_conversation_format_errors = "\n".join(conversation_format_errors)
            all_message_format_errors = "\n".join(message_format_errors)
            all_tokenization_errors = "\n".join(tokenization_errors)
            error_report = f"""
                Data error report
                ----------------------- \n
                The passed datasets contains some errors as listed below. Please make sure to fix these errors in order to start training.

                YAML data load errors: \n\n {all_yaml_data_errors} \n\n
                Conversation format errors: \n\n {all_conversation_format_errors} \n\n
                Message format errors: \n\n {all_message_format_errors} \n\n
                Tokenization errors: \n\n {all_tokenization_errors} \n\n
            """
            if args.save_reports:
                with open(args.error_report_txt, "w") as f:
                    f.write(error_report)

            print(error_report)
        else:
            print("No errors! Data is correctly formatted!")

        if name == "train" and not is_error:
            expected_stats = get_train_stats(
                num_tokens, datasets_proportion, train_args, return_type="expected"
            )
            stats = {
                "expected": expected_stats,
            }

            filenames = (
                f"{instruct_file}"
                if pretrain_file == ""
                else f"{instruct_file} and {pretrain_file}"
            )

            print(
                f"Stats for {filenames} \n {20 * '-'} \n {json.dumps(stats, indent=4)}"
            )

    if args.save_reports:
        if name == "train":
            with open(args.train_stats_json, "w") as file:
                json.dump(stats, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate jsonl.")
    parser.add_argument(
        "--train_yaml",
        type=str,
        help="Path to the data file",
    )
    parser.add_argument(
        "--error_report_txt",
        type=str,
        default="data_errors.txt",
        help="Path to the error report.",
    )
    parser.add_argument(
        "--train_stats_json",
        type=str,
        default="train_stats.json",
        help="Path to training statistics json file.",
    )
    parser.add_argument(
        "--save_reports", action="store_true", help="Save reports to disk"
    )
    parser.add_argument(
        "--create_corrected",
        action="store_true",
        help="Skip faulty lines and append all correct lines to `.corrected` datasets.",
    )
    args = parser.parse_args()
    main(args)
