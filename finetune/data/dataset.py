import dataclasses
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import torch.distributed as dist
from mistral_common.protocol.instruct.messages import (
    FinetuningAssistantMessage,
    SystemMessage,
)
from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerBase

from finetune.distributed import get_rank

from .args import InstructArgs
from .tokenize import (
    Mask,
    SampleType,
    TokenSample,
    TrainingInstructSample,
    build_instruct_sample,
    encode,
)

logger = logging.getLogger("dataset")


_LOADED_DATASETS: Dict[Path, List[TokenSample]] = {}


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def load_file(path: Path, world_size: int, rank: int) -> List[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not idx % world_size == rank:
                continue
            lines.append(line)
    return lines


def maybe_load_local_dataset(
    path: Path, chunk: bool, rank: int, world_size: int, instruct_tokenizer: InstructTokenizerBase, sample_type: SampleType
) -> List[TokenSample]:
    global _LOADED_DATASETS

    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    main_logger_info(f"Loading {path} ...")
    lines: List[str] = load_file(path, rank=rank, world_size=world_size)

    if chunk:
        lines += maybe_chunk_lines(lines)

    tokens_list: List[TokenSample] = []
    for line in lines:
        data = json.loads(line)

        token_sample: TokenSample = encode(
            data,
            instruct_tokenizer=instruct_tokenizer,
            as_type=sample_type,
        )
        tokens_list.append(token_sample)

    main_logger_info(f"{path} loaded and tokenized.")
    _LOADED_DATASETS[path] = tokens_list

    return _LOADED_DATASETS[path]


@dataclass
class DataDir:
    path: Path
    sample_type: SampleType

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        jsonl_files = list(self.path.rglob("*jsonl"))
        assert (
            len(jsonl_files) > 0
        ), f"{self.path} does not seem to have any files ending with '.jsonl'"
        return jsonl_files


@dataclass
class DataFile:
    path: Path
    sample_type: SampleType

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        return [self.path]


def parse_data_sources(
    pretrain_data: str,
    instruct_data: str,
) -> Tuple[List[Union[DataDir, DataFile]], List[float]]:
    seen: Set[str] = set()
    sources: List[Union[DataDir, DataFile]] = []
    weights: List[float] = []
    for sample_sources, sample_type in [
        (pretrain_data, SampleType.PRETRAIN),
        (instruct_data, SampleType.INSTRUCT),
    ]:
        for source in sample_sources.strip().split(","):
            if not source:
                continue

            source_items = source.strip().split(":")
            if len(source_items) == 1:
                path_ = source_items[0]
                weight = 1.0
            elif len(source_items) == 2:
                path_, weight_ = source_items
                weight = float(weight_)
            else:
                raise ValueError(
                    f"{source} is not correctly formatted. Make sure to format each data source as <path/to/data>:<weight> or just <path/to/data>"
                )

            assert (
                path_ not in seen
            ), f"{path_} seems to be duplicated. Make sure to only add it once."
            assert (
                weight > 0
            ), f"Make sure to define strictly positive data sampling weights, not {weight}"

            data: Union[DataDir, DataFile]
            if Path(path_).is_dir():
                data = DataDir(path=Path(path_), sample_type=sample_type)
            elif Path(path_).is_file():
                data = DataFile(path=Path(path_), sample_type=sample_type)
            else:
                raise FileNotFoundError(
                    f"The path {path_} does not exist. Make sure {path_} is either a file or directory that contains training data."
                )

            sources.append(data)
            weights.append(weight)

            seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert (
        abs(1 - sum(n_weights)) < 1e-8
    ), f"Defined data sampling weights {weights} must sum to 1."
    return sources, n_weights


@dataclasses.dataclass()
class SequenceMaskAndSizes:
    """
    Concatenation of samples to reach a given size
    """

    x: List[int]
    y: List[int]
    mask: Mask
    sizes: List[int]

    def __post_init__(self):
        assert sum(self.sizes) == len(self.x) == len(self.y) == len(self.mask)


def sequence_iterator(
    ds_it: Iterator[TokenSample],
    seq_len: int,
    is_finite: bool,
) -> Iterator[SequenceMaskAndSizes]:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """
    x_buffer: List[int] = []
    y_buffer: List[int] = []
    mask_buffer: Mask = []

    sizes: List[int] = []
    n_missing = seq_len
    for sample in ds_it:
        assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
        assert n_missing == seq_len - len(
            x_buffer
        ), f"n_missing: {n_missing} | seq_len - len(x_buffer) {seq_len - len(x_buffer)}"

        tokens, mask = sample.tokens, sample.masks[1:]
        x, y = tokens[:-1], tokens[1:]
        cur_pos = 0

        while cur_pos < len(x):
            size = len(x[cur_pos : cur_pos + n_missing])

            curr_mask = mask[cur_pos : cur_pos + n_missing]
            if not any(curr_mask):
                cur_pos += size
                # we have a sequence with a mask filled with False
                continue

            x_buffer.extend(x[cur_pos : cur_pos + n_missing])
            y_buffer.extend(y[cur_pos : cur_pos + n_missing])
            mask_buffer.extend(curr_mask)
            n_missing -= size
            sizes.append(size)

            cur_pos += size

            if n_missing == 0:
                assert len(mask_buffer) == len(x_buffer) == seq_len == len(y_buffer)
                assert sum(sizes) == seq_len
                # we don't want to yield sequences with a mask filled with False
                if any(mask_buffer):
                    yield SequenceMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                    )
                x_buffer, y_buffer = [], []
                mask_buffer = []
                sizes = []
                n_missing = seq_len

    if is_finite:
        # if dataloader is in eval, pad to seq length
        if any(mask_buffer):
            mask_buffer.extend(n_missing * [False])
            x_buffer.extend(n_missing * [0])
            y_buffer.extend(n_missing * [0])
            sizes.append(n_missing)

            yield SequenceMaskAndSizes(
                x=x_buffer,
                y=y_buffer,
                mask=mask_buffer,
                sizes=sizes,
            )


def build_dataset(
    pretrain_data: str,
    instruct_data: str,
    instruct_args: InstructArgs,
    instruct_tokenizer: InstructTokenizerBase,
    seq_len: int,
    seed: Optional[int],
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle_pretrain: bool = False,
) -> Iterator[SequenceMaskAndSizes]:
    sources, probabilities = parse_data_sources(
        pretrain_data=pretrain_data, instruct_data=instruct_data
    )

    def do_shuffle(source: Union[DataDir, DataFile]) -> bool:
        shuffle = {
            SampleType.PRETRAIN: shuffle_pretrain,
            SampleType.INSTRUCT: instruct_args.shuffle,
        }[source.sample_type]

        return not is_eval and shuffle

    dataset_iterators = [
        get_dataset_iterator(
            source,
            instruct_args=instruct_args,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=do_shuffle(source),
        )
        for source in sources
    ]

    sequence_iterators = [
        sequence_iterator(
            ds_it=it,
            seq_len=seq_len,
            is_finite=is_eval,
        )
        for it in dataset_iterators
    ]

    if is_eval:
        combined_iterator = itertools.chain.from_iterable(sequence_iterators)
    else:
        # make sure random_seed is different per rank and original seed
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            sequence_iterators, probabilities=probabilities, rng=rng
        )

    return combined_iterator


def get_rng(seed: int, rank: int) -> np.random.RandomState:
    random_seed = np.array((seed, rank))
    rng = np.random.RandomState(seed=random_seed)
    return rng


def get_dataset_iterator(
    source: Union[DataDir, DataFile],
    instruct_args: InstructArgs,
    instruct_tokenizer: InstructTokenizerBase,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: Optional[int],
    shuffle_at_epoch: bool,
) -> Iterator[TokenSample]:
    jsonl_files = source.jsonl_files
    rng: Optional[np.random.RandomState] = (
        get_rng(seed, rank) if seed is not None else None
    )

    chunk_dataset = (
        instruct_args.dynamic_chunk_fn_call
        and source.sample_type == SampleType.INSTRUCT
    )

    if not is_finite:
        # train mode
        while True:
            for jsonl_file in jsonl_files:
                if shuffle_at_epoch:
                    assert rng is not None, "`seed` has to be passed when shuffling"
                    # will preload all data into RAM, shuffle and yield
                    yield from preload_and_yield(
                        jsonl_file,
                        chunk_dataset=chunk_dataset,
                        rank=rank,
                        world_size=world_size,
                        rng=rng,
                        instruct_tokenizer=instruct_tokenizer,
                        sample_type=source.sample_type,
                    )
                else:
                    # will read data on-the-fly and yield
                    main_logger_info(f"Lazily loading {jsonl_file} ...")
                    yield from lazy_load_and_yield(
                        jsonl_file,
                        rank=rank,
                        world_size=world_size,
                        instruct_tokenizer=instruct_tokenizer,
                        sample_type=source.sample_type,
                    )
    else:
        # eval mode
        for jsonl_file in jsonl_files:
            # No need to shuffle for eval
            yield from lazy_load_and_yield(
                jsonl_file,
                rank=rank,
                world_size=world_size,
                instruct_tokenizer=instruct_tokenizer,
                sample_type=source.sample_type,
            )


def preload_and_yield(
    jsonl_file: Path,
    chunk_dataset: bool,
    rank: int,
    world_size: int,
    rng: np.random.RandomState,
    instruct_tokenizer: InstructTokenizerBase,
    sample_type: SampleType,
) -> Iterator[TokenSample]:
    # only instruct data has to be chunked
    # load dataset if not already loaded. Make sure to only load 1/world_size dataset
    tokens_list = maybe_load_local_dataset(
        jsonl_file, chunk=chunk_dataset, rank=rank, world_size=world_size, instruct_tokenizer=instruct_tokenizer, sample_type=sample_type
    )

    if sample_type == SampleType.PRETRAIN:
        assert chunk_dataset is False, "Pretrain data should not have chunking enabled."

    main_logger_info(f"Shuffling {jsonl_file} ...")
    rng.shuffle(tokens_list)  # type: ignore

    for token_sample in tokens_list:
        yield token_sample

def lazy_load_and_yield(
    jsonl_file: Path,
    rank: int,
    world_size: int,
    instruct_tokenizer: InstructTokenizerBase,
    sample_type: SampleType,
):
    with jsonl_file.open() as file_handle:
        for idx, line in enumerate(file_handle):
            if not idx % world_size == rank:
                continue

            data = json.loads(line)
            yield encode(
                data,
                instruct_tokenizer=instruct_tokenizer,
                as_type=sample_type,
            )


def maybe_chunk_lines(lines: List[str]) -> List[str]:
    extra_lines: List[str] = []
    for line in lines:
        data = json.loads(line)
        # multi-turn fn call data will be chunked and shorter conversations are added additionally
        maybe_chunked_lines = maybe_chunk_data(data)
        extra_lines.extend([json.dumps(line) for line in maybe_chunked_lines])

    return extra_lines


def maybe_chunk_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # think about always allowing both open-ai and non-open-ai data
    sample = build_instruct_sample(data)

    def num_assistant_messages(sample: TrainingInstructSample) -> int:
        return len(
            [m for m in sample.messages if isinstance(m, FinetuningAssistantMessage)]
        )

    chunk_data = []
    while sample.only_last is True and num_assistant_messages(sample) > 1:
        assert sample == build_instruct_sample(sample.dict())
        last_message = sample.messages.pop()

        # 1. First pop until and including last assistant message
        system_message = None
        while not isinstance(last_message, FinetuningAssistantMessage):
            last_message = sample.messages.pop()
            if isinstance(last_message, SystemMessage):
                system_message = last_message

        # 2. Second pop until and excluding last assistant message
        prev_last_message = sample.messages[-1]
        while not isinstance(prev_last_message, FinetuningAssistantMessage):
            last_message = sample.messages.pop()
            if isinstance(last_message, SystemMessage):
                system_message = last_message

            prev_last_message = sample.messages[-1]

        # if system_message is not None, append again
        if system_message is not None:
            sample.messages.append(system_message)
        chunk_data.append(sample.dict())

    return chunk_data


def interleave_iterators(iterators: List[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
