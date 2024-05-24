import copy
import json
from pathlib import Path

import numpy as np
import pytest
from mistral_common.protocol.instruct.messages import FinetuningAssistantMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from finetune.data.args import DataArgs, InstructArgs
from finetune.data.data_loader import build_data_loader
from finetune.data.dataset import (
    DataFile,
    SampleType,
    get_dataset_iterator,
    get_rng,
    lazy_load_and_yield,
    maybe_chunk_lines,
    parse_data_sources,
    preload_and_yield,
)
from finetune.data.tokenize import build_instruct_sample, encode

from .test_utils import spawn_for_all_world_sizes

# fmt: off
EXPECTED_X = [
    [
        # for pretrain
        [
            # for DP=1
            [2051851, 1961139, 2000184, 2081307, 2341123, 1225437, 1739008, 724695, 570810, 632094]
        ],
        [
            # for DP=2
            [2020745, 1938377, 2244286, 2042079, 1824023],
            [2103241, 2032118, 1868430, 1093072, 770996],
        ]
    ],
    [
        # for instruct
        [
            # for DP=1
            [1379941, 1438894, 965536, 1019713, 889921, 999322, 1647173, 941080, 1281597, 1584884]
        ],
        [
            # for DP=2
            [1379941, 1438894, 889899, 1005451, 876854],
            [1034325, 999322, 982295, 941080, 725946],
        ]
    ]
]
EXPECTED_Y = [
    [
        # for pretrain
        [
            # for DP=1
            [2081367, 1961098, 1970714, 2110856, 2334822, 1251057, 1745267, 699854, 571600, 660015]
        ],
        [
            # for DP=2
            [2021840, 1966833, 2223275, 2063077, 1824011],
            [2132793, 2002569, 1870876, 1122569, 757126],
        ]
    ],
    [
        # for instruct
        [
            # for DP=1
            [1409448, 1430886, 937609, 1019339, 889921, 970976, 1660330, 942631, 1308399, 1583658]
        ],
        [
            # for DP=2
            [1409448, 1430886, 895531, 990091, 863522],
            [1041462, 970976, 991091, 942631, 737311]
        ]
    ]
]
EXPECTED_MASKS = [
    [
        # for pretrain
        [
            # for DP=1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            # for DP=2
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    ],
    [
        # for instruct
        [
            # for DP=1
            [47, 0, 34, 0, 0, 82, 0, 0, 0, 0]
        ],
        [
            # for DP=2
            [47, 0, 4, 0, 0],
            [19, 82, 0, 0, 23],
        ]
    ]
]

EXPECTED_X_FUNC = [
    [
        # for DP=1
        [1005531, 1551735, 1261711, 1531024, 1280259, 1069883, 858107, 1021583, 1203265, 1242999],
    ],
    [
        # for DP=2
        [985281, 1217766, 1442139, 1533790, 1253607],
        [1005531, 1551735, 1261711, 1531024, 1280259],
    ]
]

EXPECTED_Y_FUNC = [
    [
        # for DP=1
        [977126, 1580120, 1233326, 1559463, 1280241, 1042456, 879031, 994127, 1196263, 1270581],
    ],
    [
        # for DP=2
        [957934, 1218899, 1441783, 1533011, 1224541],
        [977126, 1580120, 1233326, 1559463, 1280241],
    ]
]

EXPECTED_MASKS_FUNC = [
    [
        # for DP=1
        [91, 0, 0, 0, 0, 77, 0, 0, 53, 0],
    ],
    [
        # for DP=2
        [16, 47, 0, 86, 98],
        [91, 0, 0, 0, 0],
    ]
]
# fmt: on


class MockTokenizer:
    def encode(self, content: str, *args, **kwargs) -> str:
        return content


class MockInstructTokenizerBaseBase:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def encode_user_message(self, message, *args, **kwargs):
        return message.content

    def encode_assistant_message(self, message, *args, **kwargs):
        return message.content

    def start(self):
        return []


def stringify(samples):
    lines = []
    for sample in samples:
        string_list = sample.tokens
        lines.append("".join(string_list))

    return lines


@pytest.mark.parametrize(
    ("world_size", "model_parallel", "is_instruct"),
    [
        (1, 1, False),
        (2, 1, False),
        (2, 2, False),
        (1, 1, True),
        (2, 1, True),
        (2, 2, True),
    ],
)
def test_data_loader_dist(world_size, model_parallel, is_instruct):
    spawn_for_all_world_sizes(
        _check_data_loader_dist,
        world_sizes=[world_size],
        args=[model_parallel, is_instruct],
        deterministic=True,
    )


def _check_data_loader_dist(
    rank: int,
    world_size: int,
    filename: str,
    filename_rpc: str,
    model_parallel: int,
    is_instruct: bool,
):
    dp_world_size = world_size // model_parallel
    dp_rank = rank // model_parallel

    seed = 0
    seq_len = 100
    batch_size = 1

    instruct = InstructArgs(shuffle=False, dynamic_chunk_fn_call=False)

    if is_instruct:
        # at the moment we have to pass some instuction finetuning
        data_args = DataArgs(
            data="",
            instruct_data="tests/fixtures/sample_instruct.jsonl:.1,tests/fixtures/sample_instruct_2.jsonl:.1,tests/fixtures/sample_instruct_3.jsonl:.1",
            instruct=instruct,
        )
    else:
        data_args = DataArgs(
            data="tests/fixtures/sample_pretrain_1.jsonl:1.0,tests/fixtures/sample_pretrain_2.jsonl:1.0",
            instruct_data="tests/fixtures/sample_instruct.jsonl:.01",
            instruct=instruct,
        )

    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    data_loader = build_data_loader(
        instruct_tokenizer,
        data_args,
        batch_size,
        seq_len,
        seed=seed,
        rank=dp_rank,
        world_size=dp_world_size,
        is_eval=False,
    )

    x_sums = []
    y_sums = []
    masks = []

    num_samples = 10 // dp_world_size

    for _ in range(num_samples):
        batch = next(data_loader)
        x_sums.append(batch.x.sum())
        y_sums.append(batch.y.sum())
        mask_sum = batch.y_mask.sum() if batch.y_mask is not None else 0
        masks.append(mask_sum)

    expected_x_sums = EXPECTED_X[is_instruct][dp_world_size - 1][dp_rank]
    expected_y_sums = EXPECTED_Y[is_instruct][dp_world_size - 1][dp_rank]
    expected_masks = EXPECTED_MASKS[is_instruct][dp_world_size - 1][dp_rank]

    print(f"rank: {rank}, world_size: {world_size}, x: {x_sums}")
    print(f"rank: {rank}, world_size: {world_size}, y: {y_sums}")
    print(f"rank: {rank}, world_size: {world_size}, x shape: {masks}")

    assert x_sums == expected_x_sums, x_sums
    assert y_sums == expected_y_sums, y_sums
    assert masks == expected_masks, masks


@pytest.mark.parametrize("world_size", [1, 2])
def test_data_loader_dist_fn_call(world_size):
    spawn_for_all_world_sizes(
        _check_data_loader_dist_fn_call,
        world_sizes=[world_size],
        deterministic=True,
    )


def _check_data_loader_dist_fn_call(
    rank: int,
    world_size: int,
    filename: str,
    filename_rpc: str,
):
    dp_world_size = world_size
    dp_rank = rank

    seed = 0
    seq_len = 100
    batch_size = 1

    data_args = DataArgs(
        data="",
        instruct_data="tests/fixtures/sample_instruct_fn_call_short.jsonl:.3",
        instruct=InstructArgs(shuffle=True, dynamic_chunk_fn_call=True),
    )

    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    data_loader = build_data_loader(
        instruct_tokenizer,
        data_args,
        batch_size,
        seq_len,
        seed=seed,
        rank=dp_rank,
        world_size=dp_world_size,
        is_eval=False,
    )

    x_sums = []
    y_sums = []
    masks = []

    num_samples = 10 // dp_world_size

    for _ in range(num_samples):
        batch = next(data_loader)
        x_sums.append(batch.x.sum())
        y_sums.append(batch.y.sum())
        mask_sum = batch.y_mask.sum() if batch.y_mask is not None else 0
        masks.append(mask_sum)

    expected_x_sums = EXPECTED_X_FUNC[dp_world_size - 1][dp_rank]
    expected_y_sums = EXPECTED_Y_FUNC[dp_world_size - 1][dp_rank]
    expected_masks = EXPECTED_MASKS_FUNC[dp_world_size - 1][dp_rank]

    assert x_sums == expected_x_sums, x_sums
    assert y_sums == expected_y_sums, y_sums
    assert masks == expected_masks, masks


def test_data_loader_equal_fsdp():
    spawn_for_all_world_sizes(
        _check_data_loader_equal_fsdp,
        world_sizes=[2],
        deterministic=True,
    )


def _check_data_loader_equal_fsdp(
    rank: int,
    world_size: int,
    filename: str,
    filename_rpc: str,
):
    model_parallel = 2
    world_size // model_parallel
    rank // model_parallel

    seed = 0
    seq_len = 100
    batch_size = 1

    instruct = InstructArgs(shuffle=False, dynamic_chunk_fn_call=False)

    data_args = DataArgs(
        data="",
        instruct_data="tests/fixtures/sample_instruct.jsonl:.1,tests/fixtures/sample_instruct_2.jsonl:.1,tests/fixtures/sample_instruct_3.jsonl:.1",
        instruct=instruct,
    )

    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    data_loader_0 = build_data_loader(
        instruct_tokenizer,
        data_args,
        batch_size,
        seq_len,
        seed=seed,
        rank=0,
        world_size=world_size,
        is_eval=False,
    )

    data_loader_1 = build_data_loader(
        instruct_tokenizer,
        data_args,
        batch_size,
        seq_len,
        seed=seed,
        rank=1,
        world_size=world_size,
        is_eval=False,
    )

    x_sums = []
    y_sums = []

    num_samples = 10 // 2  # run 5 * 2 training steps

    for _ in range(num_samples):
        batch = next(data_loader_0)
        x_sums.append(batch.x.sum())
        y_sums.append(batch.y.sum())

        batch = next(data_loader_1)
        x_sums.append(batch.x.sum())
        y_sums.append(batch.y.sum())

    # check that mp can match ddp for both ranks
    expected_x_sums = [
        y for x in zip(EXPECTED_X[1][1][0], EXPECTED_X[1][1][1]) for y in x
    ]
    expected_y_sums = [
        y for x in zip(EXPECTED_Y[1][1][0], EXPECTED_Y[1][1][1]) for y in x
    ]

    assert x_sums == expected_x_sums, x_sums
    assert y_sums == expected_y_sums, y_sums


def test_dynamic_fn_call_chunk():
    jsonl_file = Path("tests/fixtures/sample_instruct_fn_call_short.jsonl")

    non_chunked_samples = []
    with jsonl_file.open() as file_handle:
        for line in file_handle:
            non_chunked_samples.append(build_instruct_sample(json.loads(line)))

    num_expected_chunks = 0
    for sample in non_chunked_samples:
        if sample.only_last:
            num_expected_chunks += (
                sum(isinstance(m, FinetuningAssistantMessage) for m in sample.messages)
                - 1
            )

    chunked_samples = []
    with jsonl_file.open() as file_handle:
        lines = file_handle.readlines()
        extra_lines = maybe_chunk_lines(lines)

        for line in extra_lines:
            chunked_samples.append(build_instruct_sample(json.loads(line)))

    assert num_expected_chunks == len(chunked_samples)


def test_dynamic_fn_call_chunk_integration():
    jsonl_file = Path("tests/fixtures/sample_instruct_fn_call_multi.jsonl")

    multi_samples = []
    with jsonl_file.open() as file_handle:
        for line in file_handle:
            multi_samples.append(build_instruct_sample(json.loads(line)))

    jsonl_file = Path("tests/fixtures/sample_instruct_fn_call_single.jsonl")

    chunked_samples = []
    with jsonl_file.open() as file_handle:
        for line in file_handle:
            chunked_samples.append(build_instruct_sample(json.loads(line)))

    with jsonl_file.open() as file_handle:
        lines = file_handle.readlines()
        extra_lines = maybe_chunk_lines(lines)

        for line in extra_lines:
            chunked_samples.append(build_instruct_sample(json.loads(line)))

    assert list(reversed(multi_samples)) == chunked_samples


def test_fn_call():
    batch_size = 1
    data_args = DataArgs(
        data="",
        instruct_data="",
        eval_instruct_data="tests/fixtures/sample_instruct_fn_call.jsonl",
    )
    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    seq_len = 10000

    data_loader = build_data_loader(
        instruct_tokenizer,
        data_args,
        batch_size,
        seq_len,
        seed=None,
        rank=0,
        world_size=1,
        is_eval=True,
    )

    all_loss_strings = []
    for batch in data_loader:
        y_mask = (
            np.asarray(batch.y_mask, int)
            if batch.y_mask is not None
            else np.ones_like(batch.x)
        )

        start_index = end_index = 0
        for size in batch.sizes:
            end_index += size
            tokens = batch.y[start_index:end_index]
            mask = y_mask[start_index:end_index]

            tokens_for_loss = [int(y) for i, y in enumerate(tokens) if mask[i] == 1]
            start_index += size

            decoded = instruct_tokenizer.tokenizer.decode(tokens_for_loss)
            if len(decoded) > 0:
                all_loss_strings.append(decoded)

    # Verify that the loss is always only computed over the
    expected_loss_strings = []
    with open(data_args.eval_instruct_data, "r") as f:
        for line in f:
            data = json.loads(line)
            last_message = data["interactions"][-1]
            if "content" in last_message:
                expected_loss_strings.append(last_message["content"])
            elif "tool_calls" in last_message:
                tool_calls = last_message["tool_calls"]
                arguments = tool_calls[0]["function"]["arguments"]
                string = [
                    {
                        "name": call["function"]["name"],
                        "arguments": json.loads(arguments),
                    }
                    for call in tool_calls
                ]
                expected_loss_strings.append(json.dumps(string))

    assert expected_loss_strings == all_loss_strings


def test_data_weighting():
    data_args = DataArgs(
        data="",
        instruct_data="",
        eval_instruct_data="tests/fixtures/sample_instruct.jsonl",
    )

    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    jsonl_file = Path(data_args.eval_instruct_data)

    with jsonl_file.open() as file_handle:
        data = json.loads(next(file_handle))

    token_sample = encode(data, instruct_tokenizer, SampleType.INSTRUCT)

    data_weight_0 = copy.deepcopy(data)
    data_weight_0["interactions"][-1]["weight"] = 0

    token_sample_weight_0 = encode(
        data_weight_0, instruct_tokenizer, SampleType.INSTRUCT
    )

    data_weight_1 = copy.deepcopy(data)
    data_weight_1["interactions"][-1]["weight"] = 1

    token_sample_weight_1 = encode(
        data_weight_1, instruct_tokenizer, SampleType.INSTRUCT
    )

    assert (
        token_sample.tokens
        == token_sample_weight_0.tokens
        == token_sample_weight_1.tokens
    )
    assert token_sample.masks == token_sample_weight_1.masks
    assert token_sample.masks != token_sample_weight_0.masks
    assert not any(token_sample_weight_0.masks)


def test_eval_dataloader():
    batch_size = 1

    data_args = DataArgs(
        data="",
        instruct_data="",
        eval_instruct_data="tests/fixtures/sample_instruct.jsonl,tests/fixtures/sample_instruct_2.jsonl,tests/fixtures/sample_instruct_3.jsonl",
    )

    instruct_tokenizer = MistralTokenizer.v3().instruct_tokenizer

    # make sure that for every seq len the same data is seen
    for world_size in [1, 2, 8]:
        for seq_len in [10, 100, 1000, 10000]:
            x_sums = []
            y_sums = []
            y_masks = []

            data_loaders = []
            for rank in range(world_size):
                data_loaders.append(
                    build_data_loader(
                        instruct_tokenizer,
                        data_args,
                        batch_size,
                        seq_len,
                        seed=None,
                        rank=rank,
                        world_size=world_size,
                        is_eval=True,
                    )
                )

            for data_loader in data_loaders:
                for batch in data_loader:
                    mask = (
                        np.asarray(batch.y_mask, int)
                        if batch.y_mask is not None
                        else np.ones_like(batch.x)
                    )
                    x_sums.append((batch.x * mask).sum())
                    y_sums.append((batch.y * mask).sum())
                    y_masks.append(mask.sum())

                    assert len(batch.x) == len(mask) == len(batch.y) == seq_len

            assert sum(x_sums) == 71404835
            assert sum(y_sums) == 71404795
            assert sum(y_masks) == 5538


def test_shuffle_data():
    instruct_tokenizer = MockInstructTokenizerBaseBase()

    data_args = DataArgs(data="", instruct_data="", eval_instruct_data="")

    data_file = Path("tests/fixtures/sample_instruct_long_1.jsonl")

    dataset_iterator = get_dataset_iterator(
        source=DataFile(path=data_file, sample_type=SampleType.INSTRUCT),
        instruct_args=data_args.instruct,
        instruct_tokenizer=instruct_tokenizer,
        rank=0,
        world_size=1,
        is_finite=False,
        seed=0,
        shuffle_at_epoch=True,
    )

    with data_file.open() as f:
        lines = f.readlines()
        lines = [
            encode(
                json.loads(line),
                instruct_tokenizer=instruct_tokenizer,
                as_type=SampleType.INSTRUCT,
            )
            for line in lines
        ]
        prev_lines = stringify(lines)

    num_lines = len(prev_lines)

    samples = []
    # run 4 epochs
    for i in range(4 * num_lines):
        samples.append(next(dataset_iterator))

        if (i + 1) % num_lines == 0:
            # epoch finished!
            # check that order is different but all lines have the same hash
            lines = stringify(samples)
            assert lines != prev_lines, "No shuffling - make sure dataset is shuffled!"
            assert sorted(lines) == sorted(
                prev_lines
            ), "datasets need to match at every epoch"

            prev_lines = lines
            samples = []


@pytest.mark.parametrize("world_size", [1, 2])
def test_shuffle_data_same_as_no_shuffle(world_size):
    spawn_for_all_world_sizes(
        _check_shuffle_data_same_as_no_shuffle,
        world_sizes=[world_size],
        deterministic=True,
    )


def _check_shuffle_data_same_as_no_shuffle(
    rank: int,
    world_size: int,
    filename: str,
    filename_rpc: str,
):
    instruct_tokenizer = MockInstructTokenizerBaseBase()

    instruct = InstructArgs(shuffle=False, dynamic_chunk_fn_call=False)

    data_args = DataArgs(
        data="tests/fixtures/sample_pretrain_1.jsonl:1.0,tests/fixtures/sample_pretrain_2.jsonl:1.0",
        instruct_data="tests/fixtures/sample_instruct.jsonl:.1,tests/fixtures/sample_instruct_2.jsonl:.1,tests/fixtures/sample_instruct_3.jsonl:.1",
        instruct=instruct,
    )

    sources, _ = parse_data_sources(
        pretrain_data=data_args.data, instruct_data=data_args.instruct_data
    )

    seed = 0
    rng = get_rng(seed, rank)

    for source in sources:
        jsonl_files = source.jsonl_files
        chunk_dataset = source.sample_type == SampleType.INSTRUCT

        for jsonl_file in jsonl_files:
            samples = [[], []]
            for shuffle in [True, False]:
                print(jsonl_file)
                if shuffle:
                    iterator = preload_and_yield(
                        jsonl_file,
                        chunk_dataset=chunk_dataset,
                        rank=rank,
                        world_size=world_size,
                        rng=rng,
                        instruct_tokenizer=instruct_tokenizer,
                        sample_type=source.sample_type,
                    )
                else:
                    iterator = lazy_load_and_yield(
                        jsonl_file,
                        rank=rank,
                        world_size=world_size,
                        instruct_tokenizer=instruct_tokenizer,
                        sample_type=source.sample_type,
                    )

                for tokens in iterator:
                    samples[shuffle].append(tokens)

            strings_0 = sorted(stringify(samples[0]))
            strings_1 = sorted(stringify(samples[1]))

            assert strings_0 == strings_1
