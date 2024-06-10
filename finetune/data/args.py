import logging
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class InstructArgs(Serializable):
    shuffle: bool = True

    # For function calling training examples only the last tool call
    # of the assistant message can be used for training. Therefore,
    # we chunk longer function calling conversations into multiple
    # training samples to not lose any data. E.g.:
    # [[
    #   UserMessage_1, AssistantToolCallMessage_1, ToolMessage_1, AssistantMessage_1
    #   UserMessage_2, AssistantToolCallMessage_2, ToolMessage_2, AssistantMessage_2
    # ]]
    # => is chunked into two training samples:
    # [[
    #   UserMessage_1, AssistantToolCallMessage_1, ToolMessage_1, AssistantMessage_1
    # ],
    # [
    #   UserMessage_1, AssistantToolCallMessage_1, ToolMessage_1, AssistantMessage_1
    #   UserMessage_2, AssistantToolCallMessage_2, ToolMessage_2, AssistantMessage_2
    # ]]
    # NOTE: Only if your data is already pre-chunked should this argument be set to False
    dynamic_chunk_fn_call: bool = True


@dataclass()
class DataArgs(Serializable):
    # The data arguments `data` and `instruct_data` are a string in the format
    # "data_source_dir_1:weight_1,data_source_dir_2:weight_2,...". The weight
    # will be used to sample the data sources. If the sum of the weights is
    # not 1 when concatenating the two arguments `data` and `instruct_data`,
    # it will be normalized. The data sources folders must contain jsonl files.
    # If the value is an empty string, no data will be used for the corresponding
    # data type.
    data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key. See Readme for more details. Can be left empty.
    )
    shuffle: bool = False
    instruct_data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "interactions" key. See Readme for more details. Can be left empty.
    )
    eval_instruct_data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "interactions" key. See Readme for more details. Can be left empty.
    )
    instruct: InstructArgs = field(default_factory=InstructArgs)

    def __post_init__(self) -> None:
        if (
            self.instruct.shuffle is False
            and self.instruct.dynamic_chunk_fn_call is True
        ):
            raise ValueError(
                "Make sure to either enable `data.instruct.shuffle=True` or `data.instruct.dynamic_chunk_fn_call=False`. Dynamic chunking is only possible if data is loaded and shuffled before training."
            )
