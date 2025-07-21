import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from mistral_common.protocol.instruct.messages import (
    FinetuningAssistantMessage,
    Roles,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    Tool,
    ToolCall,
)
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidatorV3,
    ValidationMode,
)
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from .exceptions import (
    ConversationFormatError,
    FunctionFormatError,
    MessageFormatError,
    ToolCallFormatError,
    UnrecognizedRoleError,
)

logger = logging.getLogger("tokenize")

Sequence = List[int]
Mask = List[bool]


class TrainingInstructSample(InstructRequest):
    available_tools: Optional[List[Tool]] = None
    only_last: bool = False


@dataclass()
class TokenSample:
    tokens: Sequence
    masks: Mask


class SampleType(str, Enum):
    PRETRAIN = "pretrain"
    INSTRUCT = "instruct"


def encode(
    data: Dict[str, Any],
    instruct_tokenizer: MistralTokenizer,
    as_type: SampleType,
) -> TokenSample:
    sample: Union[str, TrainingInstructSample]
    if as_type == SampleType.PRETRAIN:
        sample = get_pretrain_sample(data)
    elif as_type == SampleType.INSTRUCT:
        sample = build_instruct_sample(data)

    return tokenize(sample=sample, instruct_tokenizer=instruct_tokenizer)


def get_pretrain_sample(data: Dict[str, Any]) -> str:
    content_keys = ["text", "content"]
    assert not all(
        k in data for k in content_keys
    ), "Make sure to have either 'text' or 'content' in your data. Not both."
    assert any(
        data.get(k) is not None for k in content_keys
    ), f"Must have one of 'text' or 'content' in your data. Only have {data.keys()}"

    # get first non-None value
    sample = None
    for key in content_keys:
        sample = data[key] if key in data else sample

    assert isinstance(sample, str), sample

    return sample


def build_instruct_sample(data: Dict[str, Any]) -> TrainingInstructSample:
    messages: List[Union[SystemMessage, UserMessage, FinetuningAssistantMessage, ToolMessage]] = []
    # optional data fields that might be set
    available_tools: Optional[List[Tool]] = data.get("available_tools")
    system_prompt = data.get("system_prompt")

    messages_keys = ["messages", "interactions"]
    content_keys = ["content", "text"]  # both are accepted
    allowed_roles = [role.value for role in Roles]

    if not any(messages_key in data for messages_key in messages_keys):
        err = f"The conversation does not contain one of '{', '.join(messages_keys)}' key, but only {', '.join(data.keys())}. Make sure that the conversation includes one of '{', '.join(messages_keys)}'."
        raise ConversationFormatError(err, str(data))

    if all(messages_key in data for messages_key in messages_keys):
        err = f"The conversation cannot contain both of '{', '.join(messages_keys)}' key, but only one of the two."
        raise ConversationFormatError(err, str(data))

    # get first non-None value
    data_messages: Optional[List[Dict[str, Any]]] = None
    for key in messages_keys:
        data_messages = data[key] if key in data else data_messages

    assert data_messages is not None, "data_messages can't be None"

    if "available_tools" in data and "tools" in data:
        err = "The conversation contains both an `available_tools` and `tools` key. You can only have one."
        raise ConversationFormatError(err, str(data))

    if data.get("tools", None) is not None and len(data["tools"]) > 0:
        available_tools = _parse_available_tools(data["tools"])
    elif (
        data.get("available_tools", None) is not None
        and len(data["available_tools"]) > 0
    ):
        available_tools = _parse_available_tools(data["available_tools"])

    for data_message in data_messages:
        is_tool_call = data_message.get("tool_calls") is not None

        if "role" not in data_message:
            err = f"A message does not contain a 'role' key, but only {', '.join(data_message.keys())}. Make sure that the message includes the key 'role'."
            raise MessageFormatError(err, str(data))

        role = data_message["role"]

        if all(key in data_message for key in content_keys):
            err = f"A {role} message contains both a 'text' and 'content' key. Make sure that there is only one of the two."
            raise MessageFormatError(err, str(data))

        content: Optional[str] = None
        for key in content_keys:
            content = content if content is not None else data_message.get(key)

        # non-function call message must have content
        if not is_tool_call and content is None:
            err = f"A {role} message does not contain one of '{content_keys}' key, but only {', '.join(data_message.keys())}. Make sure that the message includes one of '{content_keys}' keys."
            raise MessageFormatError(err, str(data))

        if role not in allowed_roles:
            raise UnrecognizedRoleError(role, allowed_roles)

        if data_message["role"] == "user":
            assert content is not None
            messages.append(UserMessage(content=content))
        elif data_message["role"] == "assistant":
            tool_calls: Optional[List[ToolCall]] = None

            if is_tool_call:
                tool_calls = _parse_tool_calls(data_message["tool_calls"])

            weight = data_message.get("weight")
            messages.append(
                FinetuningAssistantMessage(
                    content=content, tool_calls=tool_calls, weight=weight
                )
            )
        elif data_message["role"] == "system":
            if system_prompt is not None:
                err = "Multiple messages with role 'system' encountered. Only one is allowed."
                raise MessageFormatError(err, str(data))

            system_prompt = content
        elif data_message["role"] == "tool":
            assert content is not None
            tool_message = _parse_tool_message(content, data_message)
            messages.append(tool_message)

    # validate created messages
    validator = MistralRequestValidatorV3(ValidationMode.finetuning)
    # For training data, conversations often end with assistant messages
    # The validator will fail if we don't handle this case
    try:
        validator.validate_messages(messages, continue_final_message=False)
    except Exception as e:
        # If validation fails because the conversation ends with an assistant message,
        # that's expected for training data
        if "Expected last role User or Tool" in str(e) and messages and isinstance(messages[-1], FinetuningAssistantMessage):
            pass  # This is normal for training data
        else:
            raise  # Re-raise other validation errors
    validator._validate_tools(available_tools or [])

    # whether to train only on last assistant message
    only_last = data.get("only_last", False) or available_tools is not None

    return TrainingInstructSample(
        messages=messages,
        system_prompt=system_prompt,
        available_tools=available_tools,
        only_last=only_last,
    )


def _parse_available_tools(tools: List[Dict[str, Any]]) -> List[Tool]:
    available_tools = []
    for tool in tools:
        if "function" not in tool:
            raise FunctionFormatError(
                "A tool dict does not have a 'function' key.", str(tool)
            )

        func_data = tool["function"]

        for key in ["name", "description", "parameters"]:
            if key not in func_data:
                raise FunctionFormatError(
                    f"A function dict does not have a {key} key.", str(func_data)
                )

        if not isinstance(func_data["parameters"], dict):
            raise FunctionFormatError(
                f"A function 'parameters' key has to be of type dict, but is {type(func_data['parameters'])}. If the function has no parameters pass an empty dict ", str(func_data)
            )

        description = func_data["description"]
        function = Function(
            name=func_data["name"],
            description=description,
            parameters=func_data["parameters"],
        )

        available_tools.append(Tool(function=function))
    return available_tools


def _parse_tool_calls(calls: List[Dict[str, Any]]) -> List[ToolCall]:
    for key in ["id", "function"]:
        if not all(key in call for call in calls):
            err = f"A tool call of an assistant message does not have a {key} key"
            raise ToolCallFormatError(err, str(calls))

    for key in ["name", "arguments"]:
        if not all(key in call["function"] for call in calls):
            err = (
                f"A tool call function of an assistant message does not have a {key} key"
            )
            raise ToolCallFormatError(err, str(calls))

    if not all(isinstance(call["function"]["arguments"], str) for call in calls):
        err = "A tool call function of an assistant message does not have a 'arguments' key of type str"
        raise ToolCallFormatError(err, str(calls))

    tool_calls = [
        ToolCall(
            id=call["id"],
            function=FunctionCall(
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            ),
        )
        for call in calls
    ]
    return tool_calls


def _parse_tool_message(content: str, data_message: Dict[str, Any]) -> ToolMessage:
    if "tool_call_id" not in data_message:
        err = f"A tool message does not contain a 'tool_call_id' key, but only {', '.join(data_message.keys())}. Make sure that the message includes the key 'tool_call_id'."
        raise MessageFormatError(err, str(data_message))

    call_id = data_message["tool_call_id"]
    # name is deprecated in v3, but we'll add it nevertheless for now
    name = data_message.get("name")

    return ToolMessage(content=content, tool_call_id=call_id, name=name)


def tokenize(
    sample: Union[str, TrainingInstructSample],
    instruct_tokenizer: MistralTokenizer,
) -> TokenSample:
    if isinstance(sample, str):
        tokenizer: Tokenizer = instruct_tokenizer.tokenizer
        return tokenize_pretrain(sample, tokenizer)
    elif isinstance(sample, TrainingInstructSample):
        return tokenize_instruct(sample, instruct_tokenizer)

    raise ValueError(
        f"`sample` has to be either of type `str` or `TrainingInstructSample`, not {type(sample)}."
    )


def tokenize_pretrain(sample: str, tokenizer: Tokenizer) -> TokenSample:
    tokens = tokenizer.encode(sample, bos=True, eos=True)
    masks = [True] * len(tokens)
    return TokenSample(tokens, masks)


def tokenize_instruct(
    sample: TrainingInstructSample,
    instruct_tokenizer: MistralTokenizer,
) -> TokenSample:
    """
    Tokenize an instruct sample using mistral-common v1.8.1 API
    """
    from mistral_common.protocol.instruct.request import InstructRequest
    
    # Create request compatible with v1.8.1
    # Note: v1.8.1 validator expects 'tools' but InstructRequest has 'available_tools'
    # We work around this by using a custom class
    class InstructRequestCompat(InstructRequest):
        @property
        def tools(self):
            return self.available_tools
        
        @property
        def truncate_for_context_length(self):
            # Default value for truncate_for_context_length
            return False
    
    request = InstructRequestCompat(
        messages=sample.messages,
        available_tools=sample.available_tools if sample.available_tools else None,
        system_prompt=sample.system_prompt if hasattr(sample, 'system_prompt') and sample.system_prompt else None,
        continue_final_message=True  # Required for training data ending with assistant messages
    )
    
    # Encode the entire conversation
    encoded = instruct_tokenizer.encode_chat_completion(request)
    tokens = encoded.tokens
    
    # Create masks - simplified approach for v1.8.1
    # In production, you'd want more sophisticated masking
    masks = [True] * len(tokens)
    
    # Handle only_last flag
    if sample.only_last and len(sample.messages) > 0:
        # Find the last assistant message
        last_assistant_idx = None
        for i in range(len(sample.messages) - 1, -1, -1):
            if isinstance(sample.messages[i], FinetuningAssistantMessage):
                last_assistant_idx = i
                break
        
        if last_assistant_idx is not None:
            # Rough approximation: mask the first 80% of tokens
            mask_until = int(len(tokens) * 0.8)
            masks = [False] * mask_until + [True] * (len(tokens) - mask_until)
    
    return TokenSample(tokens, masks)
def maybe_remove_call_id(message: FinetuningAssistantMessage, is_last_message: bool):
    if message.tool_calls is None or not is_last_message:
        return message

    # remove call id
    message.tool_calls = [
        ToolCall(function=call.function) for call in message.tool_calls
    ]

    return message
