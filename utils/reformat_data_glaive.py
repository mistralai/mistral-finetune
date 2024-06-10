#!/usr/bin/env python3
import argparse
import json
import os
import random
import string


def reformat_jsonl(input_file):  # noqa: C901
    output_file = os.path.splitext(input_file)[0] + "_reformatted.jsonl"
    skipped_samples = []

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for i, line in enumerate(infile):
            reformat_data = True
            data = json.loads(line)

            # Extract function description
            try:
                function_desc = json.loads(data["function_description"])
            except json.decoder.JSONDecodeError:
                function_desc = (
                    data["function_description"].replace("\n", "").replace("}{", "},{").replace("\\t", "")
                )
                function_desc = "[{" + function_desc[1:-1] + "}]"
                function_desc = json.loads(function_desc)

            function_desc = function_desc if isinstance(function_desc, list) else [function_desc]

            # Reformat tools section
            if len(function_desc) == 1 and function_desc[0] == {}:
                tools = None
            else:
                tools = []
                for f in function_desc:
                    if f["parameters"] is None:
                        f["parameters"] = {}
                    tools.append({"type": "function", "function": f})

            messages = []

            # Process conversations
            for idx, msg in enumerate(data["conversations"]):
                role = msg["from"]
                content = msg["value"]

                if role == "system":
                    messages.append(
                        {"role": "system", "content": content.split(" -")[0]}
                    )
                elif role == "human":
                    messages.append({"role": "user", "content": content})
                elif role == "function-call":
                    try:
                        function_call = json.loads(content)
                    except json.decoder.JSONDecodeError:
                        content = content.replace("'", "").replace("\\", "'")
                        try:
                            function_call = json.loads(content)
                        except:  # noqa: E722
                            skipped_samples.append(str(i))
                            reformat_data = False
                            break

                    if not isinstance(function_call, list):
                        function_calls = [function_call]
                    else:
                        function_calls = function_call

                    tool_calls = []
                    for function_call in function_calls:
                        assert not isinstance(function_call, list)
                        tool_call_id = "".join(
                            random.choices(string.ascii_letters + string.digits, k=9)
                        )

                        if "arguments" in function_call and not isinstance(function_call["arguments"], str):
                            function_call["arguments"] = str(function_call["arguments"])
                        elif "arguments" not in function_call:
                            function_call["arguments"] = ""

                        tool_calls.append({"id": tool_call_id, "type": "function", "function": function_call})

                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": tool_calls
                        }
                    )
                elif role == "function-response":
                    if "tool_calls" not in messages[-1]:
                        skipped_samples.append(str(i))
                        reformat_data = False
                        break

                    assert len(messages[-1]["tool_calls"]) == 1
                    tool_call_id = messages[-1]["tool_calls"][0]["id"]
                    messages.append(
                        {
                            "role": "tool",
                            "content": content,
                            "tool_call_id": tool_call_id,
                        }
                    )
                elif role == "gpt":
                    messages.append({"role": "assistant", "content": content})

            output_data = {"messages": messages}

            if tools is not None:
                output_data["tools"] = tools

            if reformat_data:
                outfile.write(json.dumps(output_data) + "\n")

    os.rename(output_file, input_file)
    print(
        f"Skipped {len(skipped_samples)} samples ({len(skipped_samples) / i:.2%}). The following samples are incorrectly formatted: \n\n {', '.join(skipped_samples)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat a JSONL file.")
    parser.add_argument("file", type=str, help="The input JSONL file")

    args = parser.parse_args()
    reformat_jsonl(args.file)
