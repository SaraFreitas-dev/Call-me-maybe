import json
import sys
import os
from pydantic import ValidationError
from typing import Any
from src.schemas import (FunctionDefinition, PromptEntry, FunctionCall)


def _load_json_file(path: str) -> list[dict[str, Any]]:
    """
    Opens a JSON file, loads its contents using json.load(),
    and handles I/O and JSON parsing errors.
    """
    try:
        with open(path, 'r') as file:
            data: list[dict[str, Any]] = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in file: {path}")
        sys.exit(1)


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """
    Calls _load_json_file() and validates each entry as a FunctionDefinition.
    """
    data = _load_json_file(path)
    fn: list[FunctionDefinition] = []

    try:
        for dict_data in data:
            fn_data = FunctionDefinition(**dict_data)
            fn.append(fn_data)
        return fn
    except ValidationError as e:
        print(f"Error loading FunctionDefinition JSON: {e}")
        sys.exit(1)


def load_test_prompts(path: str) -> list[PromptEntry]:
    """
    Calls _load_json_file() and validates each entry as a PromptEntry.
    """
    data = _load_json_file(path)
    prompts: list[PromptEntry] = []

    try:
        for dict_data in data:
            prompts_data = PromptEntry(**dict_data)
            prompts.append(prompts_data)
        return prompts
    except ValidationError as e:
        print(f"Error loading PromptEntry JSON: {e}")
        sys.exit(1)


def save_function_calls(path: str,
                        results: list[FunctionCall]) -> None:
    """
    Serializes a list of FunctionCall instances to a JSON file
    at the given path, creating parent directories if needed.
    """
    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data: list[dict[str, Any]] = []

    for r in results:
        data.append(r.model_dump())

    try:
        with open(path, "w") as file:
            json.dump(data, file, indent=2)
    except OSError as e:
        print(f"Error writing the output file: {path} ({e})")
        sys.exit(1)
