"""
Handles all file input/output operations.

Responsibilities:
Load function definitions from JSON files.
Load user prompts and test cases.
Validate input data structure.
Save generated function calls to the output JSON file.
"""
import json
import sys
from pydantic import ValidationError
from typing import Any
from src.schemas import (FunctionDefinition, PromptEntry)


def _load_json_file(path: str) -> list[dict[str, Any]]:
    """
    Opens a JSON file, loads its contents using json.load(),
    and handles I/O and JSON parsing errors.
    """
    try:
        with open(path, 'r') as file:
             data = json.load(file)
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
    pe: list[PromptEntry] = []

    try:
        for dict_data in data:
            pe_data = PromptEntry(**dict_data)
            pe.append(pe_data)
        return pe
    except ValidationError as e:
        print(f"Error loading PromptEntry JSON: {e}")
        sys.exit(1)
