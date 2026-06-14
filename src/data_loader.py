"""
Handles all file input/output operations.

Responsibilities:

Load function definitions from JSON files.
Load user prompts and test cases.
Validate input data structure.
Save generated function calls to the output JSON file.
_load_json_file(path)
    → Opens a JSON file, loads its contents using json.load(), and handles I/O and JSON parsing errors.

load_function_definitions(path)
    → Calls _load_json_file() and validates each entry as a FunctionDefinition.

load_test_prompts(path)
    → Calls _load_json_file() and validates each entry as a PromptEntry.
"""
import json
import sys
from pydantic import validationError
