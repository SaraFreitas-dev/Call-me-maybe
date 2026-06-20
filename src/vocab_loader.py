"""
Handles tokenizer and vocabulary resources.

Responsibilities:

Load tokenizer vocabulary files.
Load merge/tokenizer configuration files.
Build token lookup structures.
Provide vocabulary utilities for constrained decoding.
Translate between tokens and token IDs when required.

Input Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Language Model
"""
import json
import sys
from typing import Any


def load_vocab(llm_model: Any) -> dict[str, int]:
    """
    Load the vocabulary file from the LLM model and parse it into a dictionary.

    Returns a mapping of token strings to their integer IDs,
    e.g. {"!": 0, '"': 1, ...}
    """
    hugging_face_path = llm_model.get_path_to_vocab_file()
    try:
        with open(hugging_face_path, 'r') as file:
             data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: file not found: {hugging_face_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in file: {hugging_face_path}")
        sys.exit(1)


def build_id_to_str(str_to_id: dict[str, int]) -> dict[int, str]:
    """
    After getting an id, this function will retrieve the
    str associated with it
    e.g:  {5476: "{",  ...}    ← Asks what str has the id 5476?
    Converts each str: id into id: str into the new dict
    """
    id_to_str: dict[int, str] = {value: key
                                 for key, value in str_to_id.items()}
    return id_to_str


def replace_space_markers(token_str: str) -> str:
    """
    Converts the special BPE markers into normal spacing
    e.g.: 
    "Ġthe"  →  " the"  (Ġ turns into a space)
    "{"     →  "{"     (has no marker, so it remains the same)
    "Ċ"     →  "\n"    (Ċ becames a newline)
    """
    return token_str.replace("Ġ", " ").replace("Ċ", "\n")


def build_structural_sets(str_to_id: dict[str, int]) -> set[int]:
    """
    Returns a set of ints relevant for the constrained decoder
    """
    