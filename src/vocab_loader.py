"""
Defines the project's data structures and validation models
For each JSON file:
input (definitions & tests) and output(results)
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
    Converts each 'str: id' into 'id: str' into the new dict
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


def build_token_categories(
        str_to_id: dict[str, int]) -> dict[str, set[int]]:
    """
    Pre-computes sets of token IDs grouped by their structural role
    in JSON, so the constrained decoder can look them up instantly
    instead of scanning the full vocabulary at every generation step.

    Returns a dictionary mapping each structural category to the set
    of token IDs that belong to it:
        {
            "{":        token IDs representing an open brace
            "}":        token IDs representing a close brace
            '"':        token IDs representing a double quote
            ":":        token IDs representing a colon
                        (ignoring surrounding spaces)
            ",":        token IDs representing a comma
                        (ignoring surrounding spaces)
            "numeric":  token IDs made up only of digits,
                        '.', '-', '+', 'e', 'E'
            "special":  token IDs for special tokens like <|im_end|>
        }
    """
    sets: dict[str, set[int]] = {
        "{": set(),
        "}": set(),
        '"': set(),
        ":": set(),
        ",": set(),
        "numeric": set(),
        "special": set(),
    }

    for (token_str, id) in str_to_id.items():
        normalized_token = replace_space_markers(token_str)
        # Markers
        if normalized_token == '{':
            sets['{'].add(id)
        elif normalized_token == '}':
            sets['}'].add(id)
        elif normalized_token == '"':
            sets['"'].add(id)
        elif normalized_token.strip() == ':':
            sets[':'].add(id)
        elif normalized_token.strip() == ',':
            sets[','].add(id)
        # Numeric and special values
        elif (all(c in "0123456789.-+eE" for c in normalized_token)
              and normalized_token != ''):
            sets['numeric'].add(id)
        elif token_str.startswith('<|') and token_str.endswith('|>'):
            sets['special'].add(id)
    return sets
