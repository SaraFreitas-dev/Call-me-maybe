"""
get_valid_string_tokens(...)   → set[int]
get_valid_number_tokens(...)   → set[int]
get_valid_bool_tokens(...)     → set[int]

get_valid_tokens_for_name(...)       → set[int]   ← dentro do valor "name"
get_valid_tokens_for_param_key(...)  → set[int]   ← dentro de uma chave de parâmetro
get_valid_tokens_for_param_value(...) → set[int]  ← dentro de um valor de parâmetro

generate_function_call(model, prompt, fn_def, str_to_id, id_to_str, categories) → dict
"""
from llm_sdk import Small_LLM_Model
from src.data_loader import load_function_definitions, load_test_prompts
from src.vocab_loader import load_vocab
from typing import Any


class ConstrainedDecoder:
    """
    Core component of the project.
    Perform token-by-token constrained decoding.
    Restrict valid next-token choices.
    Ensure only valid JSON structures can be generated.
    Ensure only existing functions and parameters are used.
    Prevent invalid outputs from the language model.
    Generate structured function calls safely.
    """
    def __init__(model: Any) -> None: