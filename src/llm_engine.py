"""
Acts as the interface between the application and the language model.
Talks to the model and builds the prompts
"""
from typing import Any
from llm_sdk import Small_LLM_Model
from src.schemas import FunctionDefinition


def create_model(model_name: str = "Qwen/Qwen3-0.6B") -> Any:
    """
    Creates the LLM model.
    Accepts an optional model_name to support different models
    beyond the default Qwen3-0.6B.
    Returns a new instance ready to use.
    This function is used only to simplify any future changes.
    """
    return Small_LLM_Model(model_name=model_name)


def build__prompt_request(user_request: str,
                          fn_defs: list[FunctionDefinition]) -> str:
    """
    Rceives the user prompt and the lists of available functions
    And returns a str ready for the LLM Model to understand and
    e.g:
    FunctionDefinition(
    name="fn_add_numbers",
    description="Add two numbers together...",
    parameters={"a": ParameterSchema(type="number"),
                "b": ParameterSchema(type="number")})
    And converts it to:
    - Instructions to the LLM
    - Options to choose from: fn_add_numbers(a: number,
                              b: number): Add two numbers together...
    - Correct output expected (function call)
    """
    fn_defs_lines: list[str] = []
    fn_defs_block: list[str] = []
    params_str: str = ""

    for fn in fn_defs:
        # build a: number, b: number (e.g.)
        param_parts = []
        for param_name, param_schema in fn.parameters.items():
            param_parts.append(f"{param_name}: {param_schema.type}")
        params_str = ", ".join(param_parts)

        # Name, description, parameters and returns
        fn_defs_lines.append(f"- {fn.name}({params_str}): {fn.description}")
    fn_defs_block = "\n".join(fn_defs_lines)

    return (
        "You are a function calling assistant.\n\n"
        "Available functions:\n"
        f"{fn_defs_block}\n\n"
        f"User request: {user_request}\n\n"
        "Output the correct function call as JSON:\n"
        '{"name": "'
    )
