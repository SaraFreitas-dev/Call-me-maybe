"""
Acts as the interface between the application and the language model.
Talks to the model and builds the prompts
"""
from typing import Any
from llm_sdk import Small_LLM_Model
from src.schemas import FunctionDefinition

def create_model() -> Any:
    """
    Creates the LLM model
    Returns a new instance ready to use
    This function is used only to simplify any future changes
    """
    return Small_LLM_Model()


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




"""
You are a function calling assistant.

Available functions:
- fn_greet(name: string): Generate a greeting...

User request: "Greet john"

Output the correct function call as JSON:
{"name": "
"""
