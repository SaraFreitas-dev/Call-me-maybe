"""

Main entry point of the application.

Responsibilities:

Parse command-line arguments.
Load input files.
Initialize the LLM engine.
Run the constrained decoding process.
Save generated function calls to the output file.
Coordinate the overall execution flow.
"""
# No fim do ficheiro temporariamente
from src.schemas import FunctionDefinition

if __name__ == "__main__":
    data = {
        "name": "fn_add_numbers",
        "description": "Add two numbers.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "returns": {"type": "number"}
    }
    fn = FunctionDefinition(**data)
    print(fn.name)
    print(fn.parameters["a"].type)
