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
from src.data_loader import load_function_definitions, load_test_prompts

def main() -> None:
    fn_defs = load_function_definitions("data/input/functions_definition.json")
    prompts = load_test_prompts("data/input/function_calling_tests.json")

    print(f"Loaded {len(fn_defs)} functions:")
    for fn in fn_defs:
        print(f"  - {fn.name}")

    print(f"\nLoaded {len(prompts)} prompts:")
    for p in prompts:
        print(f"  - {p.prompt}")

if __name__ == "__main__":
    main()