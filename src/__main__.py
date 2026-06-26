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
import argparse
from llm_sdk import Small_LLM_Model
from src.data_loader import load_function_definitions, load_test_prompts
from src.vocab_loader import load_vocab


def parse_args() -> argparse.Namespace:
    """
    Parse the uv run arguments:
    --functions_definition with
        default data/input/functions_definition.json
    --input with default data/input/function_calling_tests.json
    --output with default data/output/function_calling_results.json
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()
    return args


def run() -> None:
    """
    Main function, runs the program
    """
    args = parse_args()
    model = Small_LLM_Model()

    fn_defs = load_function_definitions(args.functions_definition)
    prompts = load_test_prompts(args.input)

    print(f"Loaded {len(fn_defs)} functions:")
    for fn in fn_defs:
        print(f"  - {fn.name}")

    print(f"\nLoaded {len(prompts)} prompts:")
    for p in prompts:
        print(f"  - {p.prompt}")

    str_to_id = load_vocab(model)

    for token_str, token_id in str_to_id.items():
        if token_str.startswith("<|") and token_str.endswith("|>"):
            print(token_str, token_id)


if __name__ == "__main__":
    run()
