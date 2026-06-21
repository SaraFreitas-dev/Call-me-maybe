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
from llm_sdk import Small_LLM_Model
from src.data_loader import load_function_definitions, load_test_prompts
from src.vocab_loader import load_vocab


def main() -> None:
    model = Small_LLM_Model()

    fn_defs = load_function_definitions("data/input/functions_definition.json")
    prompts = load_test_prompts("data/input/function_calling_tests.json")

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
    main()
