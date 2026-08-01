import argparse
import src.env_setup  # noqa: F401  (import sets HF_HOME as a side effect)
from llm_sdk import Small_LLM_Model
from src.data_loader import load_function_definitions, load_test_prompts
from src.vocab_loader import load_vocab
from src.data_loader import save_function_calls
from src.constrained_decoder import ConstrainedDecoder
from src.schemas import FunctionCall


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

    str_to_id = load_vocab(model)

    constrained_decoder = ConstrainedDecoder(model, fn_defs, str_to_id)

    results: list[FunctionCall] = []

    for p in prompts:
        function_call = constrained_decoder.generate_function_call(p.prompt)
        results.append(function_call)

    save_function_calls(args.output, results)


if __name__ == "__main__":
    run()
