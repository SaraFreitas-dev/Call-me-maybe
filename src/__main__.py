import argparse
import sys
from typing import Any
import src.env_setup  # noqa: F401  (import sets HF_HOME as a side effect)
from src.data_loader import load_function_definitions, load_test_prompts
from src.vocab_loader import load_vocab
from src.data_loader import save_function_calls
from src.constrained_decoder import ConstrainedDecoder
from src.schemas import FunctionCall
from src.llm_engine import create_model
from src.recovery import (generate_with_recovery,
                          export_recovery_report)


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
                        default="data/input/functions_definition.json",
                        help="Path to the JSON file containing "
                        "function definitions")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="Path to the JSON file containing input prompts")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json",
                        help="Path where generated function calls "
                        "will be saved")
    parser.add_argument("--model",
                        default="Qwen/Qwen3-0.6B",
                        help="Name of the language model to use")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print detailed generation trace to stdout")
    args = parser.parse_args()
    return args


def run() -> None:
    """
    Main function, runs the program
    """
    args = parse_args()
    model = create_model(model_name=args.model)

    fn_defs = load_function_definitions(args.functions_definition)
    prompts = load_test_prompts(args.input)

    str_to_id = load_vocab(model)

    const_decoder = ConstrainedDecoder(model,
                                       fn_defs,
                                       str_to_id,
                                       verbose=args.verbose)

    results: list[FunctionCall] = []
    total: int = len(prompts)
    recovery_log: list[dict[str, Any]] = []

    if args.verbose:
        print(f"╭───Total functions: {total} " + "─" * 30)

    for index, prompt_entry in enumerate(prompts, start=1):
        try:
            function_call = generate_with_recovery(
                const_decoder,
                prompt_entry.prompt,
                recovery_log)

            results.append(function_call)
            if args.verbose:
                const_decoder._print_verbose_report(
                    prompt=prompt_entry.prompt,
                    index=index,
                    total=total,
                    function_call=function_call,
                    error=None,
                )
        except Exception as error:
            print(
                f"Warning: failed to process prompt "
                f"{prompt_entry.prompt!r} "
                f"({type(error).__name__}): {error}",
                file=sys.stderr,
            )
            if args.verbose:
                const_decoder._print_verbose_report(
                    prompt=prompt_entry.prompt,
                    index=index,
                    total=total,
                    function_call=None,
                    error=error,
                )

    save_function_calls(args.output, results)

    if args.verbose:
        export_recovery_report("data/output/recovery_report.json",
                               recovery_log)
        print(
            f"\nFinished: {len(results)}/{total} "
            "functions generated successfully"
        )


if __name__ == "__main__":
    run()
