import json
import os
from typing import Any
from src.schemas import FunctionCall
from src.constrained_decoder import ConstrainedDecoder


def generate_with_recovery(
        decoder: ConstrainedDecoder,
        prompt: str,
        recovery_log: list[dict[str, Any]],
        max_attempts: int = 2) -> FunctionCall:
    """
    Attempts generation, retrying with a larger token budget on
    failure. Appends a record to recovery_log for each failed
    attempt, so callers can later export a full recovery report.
    Raises:
        ValueError: If all attempts fail, re-raises the last error.
    """
    last_error: ValueError | None = None
    max_tokens = 200

    for attempt in range(1, max_attempts + 1):
        try:
            return decoder.generate_function_call(
                prompt, max_tokens=max_tokens)
        except ValueError as error:
            last_error = error
            recovery_log.append({
                "prompt": prompt,
                "attempt": attempt,
                "error": str(error),
            })
            if decoder.verbose:
                print(f"│ recovery attempt {attempt}/{max_attempts} "
                      f"failed: {error}")
            max_tokens *= 2

    assert last_error is not None
    raise last_error


def export_recovery_report(path: str,
                           recovery_log: list[dict[str, Any]]) -> None:
    """
    Writes the full recovery log to a JSON file. If no recovery
    attempts were recorded, writes a confirmation that the run
    completed without incident instead.
    """
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data: Any = recovery_log if recovery_log else {
        "status": "No recovery attempts were necessary; "
                  "all prompts were processed successfully."
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    if recovery_log:
        attempts = len(recovery_log)
        print(f"⚠️ Recovery info: {attempts} recovery attempt(s) "
              f"logged. Details saved to {path}")
    else:
        print(f"✅ Recovery info: no recovery needed. "
              f"Confirmation saved to {path}")
