from src.vocab_loader import (build_id_to_str,
                              build_token_categories)
from src.schemas import (FunctionDefinition, FunctionCall)
from src.llm_engine import build_prompt_request
from src.vocab_loader import replace_space_markers
from src.validators import (get_regex_keyword_value,
                            get_valid_bool_tokens,
                            get_valid_number_tokens,
                            get_valid_string_tokens,
                            get_valid_name_value_tokens,
                            get_valid_param_key_tokens,
                            get_valid_integer_tokens)
from typing import Any
import json


class ConstrainedDecoder:
    """
    Core component of the project.
    Perform token-by-token constrained decoding.
    Generate structured function calls safely.
    """
    def __init__(self,
                 model: Any,
                 fn_defs: list[FunctionDefinition],
                 vocab: dict[str, int],
                 verbose: bool = False) -> None:

        self.model = model
        self.fn_defs = fn_defs
        self.vocab = vocab  # token_string → token_id  (real vocab format)
        self.id_to_str: dict[int, str] = build_id_to_str(vocab)
        self.token_categories: dict[str, set[int]] = build_token_categories(
            vocab)
        self.normalized_vocab: dict[str, int] = {
            replace_space_markers(token_str): token_id
            for token_str, token_id in vocab.items()
            }
        self.verbose = verbose  # Bonus flag

    # ─────────────────────── State machine ───────────────────────

    def _get_current_state(self, partial_json: str) -> str:
        """
        Infers the current JSON generation state from the partial
        output generated so far.

        Returns one of the following state strings:
            "start"           → nothing written yet, expect '{'
            "name_value" → _get_valid_name_value_tokens
            "arg_key"    → _get_valid_param_key_tokens
            "arg_value"  → _get_valid_X_tokens (number/string/boolean)
            "complete"   → generation is finished
            "structural" → token_categories ('{', ',', ':', etc.)
        """
        if not partial_json:
            return "start"
        elif partial_json.endswith("}}"):
            return "complete"
        elif '"parameters"' not in partial_json:
            if partial_json.count('"') < 4:
                return "name_value"
            else:
                return "structural"
        else:
            # Already contains "parameters"
            params_content = partial_json.split('"parameters"')[-1]
            if '{' not in params_content:
                return "structural"

            after_brace = params_content.split('{', 1)[-1]
            last_segment = after_brace.rsplit(',', 1)[-1]

            if ':' not in last_segment:
                return "arg_key"

            value_part = last_segment.split(':', 1)[-1]
            if value_part.count('"') % 2 == 1:
                # inside an open string
                return "arg_value"
            elif partial_json[-1] not in [',', '}', ' ', '"']:
                return "arg_value"
            else:
                return "structural"

    def _get_tokens_for_state(self,
                              partial_json: str,
                              fn_def: FunctionDefinition | None,
                              written_params: list[str],
                              current_param: str,
                              written_so_far: str,
                              in_string: bool,
                              prompt_text: str) -> set[int]:
        """
        Central dispatcher for constrained decoding.
        Returns the set of valid token IDs for the current
        JSON generation state and context.
        """
        state: str = self._get_current_state(partial_json)
        valid_tokens: set[int] = set()

        if state == "start":
            valid_tokens = self.token_categories["{"]

        elif state == "name_value":
            valid_tokens = get_valid_name_value_tokens(
                self.normalized_vocab, self.token_categories,
                self.fn_defs, written_so_far)

        elif state == "arg_key":
            if fn_def is not None:
                quote_count = partial_json.count('"')
                if quote_count % 2 == 0:
                    last = partial_json.rstrip()[-1]
                    if last == '"':
                        # Just closed a key, missing the ':'
                        valid_tokens = self.token_categories[':']
                    else:
                        # After the ':' + value + ',' → New Key
                        valid_tokens = self.token_categories['"']
                else:
                    valid_tokens = get_valid_param_key_tokens(
                                    self.normalized_vocab,
                                    self.token_categories,
                                    fn_def,
                                    written_params,
                                    written_so_far)
        elif state == "arg_value":
            if fn_def is not None:
                param_type = fn_def.parameters[current_param].type
                if param_type == "number":
                    valid_tokens = get_valid_number_tokens(
                        self.normalized_vocab,
                        self.token_categories,
                        written_so_far)
                elif param_type == "integer":
                    valid_tokens = get_valid_integer_tokens(
                        self.normalized_vocab,
                        self.token_categories,
                        written_so_far)
                elif param_type == "string":
                    valid_tokens = get_valid_string_tokens(
                        self.normalized_vocab,
                        self.token_categories,
                        in_string,
                        written_so_far,
                        prompt_text)
                elif param_type == "boolean":
                    valid_tokens = get_valid_bool_tokens(
                        self.normalized_vocab,
                        self.token_categories,
                        written_so_far)

        elif state == "structural":
            quote_count = partial_json.count('"')

            if quote_count % 2 == 1:
                # Until the quotes are closed → Writing "parameters"
                target = "parameters"
                remaining = target[len(written_so_far):]
                for normalized, token_id in self.normalized_vocab.items():
                    if remaining.startswith(normalized) and normalized:
                        valid_tokens.add(token_id)
                if written_so_far == target:
                    valid_tokens = self.token_categories['"']

            else:
                last = partial_json.rstrip()[-1]

                if last == "{":
                    valid_tokens = self.token_categories['"']
                elif last == ":":
                    params_content = partial_json.split('"parameters"')[-1]
                    if ('"parameters"' in partial_json
                            and '{' not in params_content):
                        valid_tokens = self.token_categories['{']
                    else:
                        valid_tokens = self.token_categories['"']
                elif last == ",":
                    valid_tokens = self.token_categories['"']
                elif last == '"':
                    if partial_json.rstrip().endswith('"parameters"'):
                        valid_tokens = self.token_categories[':']
                    elif (fn_def is not None and
                          len(written_params) >= len(fn_def.parameters)):
                        # All the parameters are written - Only needs to close
                        valid_tokens = self.token_categories['}']
                    else:
                        valid_tokens = self.token_categories[',']
                elif last == '}':
                    valid_tokens = self.token_categories['}']

        elif state == "complete":
            return set()

        return valid_tokens

    def _apply_token(self,
                     state: str,
                     token_text: str,
                     fn_def: FunctionDefinition | None,
                     written_params: list[str],
                     current_param: str,
                     written_so_far: str,
                     in_string: bool
                     ) -> (tuple[FunctionDefinition | None,
                                 list[str], str, str, bool]):
        """
        Given the state BEFORE this token and the chosen token's text,
        Updates the parsing state given the token just generated.
        """

        if state == "name_value":
            if token_text == '"':
                for fn in self.fn_defs:
                    if fn.name == written_so_far:
                        fn_def = fn
                        break
                written_so_far = ""
            else:
                written_so_far += token_text

        elif state == "arg_key":
            if token_text == '"':
                current_param = written_so_far
                written_so_far = ""
            elif token_text.strip() == ':':
                pass  # nothing to track, structural separator
            else:
                written_so_far += token_text

        elif state == "arg_value" and fn_def is not None:
            param_type = fn_def.parameters[current_param].type

            if param_type == "string":
                if token_text == '"':
                    if not in_string:
                        in_string = True
                    else:
                        in_string = False
                        written_params.append(current_param)
                        current_param = ""
                        written_so_far = ""
                else:
                    written_so_far += token_text

            else:
                # number / boolean: value already ended if this token
                # is the separator that comes right after it
                if token_text.strip() in (",", "}"):
                    written_params.append(current_param)
                    current_param = ""
                    written_so_far = ""
                else:
                    written_so_far += token_text

        elif state == "structural":
            if token_text == '"' and written_so_far == "parameters":
                written_so_far = ""
            elif token_text.strip() not in ('"', ",", ":", "{", "}"):
                written_so_far += token_text

        return fn_def, written_params, current_param, written_so_far, in_string

    # ──────────────────── Bonus --verbose flag ────────────────────
    def _print_verbose_report(self,
                              prompt: str,
                              index: int,
                              total: int,
                              function_call: FunctionCall | None,
                              error: Exception | None) -> None:
        """
        Print a verbose report for one processed prompt.
        Shows the generated function call or the error that occurred.
        """
        print(f"\n╭─── function {index}/{total} " + "─" * 30)
        print(f"│ prompt: {prompt!r}")

        if function_call is not None:
            print(f"│ name: {function_call.name}")
            print(f"│ parameters: {function_call.parameters}")
            print("╰─── ✅ success")
        else:
            print(f"│ error ({type(error).__name__}): {error}")
            print("╰─── ❌ failed")

    # ──────────────────── Main generation loop ────────────────────
    def generate_function_call(
            self,
            prompt: str,
            max_tokens: int = 200) -> FunctionCall:
        """
        Generate a schema-valid function call from a natural-language prompt.

        Uses token-by-token constrained decoding and returns the selected
        function name with its parameters.
        """

        # tokenize the prompt — encode() returns a 2D tensor
        full_prompt = build_prompt_request(prompt, self.fn_defs)
        input_ids = self.model.encode(full_prompt)[0].tolist()

        generated_ids: list[int] = []
        partial_json = '{"name": "'
        written_params: list[str] = []
        current_param: str = ""
        written_so_far: str = ""
        in_string: bool = False
        fn_def: FunctionDefinition | None = None
        logits: list[float] = []

        for _ in range(max_tokens):
            state = self._get_current_state(partial_json)

            # Regex keyword shortcut: inject the whole value at once
            if (state == "arg_value" and fn_def is not None
                    and "regex" in current_param.lower()
                    and not in_string and written_so_far == ""):
                keyword_value = get_regex_keyword_value(prompt)
                if keyword_value is not None:
                    escaped_pattern = keyword_value.replace('\\', '\\\\')
                    injected = f'"{escaped_pattern}"'
                    injected_ids = self.model.encode(injected)[0].tolist()
                    generated_ids.extend(injected_ids)
                    partial_json += injected
                    written_params.append(current_param)
                    current_param = ""
                    written_so_far = ""
                    in_string = False
                    if self._get_current_state(partial_json) == "complete":
                        break
                    continue

            # get logits for all vocabulary tokens
            all_ids = input_ids + generated_ids
            logits = list(self.model.get_logits_from_input_ids(all_ids))

            # determine valid tokens at this position
            state = self._get_current_state(partial_json)
            valid_ids = self._get_tokens_for_state(
                partial_json,
                fn_def,
                written_params,
                current_param,
                written_so_far,
                in_string,
                prompt
            )

            if not valid_ids:
                raise ValueError(
                    f"No valid tokens available for state {state!r}. "
                    f"Partial JSON: {partial_json!r}")

            # mask all invalid tokens
            masked_logits = [
                logit if i in valid_ids else float('-inf')
                for i, logit in enumerate(logits)
            ]
            token_id = masked_logits.index(max(masked_logits))
            generated_ids.append(token_id)

            token_str = self.id_to_str[token_id]
            token_text = replace_space_markers(token_str)
            partial_json += token_text

            if self.verbose:
                # Print the state with the bonus --verbose flag
                print(
                    f"│ state={state} "
                    f"token_id={token_id} "
                    f"token={token_text!r} "
                    f"valid={len(valid_ids)}")

            (fn_def,
             written_params,
             current_param,
             written_so_far,
             in_string) = self._apply_token(
                 state, token_text, fn_def, written_params,
                 current_param, written_so_far, in_string)

            if self._get_current_state(partial_json) == "complete":
                break

        if self._get_current_state(partial_json) != "complete":
            # Max tokens were reached
            raise ValueError(
                f"Generation did not complete within {max_tokens} tokens. "
                f"Partial JSON: {partial_json!r}")

        if fn_def is None:
            raise ValueError("Model failed to produce a valid function name")

        try:
            parsed = json.loads(partial_json)
        except json.JSONDecodeError as e:
            raise ValueError("Model produced invalid JSON: "
                             f"{partial_json!r}") from e

        return FunctionCall(
            prompt=prompt,
            name=parsed["name"],
            parameters=parsed["parameters"],
        )
