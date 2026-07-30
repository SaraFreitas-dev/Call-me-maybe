from src.vocab_loader import (build_id_to_str,
                              build_token_categories)
from src.schemas import (FunctionDefinition, FunctionCall)
from src.llm_engine import build_prompt_request
from src.vocab_loader import replace_space_markers
from typing import Any
import re


class ConstrainedDecoder:
    """
    Core component of the project.
    Perform token-by-token constrained decoding.
    Restrict valid next-token choices.
    Ensure only valid JSON structures can be generated.
    Ensure only existing functions and parameters are used.
    Prevent invalid outputs from the language model.
    Generate structured function calls safely.
    """
    def __init__(self,
                 model: Any,
                 fn_defs: list[FunctionDefinition],
                 vocab: dict[str, int]
                 ) -> None:
        self.model = model
        self.fn_defs = fn_defs
        self.vocab = vocab  # token_string → token_id  (real vocab format)
        self.id_to_str: dict[int, str] = build_id_to_str(vocab)
        self.token_categories: dict[str, set[int]] = build_token_categories(
            vocab)

    # ──────────────────────── Token validators ────────────────────────
    def _get_valid_bool_tokens(
            self,
            partial_value: str) -> set[int]:
        """
        Returns the set of token IDs that can legally continue
        a JSON boolean value at the current generation step.

        JSON booleans are exactly 'true' or 'false' (lowercase).
        At each step, only tokens that continue one of these two
        targets from the current partial value are allowed.

        e.g.:
            partial_value = ""   → tokens starting "true" or "false"
            partial_value = "t"  → tokens continuing "rue"
            partial_value = "tr" → tokens continuing "ue"FunctionCall
        """
        valid: set[int] = set()
        remaining: str = ""
        normalized: str = ""

        for target in ["true", "false"]:
            if target.startswith(partial_value.lower()):
                remaining = target[len(partial_value):]
            for token_str, token_id in self.vocab.items():
                normalized = replace_space_markers(token_str)
                if remaining.startswith(normalized) and normalized:
                    valid.add(token_id)
        return valid

    def _get_valid_number_tokens(
            self,
            partial_value: str) -> set[int]:
        """
        Returns the set of token IDs that can legally continue
        a JSON number value at the current generation step.

        Valid JSON number characters are: digits (0-9), decimal
        point (.), minus (-), plus (+), and scientific notation (e, E).
        Only tokens whose normalized string, when appended to the
        partial value, form a valid number prefix are allowed.

        e.g.:
            partial_value = ""    → tokens starting any valid number
            partial_value = "2"   → continuing a number: ".", "e", digits
            partial_value = "2."  → continuing a decimal: digits only
        """
        valid: set[int] = set()
        normalized: str = ""

        pattern = r'^-?(\d+(\.\d*)?([eE][+-]?\d*)?)?$'

        for token_str, token_id in self.vocab.items():
            normalized = replace_space_markers(token_str)
            if normalized and re.match(pattern, partial_value + normalized):
                valid.add(token_id)
        return valid

    def _get_valid_string_tokens(
            self,
            in_string: bool,
            partial_value: str) -> set[int]:
        """
        Returns the set of token IDs that can legally appear
        at the current position inside a JSON string value.

        If in_string is False, only the opening quote '"' is valid.
        If in_string is True, any printable token is valid as string
        content, plus the closing quote '"' to end the string.
        Tokens containing unescaped quotes (that would prematurely
        close the string) are excluded from the valid set.

        e.g.:
            in_string = False → only {id of '"'}
            in_string = True  → all printable tokens + closing '"'
        """
        valid: set[int] = set()
        normalized: str = ""

        if not in_string:
            return self.token_categories['"']
        else:
            for token_str, token_id in self.vocab.items():
                normalized = replace_space_markers(token_str)
                if normalized == '"':
                    valid.add(token_id)
                elif '"' not in normalized:
                    valid.add(token_id)
                else:
                    continue
        return valid

    # ──────────────────── Name and key validators ────────────────────
    def _get_valid_name_value_tokens(
            self,
            written_so_far: str) -> set[int]:
        """
        Returns the set of token IDs that can legally continue
        the function name value at the current generation step.

        Only tokens that continue at least one valid function name
        from fn_defs, given what has already been written, are allowed.
        Once a function name is fully written, only the closing
        quote '"' is valid.

        e.g.:
            fn_defs has ["fn_greet", "fn_add_numbers"]
            written_so_far = ""    → tokens starting
                                    "fn_greet" or "fn_add_numbers"
            written_so_far = "fn_" → tokens continuing
                                    "greet" or "add_numbers"
            written_so_far = "fn_greet" → only {id of '"'}
        """
        valid: set[int] = set()
        remaining: str = ""
        normalized: str = ""

        for fn in self.fn_defs:
            if written_so_far == fn.name:
                # The name is complete - close the bracket
                return self.token_categories['"']

            if fn.name.startswith(written_so_far.lower()):
                remaining = fn.name[len(written_so_far):]
                for token_str, token_id in self.vocab.items():
                    normalized = replace_space_markers(token_str)
                    if remaining.startswith(normalized) and normalized:
                        valid.add(token_id)
        return valid

    def _get_valid_param_key_tokens(
            self,
            fn_def: FunctionDefinition,
            written_params: list[str],
            written_so_far: str) -> set[int]:
        """
        Returns the set of token IDs that can legally continue
        a parameter key name at the current generation step.

        Only tokens that continue a parameter name from fn_def
        that has not yet been written are allowed. Already written
        parameters are excluded from the valid options.

        e.g.:
            fn_def has parameters ["a", "b"]
            written_params = []    → tokens for "a" or "b"
            written_params = ["a"] → tokens for "b" only
            written_so_far = "b"   → only {id of '"'} to close the key
        """
        valid: set[int] = set()
        remaining: str = ""
        normalized: str = ""

        for fn_keys in fn_def.parameters.keys():
            if fn_keys in written_params:
                continue
            if written_so_far == fn_keys:
                return self.token_categories['"']
            if fn_keys.startswith(written_so_far):
                remaining = fn_keys[len(written_so_far):]
                for token_str, token_id in self.vocab.items():
                    normalized = replace_space_markers(token_str)
                    if remaining.startswith(normalized) and normalized:
                        valid.add(token_id)
        return valid

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
        elif 'parameters' not in partial_json:
            if partial_json.count('"') < 4:
                return "name_value"
            else:
                return "structural"
        else:
            # Already contains "parameters"
            params_content = partial_json.split('"parameters"')[-1]
            if ':' not in params_content:
                return "arg_key"
            elif partial_json[-1] not in [',', '}', ' ']:
                return "arg_value"
            else:
                return "structural"

    def _get_tokens_for_state(self,
                              partial_json: str,
                              fn_def: FunctionDefinition | None,
                              written_params: list[str],
                              current_param: str,
                              written_so_far: str,
                              in_string: bool) -> set[int]:
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
            valid_tokens = self._get_valid_name_value_tokens(written_so_far)
        elif state == "arg_key":
            valid_tokens = self._get_valid_param_key_tokens(fn_def,
                                                            written_params,
                                                            written_so_far)

        elif state == "arg_value":
            param_type = fn_def.parameters[current_param].type
            if param_type == "number":
                valid_tokens = self._get_valid_number_tokens(written_so_far)
            elif param_type == "string":
                valid_tokens = self._get_valid_string_tokens(in_string,
                                                             written_so_far)
            elif param_type == "boolean":
                valid_tokens = self._get_valid_bool_tokens(written_so_far)

        elif state == "structural":
            last = partial_json.rstrip()[-1]

            if last == "{":
                valid_tokens = self.token_categories['"']
            elif last == ":":
                valid_tokens = self.token_categories['"']
            elif last == ",":
                valid_tokens = self.token_categories['"']
            elif last == '"':
                # depois de fechar uma aspa → dois pontos ou vírgula
                if '"parameters"' not in partial_json:
                    valid_tokens = self.token_categories[',']
                else:
                    valid_tokens = self.token_categories[':']

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
            ...
        elif state == "arg_key":
            ...
        elif state == "arg_value":
            ...
        return fn_def, written_params, current_param, written_so_far, in_string

    # ──────────────────── Main generation loop ────────────────────

    def generate_function_call(
            self,
            prompt: str,
            max_tokens: int = 200) -> FunctionCall:
        """
        Generates a schema-valid JSON function call for the given prompt
        using token-by-token constrained decoding.

        Builds a full prompt from the user request and available functions,
        encodes it, then iteratively selects the next valid token until
        the JSON output is complete. At each step, invalid tokens are
        masked to -inf to guarantee structural and schema compliance.

        Args:
            prompt:     The original natural language user request.
            max_tokens: Maximum number of tokens to generate before stopping.

        Returns:
            A FunctionCall instance containing the original prompt, the
            selected function name, and the extracted parameters.
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
                in_string
            )

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

            (fn_def,
             written_params,
             current_param,
             written_so_far,
             in_string) = self._apply_token(
                 state, token_text, fn_def, written_params,
                 current_param, written_so_far, in_string)

            if self._get_current_state(partial_json) == "complete":
                break
