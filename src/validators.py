import re
from src.schemas import FunctionDefinition


# ──────────────────────── Regex keyword shortcut ────────────────────────

REGEX_KEYWORD_MAP = {
    "numbers": r"\d+", "digits": r"\d+",
    "vowels": "[aeiouAEIOU]",
    "consonants": r"[^aeiouAEIOU\s]",
    "letters": "[a-zA-Z]",
    "uppercase": "[A-Z]",
    "lowercase": "[a-z]",
    "whitespace": r"\s+",
    "spaces": r"\s+",
    "punctuation": r"[.,!?;:]",
}


def get_regex_keyword_value(prompt_text: str) -> str | None:
    lowered = prompt_text.lower()
    for kw, pattern in sorted(REGEX_KEYWORD_MAP.items(),
                              key=lambda x: -len(x[0])):
        if kw in lowered:
            return pattern
    return None


# ──────────────────────── Token validators ────────────────────────

def get_valid_bool_tokens(
        normalized_vocab: dict[str, int],
        token_categories: dict[str, set[int]],
        partial_value: str) -> set[int]:
    """
    Return tokens that can continue or finish a JSON boolean.
    """
    valid: set[int] = set()
    lowered_value = partial_value.lower()

    for target in ("true", "false"):
        if not target.startswith(lowered_value):
            continue

        remaining = target[len(lowered_value):]

        if remaining == "":
            valid |= token_categories[","]
            valid |= token_categories["}"]
            continue

        for normalized, token_id in normalized_vocab.items():
            if normalized and remaining.startswith(normalized.lower()):
                valid.add(token_id)

    return valid


def get_valid_number_tokens(
        normalized_vocab: dict[str, int],
        token_categories: dict[str, set[int]],
        partial_value: str) -> set[int]:
    """
    Returns the set of token IDs that can legally continue
    a JSON number value at the current generation step.
    """
    valid: set[int] = set()

    pattern = r"-?(0|[1-9]\d*)?(\.\d*)?([eE][+-]?\d*)?"
    complete_pattern = r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?"

    for normalized, token_id in normalized_vocab.items():
        if normalized and re.fullmatch(pattern, partial_value + normalized):
            valid.add(token_id)

    if partial_value and re.fullmatch(complete_pattern, partial_value):
        valid |= token_categories[',']
        valid |= token_categories['}']
    return valid


def get_valid_integer_tokens(
        normalized_vocab: dict[str, int],
        token_categories: dict[str, set[int]],
        partial_value: str) -> set[int]:
    """
    Return tokens that can continue a valid JSON integer.
    """
    valid: set[int] = set()

    partial_pattern = r"-?\d*"
    complete_pattern = r"-?\d+"

    for normalized, token_id in normalized_vocab.items():
        candidate = partial_value + normalized

        if normalized and re.fullmatch(partial_pattern, candidate):
            valid.add(token_id)

    if partial_value and re.fullmatch(complete_pattern, partial_value):
        valid |= token_categories[","]
        valid |= token_categories["}"]

    return valid


def get_valid_string_tokens(
        normalized_vocab: dict[str, int],
        token_categories: dict[str, set[int]],
        in_string: bool,
        partial_value: str,
        prompt_text: str) -> set[int]:
    """
    Returns tokens that keep partial_value an exact substring of
    prompt_text (anchored extraction), plus the closing quote once
    a non-empty value has been extracted.
    """
    if not in_string:
        return token_categories['"']

    valid: set[int] = set()
    valid |= token_categories['"']

    for normalized, token_id in normalized_vocab.items():
        if '"' in normalized or not normalized:
            continue
        candidate = partial_value + normalized
        if candidate in prompt_text:
            valid.add(token_id)
    return valid


# ──────────────────── Name and key validators ────────────────────

def get_valid_name_value_tokens(
        normalized_vocab: dict[str, int],
        token_categories: dict[str, set[int]],
        fn_defs: list[FunctionDefinition],
        written_so_far: str) -> set[int]:
    """
    Returns the set of token IDs that can legally continue
    the function name value at the current generation step.
    """
    valid: set[int] = set()
    remaining: str = ""

    for fn in fn_defs:
        if written_so_far == fn.name:
            return token_categories['"']

        if fn.name.startswith(written_so_far.lower()):
            remaining = fn.name[len(written_so_far):]
            for normalized, token_id in normalized_vocab.items():
                if remaining.startswith(normalized) and normalized:
                    valid.add(token_id)
    return valid


def get_valid_param_key_tokens(
        normalized_vocab: dict[str, int],
        token_categories: dict[str, set[int]],
        fn_def: FunctionDefinition,
        written_params: list[str],
        written_so_far: str) -> set[int]:
    """
    Returns the set of token IDs that can legally continue
    a parameter key name at the current generation step.
    """
    valid: set[int] = set()
    remaining: str = ""

    for fn_keys in fn_def.parameters.keys():
        if fn_keys in written_params:
            continue
        if written_so_far == fn_keys:
            return token_categories['"']
        if fn_keys.startswith(written_so_far):
            remaining = fn_keys[len(written_so_far):]
            for normalized, token_id in normalized_vocab.items():
                if remaining.startswith(normalized) and normalized:
                    valid.add(token_id)
    return valid
