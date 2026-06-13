# 🔒 Constrained Decoding — Guaranteeing Valid JSON Output

> **What this doc covers:** The token-by-token generation loop, how to track JSON state at each step, how to use the vocabulary file to identify valid tokens, and how to enforce schema types (number, string, boolean). This is the core algorithm of the project.

---

## Table of Contents

1. [Why Constrained Decoding Exists](#why-constrained-decoding-exists)
2. [The Generation Loop](#the-generation-loop)
3. [The Vocabulary File](#the-vocabulary-file)
4. [Tracking JSON State](#tracking-json-state)
5. [Valid Tokens at Each JSON Position](#valid-tokens-at-each-json-position)
6. [Enforcing Schema Types](#enforcing-schema-types)
7. [The -inf Trick](#the--inf-trick)
8. [Complete Walkthrough](#complete-walkthrough)
9. [Putting It All Together](#putting-it-all-together)
10. [Common Mistakes](#common-mistakes)

---

## Why Constrained Decoding Exists

At each generation step, the model outputs a probability score (logit) for every token in its vocabulary — roughly 150,000 tokens for Qwen3. Normally, you just pick the highest one.

The problem: **nothing stops the model from picking a token that breaks your JSON.**

```
Partial output so far:  '{"name": "'
Model's top candidates:
  "fn_add_numbers"   logit: 3.2   ✅ valid function name
  "fn_greet"         logit: 2.9   ✅ valid function name
  "The"              logit: 1.4   ❌ breaks JSON — not a function name
  "42"               logit: 0.8   ❌ breaks schema — name must be a string
  " "                logit: 0.3   ❌ breaks JSON — unexpected whitespace here
```

Without intervention, the model might pick `"The"` and produce broken output. With constrained decoding, you set all invalid token logits to `-inf` before selection, so the model can only ever pick a valid token.

```
After constraint:
  "fn_add_numbers"   logit: 3.2   ✅ stays
  "fn_greet"         logit: 2.9   ✅ stays
  "The"              logit: -inf  ❌ eliminated
  "42"               logit: -inf  ❌ eliminated
  " "                logit: -inf  ❌ eliminated

Selected token: "fn_add_numbers"  ← correct, guaranteed
```

**Result:** 100% valid, schema-compliant JSON every time — regardless of model size.

---

## The Generation Loop

This is the fundamental loop you implement. Everything else in the project plugs into it.

```python
import math
import json
from typing import List

def generate_constrained(
    model,
    prompt: str,
    fn_def: FunctionDefinition,
    all_fn_names: List[str],
    vocab: dict,          # token_id (int) → token_string (str)
    max_tokens: int = 200
) -> dict:
    """Generate a constrained function call JSON for a given prompt."""

    # Step 1: tokenise the prompt
    input_ids: List[int] = list(model.encode(prompt))

    generated_ids: List[int] = []
    partial_json: str = ""

    for _ in range(max_tokens):
        # Step 2: get logits for all vocabulary tokens
        all_ids = input_ids + generated_ids
        logits: List[float] = list(model.get_logits_from_input_ids(all_ids))

        # Step 3: determine valid tokens at this position
        valid_ids = get_valid_token_ids(partial_json, fn_def, all_fn_names, vocab)

        # Step 4: mask all invalid tokens
        for i in range(len(logits)):
            if i not in valid_ids:
                logits[i] = -math.inf

        # Step 5: pick the highest-scoring valid token
        next_id = int(max(range(len(logits)), key=lambda i: logits[i]))

        # Step 6: append and update state
        generated_ids.append(next_id)
        token_str = vocab[str(next_id)]      # or vocab[next_id] depending on format
        partial_json += token_str

        # Step 7: check if generation is complete
        if is_complete(partial_json, fn_def):
            break

    return json.loads(partial_json)
```

The key function is `get_valid_token_ids` — it answers the question: *given what we've generated so far, which tokens are legal next?*

---

## The Vocabulary File

`model.get_path_to_vocab_file()` returns the path to a JSON file that maps every token ID to its string representation.

```python
import json

vocab_path = model.get_path_to_vocab_file()
with open(vocab_path) as f:
    raw_vocab = json.load(f)

# raw_vocab might be structured as:
# {"0": "!", "1": '"', "2": "#", ..., "5476": "{", "9313": "}", ...}
# keys are string representations of integer IDs

# Build reverse lookup: token_string → token_id
str_to_id: dict[str, int] = {v: int(k) for k, v in raw_vocab.items()}
id_to_str: dict[int, str] = {int(k): v for k, v in raw_vocab.items()}
```

### What to look for in the vocabulary

You need to find the token IDs for specific characters and strings. Build these lookup sets once at startup, not inside the generation loop:

```python
def build_vocab_lookups(vocab: dict[int, str]) -> dict:
    """Pre-compute useful token sets from the vocabulary."""

    # Single structural characters
    open_brace   = {id for id, s in vocab.items() if s == "{"}
    close_brace  = {id for id, s in vocab.items() if s == "}"}
    open_quote   = {id for id, s in vocab.items() if s == '"'}
    colon        = {id for id, s in vocab.items() if s == ":"}
    comma        = {id for id, s in vocab.items() if s == ","}

    # Numeric tokens — digits, decimal point, minus, scientific notation
    numeric      = {id for id, s in vocab.items()
                    if all(c in "0123456789.-+eE" for c in s) and s}

    # Whitespace tokens (spaces between JSON elements)
    whitespace   = {id for id, s in vocab.items()
                    if s.strip() == "" and s != ""}

    return {
        "{": open_brace,
        "}": close_brace,
        '"': open_quote,
        ":": colon,
        ",": comma,
        "numeric": numeric,
        "whitespace": whitespace,
    }
```

### The leading-space problem

Tokenizers attach spaces to the **beginning** of the following token, not the end of the previous one. A space before `{` is part of the `{` token, not a separate space token.

```
Text:    '{"name": "fn_greet"}'
Tokens:  ['{', '"', 'name', '"', ':', ' "', 'fn', '_greet', '"', '}']
                                         ↑
                              space is part of this token: ' "'
```

This matters when you search the vocabulary for `:`. There may be tokens for `:` and ` :` (with leading space) and `: ` (with trailing space). You need to account for all variants:

```python
colon_variants = {id for id, s in vocab.items() if s.strip() == ":"}
```

---

## Tracking JSON State

To know which tokens are valid at each position, you need to know **where you are in the JSON structure**. The simplest approach is a state machine.

### The JSON structure you're generating

```
{
  "name": "<function_name>",
  "parameters": {
    "<param1>": <value1>,
    "<param2>": <value2>
  }
}
```

### States

```python
from enum import Enum, auto

class JSONState(Enum):
    START               = auto()   # haven't written anything yet → expect '{'
    AFTER_OPEN_BRACE    = auto()   # wrote '{' → expect '"name"'
    IN_NAME_KEY         = auto()   # writing the literal key "name"
    AFTER_NAME_KEY      = auto()   # wrote '"name"' → expect ':'
    IN_NAME_VALUE       = auto()   # writing the function name string
    AFTER_NAME_VALUE    = auto()   # wrote function name → expect ','
    IN_PARAMS_KEY       = auto()   # writing '"parameters"'
    AFTER_PARAMS_KEY    = auto()   # wrote '"parameters"' → expect ':'
    IN_PARAMS_OPEN      = auto()   # wrote ':' → expect '{'
    IN_ARG_KEY          = auto()   # writing an argument name key
    AFTER_ARG_KEY       = auto()   # wrote key → expect ':'
    IN_ARG_VALUE_NUM    = auto()   # writing a numeric argument value
    IN_ARG_VALUE_STR    = auto()   # writing a string argument value
    IN_ARG_VALUE_BOOL   = auto()   # writing a boolean argument value
    AFTER_ARG_VALUE     = auto()   # wrote value → expect ',' or '}'
    COMPLETE            = auto()   # wrote final '}' → done
```

### Parsing current state from partial output

Rather than tracking state with a variable, you can **infer it from what's been generated so far**. This is more robust and easier to debug:

```python
def get_current_state(partial: str, fn_def: FunctionDefinition) -> JSONState:
    """Determine current JSON generation state from partial output."""
    partial = partial.strip()

    if partial == "":
        return JSONState.START

    if partial == "{":
        return JSONState.AFTER_OPEN_BRACE

    if not partial.startswith('{"name"'):
        return JSONState.IN_NAME_KEY   # still writing "name" key

    if ',' not in partial:
        # Still in the name section
        if partial.count('"') < 4:
            return JSONState.IN_NAME_VALUE   # inside the function name string
        return JSONState.AFTER_NAME_VALUE

    if '"parameters"' not in partial:
        return JSONState.IN_PARAMS_KEY

    params_section = partial.split('"parameters"')[1].strip()

    if not params_section.startswith(':'):
        return JSONState.AFTER_PARAMS_KEY

    # We're inside the parameters object
    inner = params_section.lstrip(': {').rstrip('}')

    # Count how many param key-value pairs are complete
    # ... (count colons, quotes, commas to determine progress)

    return JSONState.IN_ARG_KEY   # simplified — expand for your implementation
```

> **Tip:** For reliability, track state as a variable that you update token by token, rather than re-parsing the full partial string each time. The walkthrough section below shows this approach.

---

## Valid Tokens at Each JSON Position

Here's what tokens are valid at each stage of generating the output JSON:

### Stage 1: Start → `{`

```python
# Only the open brace is valid
valid = vocab_lookups["{"]
```

### Stage 2: After `{` → the key `"name"`

The next tokens must spell out exactly `"name"`. You know what's coming — it's always the same literal:

```python
# Find tokens that continue the literal sequence '"name"'
# Given partial = "{", next must start '"'
# Given partial = '{"', next must be 'n' or 'na' or 'nam' etc.
already_written = len(partial) - 1  # chars after '{'
target = '"name"'
remaining = target[already_written:]
valid = {id for id, s in vocab.items() if remaining.startswith(s)}
```

### Stage 3: After `"name"` → `:`

```python
valid = vocab_lookups[":"]  # including space variants like ' :'
```

### Stage 4: Inside function name value → restricted to valid function names

This is where the model makes its function selection decision, guided by the prompt semantics:

```python
# partial ends with '{"name": "'
# next tokens must continue one of the valid function names

written_so_far = extract_name_value(partial)  # e.g. "" or "fn_" or "fn_greet"

valid = set()
for fn_name in all_function_names:          # ["fn_add_numbers", "fn_greet", ...]
    if fn_name.startswith(written_so_far):
        remaining = fn_name[len(written_so_far):]
        # find all tokens that are a prefix of 'remaining'
        for token_id, token_str in vocab.items():
            if remaining.startswith(token_str):
                valid.add(token_id)
```

### Stage 5: After function name → closing `"` then `,`

```python
# Close the name string, then comma to move to parameters
if not name_closed:
    valid = vocab_lookups['"']
else:
    valid = vocab_lookups[',']
```

### Stage 6: The literal `"parameters"` key

Same approach as `"name"` — it's a fixed literal, token by token.

### Stage 7: After `"parameters":` → `{`

```python
valid = vocab_lookups["{"]
```

### Stage 8: Argument key → restricted to valid parameter names

```python
# Only the argument names from fn_def.parameters are valid here
param_names = list(fn_def.parameters.keys())   # e.g. ["a", "b"] or ["name"]
written = extract_current_key(partial)

valid = set()
for param_name in param_names:
    if param_name not in already_written_params:
        full_key = f'"{param_name}"'
        remaining = full_key[len(written):]
        for token_id, token_str in vocab.items():
            if remaining.startswith(token_str):
                valid.add(token_id)
```

### Stage 9: Argument value → depends on schema type

This is where the type system kicks in. See the next section.

### Stage 10: After each argument → `,` or `}`

```python
remaining_params = [p for p in fn_def.parameters if p not in written_params]

if remaining_params:
    valid = vocab_lookups[","]   # more params to write
else:
    valid = vocab_lookups["}"]   # close params object

# Then after the outer params object closes:
valid = vocab_lookups["}"]       # close the root object
```

---

## Enforcing Schema Types

This is where you use `fn_def.parameters[param_name].type` to decide which tokens are valid inside the argument value.

### `type: "number"`

A JSON number can contain: digits `0-9`, decimal point `.`, minus sign `-`, plus `+`, and scientific notation `eE`. Nothing else.

```python
def get_valid_number_tokens(
    partial_value: str,
    vocab: dict[int, str]
) -> set[int]:
    """
    Returns token IDs that can legally continue a JSON number value.
    partial_value is what's been written for this number so far (e.g. "26" or "3.")
    """
    valid = set()

    for token_id, token_str in vocab.items():
        candidate = partial_value + token_str

        # Check if candidate is a valid prefix of a JSON number
        # Valid prefixes: "-", "1", "1.", "1.5", "1e", "1e+", "1e+2" etc.
        if is_valid_number_prefix(candidate):
            valid.add(token_id)

    return valid


def is_valid_number_prefix(s: str) -> bool:
    """Return True if s is a valid prefix of a JSON number."""
    import re
    # Matches complete or partial JSON numbers
    pattern = r'^-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d*)?)?$'
    return bool(re.match(pattern, s))
```

**Tricky part:** you also need to know when the number is **complete** so you can transition to the next state. A number is complete when the next character would be `,` or `}` or whitespace — i.e., when `partial_value` is a valid complete number:

```python
def is_complete_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
```

### `type: "string"`

String values in JSON are wrapped in double quotes: `"hello"`. Inside the string, any character is valid except unescaped `"` and `\`.

```python
def get_valid_string_tokens(
    in_string: bool,        # True if we've opened the '"' already
    string_content: str,    # what's been written inside the string so far
    vocab: dict[int, str]
) -> set[int]:

    if not in_string:
        # Next must be the opening quote
        return {id for id, s in vocab.items() if s == '"'}

    # Inside the string: any token that doesn't prematurely close it
    valid = set()
    for token_id, token_str in vocab.items():
        # Closing quote — valid when we want to end the string
        if token_str == '"':
            valid.add(token_id)
            continue
        # Any token that doesn't contain an unescaped quote is valid content
        if '"' not in token_str and '\\' not in token_str:
            valid.add(token_id)
        # Tokens with escaped quotes (\") are also valid
        elif '\\"' in token_str:
            valid.add(token_id)

    return valid
```

**Note:** tracking whether you're "inside" or "outside" the string requires counting unescaped `"` characters in what's been generated so far for this value.

### `type: "boolean"`

JSON booleans are exactly `true` or `false` (lowercase). This is a fixed set of tokens:

```python
def get_valid_boolean_tokens(
    partial_value: str,
    vocab: dict[int, str]
) -> set[int]:
    """Tokens that continue 'true' or 'false' from partial_value."""
    valid = set()

    for target in ["true", "false"]:
        if target.startswith(partial_value):
            remaining = target[len(partial_value):]
            for token_id, token_str in vocab.items():
                if remaining.startswith(token_str):
                    valid.add(token_id)

    return valid
```

---

## The `-inf` Trick

Negative infinity in Python's float system:

```python
import math
NEGATIVE_INF = -math.inf
```

When you set a logit to `-math.inf`, the softmax calculation gives it a probability of exactly 0:

```
softmax(-inf) = exp(-inf) / sum(...) = 0 / sum(...) = 0
```

So it can **never** be selected — not even if all other tokens have very low logits.

```python
def apply_constraints(
    logits: List[float],
    valid_token_ids: set[int]
) -> List[float]:
    """Mask all tokens not in valid_token_ids to -inf."""
    constrained = logits.copy()

    for i in range(len(constrained)):
        if i not in valid_token_ids:
            constrained[i] = -math.inf

    return constrained
```

> **Watch out:** if `valid_token_ids` is empty (you've made a logic error and no tokens are valid), all logits become `-inf` and `argmax` will return index 0 or behave unpredictably. Always assert `len(valid_token_ids) > 0` before applying constraints.

```python
assert len(valid_token_ids) > 0, (
    f"No valid tokens found at position '{partial_json}' — "
    f"check your state machine logic"
)
```

---

## Complete Walkthrough

Let's trace through generating the output for `"Greet john"` step by step.

**Function definition:**
```json
{"name": "fn_greet", "parameters": {"name": {"type": "string"}}}
```

**Expected output:**
```json
{"name": "fn_greet", "parameters": {"name": "john"}}
```

---

```
partial: ""          state: START
valid:   {"{"}
model picks: "{"
→ partial: "{"

partial: "{"         state: AFTER_OPEN_BRACE
valid:   {'"'}       ← start of '"name"' key
model picks: '"'
→ partial: '{"'

partial: '{"'        state: IN_NAME_KEY
valid:   {"n", "na", "nam", "name"}   ← tokens that start the word 'name'
model picks: "name"
→ partial: '{"name'

partial: '{"name'    state: IN_NAME_KEY
valid:   {'"'}       ← closing the key
model picks: '"'
→ partial: '{"name"'

partial: '{"name"'   state: AFTER_NAME_KEY
valid:   {":", " :", ": "}
model picks: ": "
→ partial: '{"name": '

partial: '{"name": ' state: BEFORE_NAME_VALUE
valid:   {'"'}       ← open the name value string
model picks: '"'
→ partial: '{"name": "'

partial: '{"name": "' state: IN_NAME_VALUE
valid: tokens that are prefixes of valid function names
  "fn_add_numbers" → "fn", "fn_", "fn_a", ...
  "fn_greet"       → "fn", "fn_", "fn_g", ...
  "fn_reverse_string" → ...

model sees "Greet john" → assigns highest logit to "fn_greet" prefix
model picks: "fn_greet"   (or "fn" then "_greet" depending on tokenisation)
→ partial: '{"name": "fn_greet'

partial: '{"name": "fn_greet'   state: IN_NAME_VALUE (complete)
valid:   {'"'}
model picks: '"'
→ partial: '{"name": "fn_greet"'

partial: '{"name": "fn_greet"'   state: AFTER_NAME_VALUE
valid:   {","}
model picks: ","
→ partial: '{"name": "fn_greet",'

... (similar process for '"parameters"' key and ':' and '{') ...

→ partial: '{"name": "fn_greet", "parameters": {"'

partial ends with: '{"'  (inside parameters object, writing first key)
state: IN_ARG_KEY
valid: tokens that start '"name"'  ← only param for fn_greet is "name"
model picks: "name"
→ partial: '{"name": "fn_greet", "parameters": {"name'

→ partial: '{"name": "fn_greet", "parameters": {"name"'
→ partial: '{"name": "fn_greet", "parameters": {"name": '
→ partial: '{"name": "fn_greet", "parameters": {"name": "'

state: IN_ARG_VALUE_STR (type is "string")
valid: all printable non-quote tokens + closing '"'

model extracts "john" from "Greet john" → picks "john"
→ partial: '{"name": "fn_greet", "parameters": {"name": "john'

valid: {'"'}   ← must close the string
model picks: '"'
→ partial: '{"name": "fn_greet", "parameters": {"name": "john"'

state: AFTER_ARG_VALUE
no more params → valid: {"}"}
model picks: "}"
→ partial: '{"name": "fn_greet", "parameters": {"name": "john"}'

state: AFTER_PARAMS_CLOSE
valid: {"}"}
model picks: "}"
→ partial: '{"name": "fn_greet", "parameters": {"name": "john"}}'

state: COMPLETE → stop generation

json.loads('{"name": "fn_greet", "parameters": {"name": "john"}}')
→ {"name": "fn_greet", "parameters": {"name": "john"}}  ✅
```

---

## Putting It All Together

```python
def get_valid_token_ids(
    partial: str,
    fn_def: FunctionDefinition,
    all_fn_names: List[str],
    vocab: dict[int, str],
    state: JSONState,
    context: dict          # tracks: current_param, written_params, etc.
) -> set[int]:
    """
    Central dispatch: given the current state and partial output,
    return the set of token IDs that are valid next tokens.
    """

    if state == JSONState.START:
        return tokens_equal(vocab, "{")

    if state == JSONState.AFTER_OPEN_BRACE:
        return tokens_equal(vocab, '"')

    if state == JSONState.IN_NAME_KEY:
        return tokens_continuing(vocab, '"name"', context["name_key_written"])

    if state == JSONState.AFTER_NAME_KEY:
        return tokens_equal(vocab, ":") | tokens_equal(vocab, ": ") | tokens_equal(vocab, " :")

    if state == JSONState.BEFORE_NAME_VALUE:
        return tokens_equal(vocab, '"')

    if state == JSONState.IN_NAME_VALUE:
        written = context["name_value_written"]
        return tokens_continuing_any(vocab, all_fn_names, written)

    if state == JSONState.AFTER_NAME_VALUE:
        return tokens_equal(vocab, '"')   # closing quote of name value

    # ... continue for all states ...

    if state == JSONState.IN_ARG_VALUE_NUM:
        return get_valid_number_tokens(context["current_value"], vocab)

    if state == JSONState.IN_ARG_VALUE_STR:
        return get_valid_string_tokens(
            in_string=True,
            string_content=context["current_value"],
            vocab=vocab
        )

    if state == JSONState.IN_ARG_VALUE_BOOL:
        return get_valid_boolean_tokens(context["current_value"], vocab)

    raise ValueError(f"Unhandled state: {state}")
```

---

## Common Mistakes

### ❌ Forgetting whitespace tokens

JSON allows optional whitespace between tokens. The model may want to emit ` ` (space) or `\n` between elements. Either allow whitespace tokens at structural positions, or strip them from your partial state tracking.

### ❌ Searching the whole vocabulary in the hot loop

The generation loop runs hundreds of times per prompt. If you search all 150,000 tokens every step, it'll be slow. **Pre-compute** fixed sets (structural chars, number chars) at startup. Only do dynamic searches (function name prefixes, param name prefixes) when in those specific states.

### ❌ Assuming tokens are single characters

A token like `"fn_greet"` might be emitted as one token `fn_greet`, or as `fn` + `_greet`, or as `fn` + `_` + `greet` — it depends on the vocabulary. Your prefix-matching logic must handle all of these.

```python
# Wrong: assuming 'fn_greet' is always one token
valid = {id for id, s in vocab.items() if s == "fn_greet"}

# Correct: allow any token that continues what's needed
def tokens_continuing(vocab, target: str, written: str) -> set[int]:
    remaining = target[len(written):]
    return {id for id, s in vocab.items() if remaining.startswith(s) and s != ""}
```

### ❌ Not handling the closing `}` of the root object

After the parameters object closes with `}`, you still need one more `}` to close the root object. Make sure your state machine has a state for this.

### ❌ Treating `number` as integer-only

The schema says `number` — this means both integers and floats. `2` and `2.0` and `2.5` are all valid. Don't restrict to digits only; allow `.` and `-` and `e`.

### ❌ Hardcoding function names

Your solution must work with any `functions_definition.json`. Never hardcode `"fn_add_numbers"` — always read function names dynamically from the loaded schema.

---

*See also: [`TOKENIZATION.md`](./TOKENIZATION.md) for how the vocabulary file is structured and how to handle BPE token edge cases.*
*See also: [`FUNCTION_CALLING.md`](./FUNCTION_CALLING.md) for the full input/output format and schema structure.*