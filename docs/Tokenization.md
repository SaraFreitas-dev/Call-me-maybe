# 🧩 Tokenization — How Text Becomes Numbers (and Back)

> **What this doc covers:** What a tokenizer is, how BPE works, how Qwen3's vocabulary is structured, the leading-space problem and why it matters for constrained decoding, and how to use `get_path_to_vocab_file()` effectively in this project.

---

## Table of Contents

1. [What is a Tokenizer?](#what-is-a-tokenizer)
2. [How BPE Works](#how-bpe-works)
3. [The Qwen3 Vocabulary](#the-qwen3-vocabulary)
4. [Using get_path_to_vocab_file()](#using-get_path_to_vocab_file)
5. [The Leading-Space Problem](#the-leading-space-problem)
6. [Special Tokens](#special-tokens)
7. [Tokenization in Practice](#tokenization-in-practice)
8. [Implications for Constrained Decoding](#implications-for-constrained-decoding)
9. [Debugging Tips](#debugging-tips)

---

## What is a Tokenizer?

A tokenizer is the component that sits **between raw text and the model**. It converts strings into sequences of integer IDs that the neural network can process, and converts integer sequences back into strings.

```
Text   →  [Tokenizer]  →  Token IDs  →  [Model]  →  Logits
                                                         ↓
Text   ←  [Tokenizer]  ←  Token IDs  ←    [Argmax / Sampling]
```

Neither the model nor your Python code works with raw strings internally — everything goes through this integer representation.

In this project, tokenization is exposed through two methods:

```python
model.encode("What is the sum of 2 and 3?")
# → tensor([1837, 374, 279, 2629, 315, 220, 17, 323, 220, 18, 30])

model.decode([1837, 374, 279, 2629])
# → "What is the sum"
```

And for constrained decoding, you also use:

```python
model.get_path_to_vocab_file()
# → "/path/to/vocab.json"
# This gives you the full map: token_id → token_string
```

---

## How BPE Works

Qwen3 uses **Byte Pair Encoding (BPE)** — the same algorithm used by most modern LLMs (GPT, Llama, Mistral, etc.).

### The core idea

BPE starts with individual characters and repeatedly merges the most frequent pairs into a single token. After enough merges, common words become single tokens, rare words get split into subword pieces.

### A simple example

Imagine a tiny training corpus containing only these words (with frequencies):

```
"low"     × 5
"lower"   × 2
"newest"  × 6
"widest"  × 3
```

**Step 0 — Start with characters + end-of-word marker:**
```
l o w </w>       × 5
l o w e r </w>   × 2
n e w e s t </w> × 6
w i d e s t </w> × 3
```

**Step 1 — Find most frequent pair:** `e s` appears 9 times (newest + widest). Merge → `es`:
```
l o w </w>
l o w e r </w>
n e w es t </w>
w i d es t </w>
```

**Step 2 — Next most frequent pair:** `es t` appears 9 times. Merge → `est`:
```
l o w </w>
l o w e r </w>
n e w est </w>
w i d est </w>
```

**Step 3 — Next:** `l o` appears 7 times. Merge → `lo`:
```
lo w </w>
lo w e r </w>
n e w est </w>
w i d est </w>
```

This continues for thousands of iterations until the vocabulary reaches its target size. The result is a vocabulary where:
- Very common words (`the`, `is`, `and`) → single tokens
- Common subwords (`ing`, `tion`, `est`) → single tokens
- Rare words → split into pieces (`sub` + `word` + `ization`)
- Individual characters are always available as a fallback

### Why this matters for the project

The vocabulary you get from `get_path_to_vocab_file()` is the **end result** of this BPE process applied to a massive multilingual corpus. It contains ~150,000 entries — everything from single characters to full words to common multi-word sequences.

When you're doing constrained decoding, you can't assume `fn_add_numbers` is a single token. The BPE algorithm may have split it as:

```
"fn_add_numbers"  →  ["fn", "_add", "_numbers"]     # likely
                  →  ["fn_add", "_numbers"]          # possible
                  →  ["fn", "_", "add", "_", "numbers"]  # less likely but possible
```

Your prefix-matching logic must handle all cases. See [Implications for Constrained Decoding](#implications-for-constrained-decoding).

---

## The Qwen3 Vocabulary

### Size

Qwen3-0.6B has a vocabulary of approximately **151,936 tokens**. This means:

- At each generation step, the model outputs ~151,936 logit values
- Your `get_logits_from_input_ids()` call returns a list of this length
- Your valid token set is a subset of `{0, 1, 2, ..., 151935}`

### Structure of the vocab file

The file returned by `get_path_to_vocab_file()` is a JSON file. It maps token IDs (as strings or integers, check the actual file) to their string representations:

```json
{
  "0": "!",
  "1": "\"",
  "2": "#",
  "3": "$",
  ...
  "5476": "{",
  "9313": "}",
  "1": "\"",
  ...
  "19006": "fn",
  ...
}
```

> **⚠️ Check the actual format first.** Load the file and inspect a few entries before writing any code that depends on its structure. The keys might be integers or strings; the values might include special Unicode markers.

```python
import json

vocab_path = model.get_path_to_vocab_file()
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_raw = json.load(f)

# Inspect the structure
print(type(list(vocab_raw.keys())[0]))    # str or int?
print(list(vocab_raw.items())[:10])       # first 10 entries
print(vocab_raw.get("5476") or vocab_raw.get(5476))  # try both
```

### Building your lookup tables

Once you know the format, build these two dictionaries **once** at startup:

```python
def load_vocab(path: str) -> tuple[dict[int, str], dict[str, int]]:
    """
    Load vocabulary file and return:
    - id_to_str: token_id (int) → token_string (str)
    - str_to_id: token_string (str) → token_id (int)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    id_to_str: dict[int, str] = {}
    str_to_id: dict[str, int] = {}

    for key, value in raw.items():
        token_id = int(key)
        token_str = value
        id_to_str[token_id] = token_str
        str_to_id[token_str] = token_id  # note: collisions possible if two IDs map to same string

    return id_to_str, str_to_id
```

> **Collision warning:** Two different token IDs can theoretically map to the same string (e.g. a byte-level fallback and a merged token for the same text). For `str_to_id`, the last write wins. For constrained decoding purposes, `id_to_str` is your primary lookup — you iterate over all token IDs and check their strings.

---

## The Leading-Space Problem

This is one of the most common sources of bugs in constrained decoding implementations. Pay close attention.

### What is it?

In BPE tokenizers, **spaces are attached to the beginning of the following token**, not the end of the previous one. This means the token for `" the"` (space + the) is different from `"the"` (no space).

In the Qwen3 vocabulary file, you'll see this represented with a special Unicode character. Different tokenizers use different conventions:

| Representation | Meaning | Example |
|----------------|---------|---------|
| `Ġ` (U+0120)  | Leading space | `Ġthe` = `" the"` |
| `▁` (U+2581)  | Leading space (SentencePiece) | `▁the` = `" the"` |
| `Ċ` (U+010A)  | Newline | `Ċ` = `"\n"` |
| Raw space `" "` | Literal space token | `" "` |

Qwen3 uses the `Ġ` convention (same as GPT-2 and most Hugging Face models).

### Why this breaks naive lookups

Say you want to find the token ID for `{`. You might write:

```python
open_brace_id = str_to_id["{"]    # ✅ finds the '{' token
```

But what about the colon after `"name"`? In the generated JSON, it appears as `: ` — colon followed by space. But the *next* token after the colon might be `Ġ"` (space + quote), not `"`:

```
'{"name"'  →  next tokens could be:
    ":"    then    'Ġ"'    →  produces '{"name": "'
    ": "   (single token)  →  produces '{"name": '
    ":"    then    '"'     →  produces '{"name":"'   (no space)
```

All three are valid JSON. Your constrained decoder needs to handle all of them.

### Normalizing for comparison

When checking whether a token continues a target string, normalize by replacing the special space marker with an actual space:

```python
def normalize_token(token_str: str) -> str:
    """Convert tokenizer's space markers to actual spaces."""
    return (token_str
            .replace("Ġ", " ")    # leading space (GPT-2 / Qwen style)
            .replace("▁", " ")    # leading space (SentencePiece style)
            .replace("Ċ", "\n")   # newline
            .replace("ĉ", "\t"))  # tab (rare)
```

Use this when matching tokens against target strings:

```python
def tokens_continuing(
    vocab: dict[int, str],
    target: str,
    already_written: str
) -> set[int]:
    """
    Find all token IDs whose normalized string continues 'target'
    from position len(already_written).
    """
    remaining = target[len(already_written):]
    valid = set()

    for token_id, raw_str in vocab.items():
        normalized = normalize_token(raw_str)
        if remaining.startswith(normalized) and normalized != "":
            valid.add(token_id)

    return valid
```

### Example: finding `{` tokens

```python
# Naive — might miss variants
brace_ids = {id for id, s in vocab.items() if s == "{"}

# Better — catches ' {' (with leading space) too, if you want whitespace flexibility
brace_ids = {id for id, s in vocab.items() if normalize_token(s) == "{"}

# Check what you actually have:
for tid in brace_ids:
    print(f"ID {tid}: repr={repr(vocab[tid])}")
# Might print:
# ID 5476:  repr='{'
# ID 90134: repr='Ġ{'    ← space + brace, valid after a comma
```

---

## Special Tokens

Beyond regular text tokens, the vocabulary contains **special tokens** used to control the model's behaviour. These are typically wrapped in angle brackets or similar markers:

```
<|endoftext|>    → end of document
<|im_start|>     → start of a chat turn (instruction model)
<|im_end|>       → end of a chat turn
<|pad|>          → padding token
```

For Qwen3 (an instruction-tuned model), the chat format uses:

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Greet john
<|im_end|>
<|im_start|>assistant
```

### Why this matters for your prompt

If you construct your prompt as a plain string and pass it through `model.encode()`, the tokenizer may or may not add these special tokens depending on how the SDK works. Test this:

```python
prompt = "Greet john"
ids = list(model.encode(prompt))
print(ids)
# Check if there are special token IDs at the start/end
# Compare with the special token IDs from the vocab file
```

If the model expects the chat format and you don't provide it, function selection may be less accurate. If the SDK handles this internally, you don't need to worry about it.

### Excluding special tokens from constrained decoding

Special tokens should **never** appear in your generated JSON output. Add them to a blocklist:

```python
SPECIAL_TOKEN_STRINGS = {
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>",
    # add others as you discover them in the vocab file
}

special_token_ids = {
    id for id, s in vocab.items()
    if s in SPECIAL_TOKEN_STRINGS or (s.startswith("<|") and s.endswith("|>"))
}

# Always exclude these, regardless of state
valid_ids -= special_token_ids
```

---

## Tokenization in Practice

### How your prompt gets tokenized

Let's trace what happens when you call `model.encode()`:

```python
prompt = 'You are a function calling assistant.\n\nAvailable functions:\n- fn_greet(name: string)\n\nUser: "Greet john"\n\nOutput: {"name": "'

ids = list(model.encode(prompt))
```

The tokenizer processes this left to right, greedily merging characters into the longest known BPE token:

```
"You"        → ID 2610  (single token — common word)
" are"       → ID 527   (space included — Ġare)
" a"         → ID 264   (Ġa)
" function"  → ID 734   (Ġfunction)
" calling"   → ID 8260  (Ġcalling)
...
"fn"         → ID 19006 (fn — no leading space, starts after newline+dash)
"_greet"     → ID 34521 (_greet — underscore included)
"(name"      → ID ...
...
"{"          → ID 5476
'"'          → ID 1     (opening quote of name value)
```

The key insight: **the same text tokenizes differently depending on what comes before it.** `fn` at the start of a line (after `\n- `) has no leading space token. The same `fn` in the middle of a sentence might be `Ġfn`.

### Checking a specific string's tokenization

```python
def tokenize_string(model, text: str) -> list[tuple[int, str]]:
    """Show how a string gets tokenized, with IDs and token strings."""
    ids = list(model.encode(text))
    # You'll need the vocab to map back to strings
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = {int(k): v for k, v in json.load(f).items()}
    return [(id, vocab[id]) for id in ids]

# Example
result = tokenize_string(model, "fn_add_numbers")
for token_id, token_str in result:
    print(f"  {token_id:6d}  {repr(token_str)}")

# Might output:
#   19006  'fn'
#   62091  '_add'
#   35839  '_numbers'
```

Run this for all your function names and parameter names during development. It tells you exactly which token sequences you need to allow in your constrained decoder.

---

## Implications for Constrained Decoding

Everything above feeds directly into how you implement constrained decoding. Here's a summary of what to keep in mind:

### 1. A target string may span multiple tokens

When you want to allow the function name `fn_greet`, you can't just look up a single token ID. You need to allow any token that correctly *continues* the name given what's been written so far:

```python
def get_continuation_tokens(
    vocab: dict[int, str],
    target: str,
    written_so_far: str
) -> set[int]:
    """
    Return token IDs that correctly continue 'target' from 'written_so_far'.

    Example:
      target = "fn_greet"
      written_so_far = "fn"
      → returns tokens whose normalized string is a prefix of "_greet"
        i.e. tokens like "_greet", "_gr", "_g", "_" etc.
    """
    remaining = target[len(written_so_far):]
    if not remaining:
        return set()   # target already fully written

    valid = set()
    for token_id, raw_str in vocab.items():
        norm = normalize_token(raw_str)
        # Token is valid if it's a prefix of what still needs to be written
        if remaining.startswith(norm) and norm != "":
            valid.add(token_id)

    return valid
```

### 2. Multiple function names may share a prefix

`fn_add_numbers` and `fn_greet` and `fn_reverse_string` all start with `fn`. When you're at the beginning of the name value, tokens starting with `fn` are valid for all of them:

```python
def get_valid_name_tokens(
    vocab: dict[int, str],
    all_fn_names: list[str],
    written_so_far: str
) -> set[int]:
    """Allow any token that continues at least one valid function name."""
    valid = set()
    for fn_name in all_fn_names:
        if fn_name.startswith(written_so_far):
            valid |= get_continuation_tokens(vocab, fn_name, written_so_far)
    return valid
```

### 3. Pre-compute everything you can

The generation loop runs ~50–200 times per prompt. Searching 150,000 vocabulary entries inside the loop is expensive if done naively. Pre-compute:

```python
class VocabLookup:
    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab
        self.norm_vocab = {id: normalize_token(s) for id, s in vocab.items()}

        # Pre-compute structural token sets (used every generation step)
        self.open_brace   = self._find("{")
        self.close_brace  = self._find("}")
        self.open_quote   = self._find('"')
        self.colon        = self._find_strip(":")
        self.comma        = self._find_strip(",")
        self.numeric      = self._find_numeric()
        self.special      = self._find_special()

    def _find(self, s: str) -> set[int]:
        return {id for id, norm in self.norm_vocab.items() if norm == s}

    def _find_strip(self, s: str) -> set[int]:
        return {id for id, norm in self.norm_vocab.items() if norm.strip() == s}

    def _find_numeric(self) -> set[int]:
        return {id for id, norm in self.norm_vocab.items()
                if norm and all(c in "0123456789.-+eE" for c in norm)}

    def _find_special(self) -> set[int]:
        return {id for id, s in self.vocab.items()
                if s.startswith("<|") and s.endswith("|>")}
```

### 4. Watch out for byte-level fallbacks

Some tokenizers include byte-level tokens like `<0x7B>` (which represents `{` in hex). These are fallback tokens for characters that aren't in the vocabulary as regular tokens. They should be treated equivalently to their character counterparts:

```python
def normalize_token(token_str: str) -> str:
    """Handle byte-level fallback tokens like <0x7B>."""
    if token_str.startswith("<0x") and token_str.endswith(">"):
        try:
            byte_val = int(token_str[3:-1], 16)
            return chr(byte_val)
        except ValueError:
            pass
    # ... regular normalization ...
    return (token_str
            .replace("Ġ", " ")
            .replace("Ċ", "\n"))
```

---

## Debugging Tips

### Print the tokenization of your full prompt

Before implementing constrained decoding, always check how your prompt tokenizes. Look for unexpected splits:

```python
tokens = tokenize_string(model, your_prompt)
print("Full tokenization:")
for i, (tid, tstr) in enumerate(tokens):
    print(f"  [{i:3d}] id={tid:6d}  {repr(tstr)}")
```

### Check all function name tokenizations

```python
fn_names = [fn.name for fn in function_definitions]
print("Function name tokenizations:")
for name in fn_names:
    tokens = tokenize_string(model, name)
    token_strs = [t for _, t in tokens]
    print(f"  '{name}'  →  {token_strs}")
```

This tells you exactly what token sequences to expect when generating function names.

### Find a specific character in the vocabulary

```python
def find_token(vocab: dict[int, str], target: str) -> list[tuple[int, str]]:
    """Find all token IDs whose normalized string matches target."""
    results = []
    for tid, raw in vocab.items():
        if normalize_token(raw) == target:
            results.append((tid, raw))
    return results

print(find_token(vocab, "{"))
# [(5476, '{'), (90134, 'Ġ{')]    ← two tokens for '{', one with leading space
```

### Verify your valid token sets are non-empty

Add assertions during development:

```python
valid = get_valid_token_ids(partial, state, context)
assert len(valid) > 0, (
    f"No valid tokens!\n"
    f"  partial = {repr(partial)}\n"
    f"  state   = {state}\n"
    f"  context = {context}"
)
```

Empty valid sets mean your state machine has a bug — it's entered a state where no token can legally follow.

---

*See also: [`CONSTRAINED_DECODING.md`](./CONSTRAINED_DECODING.md) for how tokenization integrates with the full generation loop.*
*See also: [`LLM_GUIDE.md`](./LLM_GUIDE.md) for the big picture of how tokenization fits into LLM inference.*
