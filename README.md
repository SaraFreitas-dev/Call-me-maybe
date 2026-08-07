*This project has been created as part of the 42 curriculum by sarfreit.*

---

<p align="center">
  <img src="https://img.shields.io/badge/42-Common%20Core-000000?style=for-the-badge" alt="42"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge" alt="uv"/>
  <img src="https://img.shields.io/badge/model-Qwen3--0.6B-orange?style=for-the-badge" alt="Qwen3"/>
</p>

<h1 align="center">📞 call me maybe</h1>
<p align="center"><i>Teaching a 0.6B parameter model to speak the language of computers — one token at a time.</i></p>

---


## 📑 Table of Contents

1. [What is this project?](#-what-is-this-project)
2. [Background — LLMs, SDKs, and the Hugging Face ecosystem](#-background--llms-sdks-and-the-hugging-face-ecosystem)
3. [Project Structure](#-project-structure)
4. [Reading Order — Where to Start](#-reading-order--where-to-start)
5. [Documentation Index (`docs/`)](#-documentation-index-docs)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [The Makefile](#-the-makefile)
9. [Algorithm Explanation — Constrained Decoding](#-algorithm-explanation--constrained-decoding)
10. [Design Decisions](#-design-decisions)
11. [Bonus Features Implemented](#-bonus-features-implemented)
12. [Performance Analysis](#-performance-analysis)
13. [Challenges Faced](#-challenges-faced)
14. [Testing Strategy](#-testing-strategy)
15. [Example Usage](#-example-usage)
16. [Resources](#-resources)

---

## 🧩 What is this project?

**call me maybe** is a **function-calling engine** for Large Language Models. Given a natural-language request such as:

> *"What is the sum of 2 and 3?"*

...and a list of available functions, the program does **not** ask the model to answer the question directly. Instead, it produces a structured, machine-executable call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

The twist: the model used here (**Qwen/Qwen3-0.6B**) is small — only ~500M parameters — and small models are notoriously unreliable at producing valid JSON on their own (often under 30% success). This project does **not** rely on prompting alone. Instead, it implements **constrained decoding**: at every single token the model generates, the program masks out every token that would break the JSON structure or the function schema, significantly improving JSON and schema reliability by restricting each
generated token to the set of valid continuations.

---

## 🔍 Background — LLMs, SDKs, and the Hugging Face ecosystem

A few concepts worth knowing before diving into the code (all covered in more depth in [`docs/`](#-documentation-index-docs)):

- **LLM (Large Language Model)** — a neural network trained to predict the next token in a sequence of text. It doesn't "understand" JSON or functions; it only ever predicts *one token at a time*, based on probabilities (logits).
- **Tokenizer** — text is not fed to the model as raw characters; it's split into sub-word units called *tokens* (e.g. `"Ġhello"`, where `Ġ` marks a leading space). Each token maps to an integer ID.
- **Hugging Face** — the ecosystem (`transformers`, `huggingface_hub`) this project's underlying model loading is built on. This project does not import `transformers` directly — it only ever talks to the model through the `llm_sdk` package (see below).
- **`llm_sdk`** — a small wrapper package (provided for this project, copied into the repository) exposing a `Small_LLM_Model` class with exactly four public methods this project is allowed to use: `encode`, `decode`, `get_logits_from_input_ids`, and `get_path_to_vocab_file`. No private/internal attributes of `llm_sdk` are used anywhere in this codebase.
- **Constrained decoding** — the core technique of this project: instead of letting the model freely pick whichever token has the highest probability, the *set of allowed tokens* is restricted at every step to only those that keep the output valid, before the model's choice is made.

---

## 🗂 Project Structure

<details>
<summary><b>📁 data/</b></summary>

Holds all JSON input and output. `input/` has the function definitions and test prompts; `output/` is generated at runtime and is gitignored.
</details>

<details>
<summary><b>📁 docs/</b></summary>

Ten topic-by-topic guides: LLM basics, the SDK, tokenization, constrained decoding, Pydantic, schemas, argparse, and uv.
</details>

<details>
<summary><b>📁 llm_sdk/</b></summary>

The provided model wrapper from the 42 School, copied here unmodified. Only its four public methods are used anywhere in this project — never private attributes.
</details>

<details>
<summary><b>📁 src/</b></summary>

Nine modules: CLI entry point, model and vocabulary setup, token validators, the constrained decoder, recovery logic, and JSON I/O.

| File | Role |
|---|---|
| `__main__.py` | CLI entry point — wires everything together |
| `env_setup.py` | Sets `HF_HOME` before any ML library import |
| `llm_engine.py` | Creates the model; builds the prompt text |
| `vocab_loader.py` | Loads vocabulary; builds token categories |
| `schemas.py` | Pydantic models for every data shape |
| `validators.py` | Pure functions: valid tokens per JSON state |
| `constrained_decoder.py` | State machine + the main generation loop |
| `recovery.py` | Retries failed generations; logs recovery |
| `data_loader.py` | Reads JSON inputs; writes JSON outputs |
</details>


```
Call-me-maybe/
├── data/
│   ├── input/
│   │   ├── functions_definition.json   # available function schemas
│   │   └── function_calling_tests.json # prompts to process
│   └── output/                         # generated at runtime (not versioned)
│       ├── function_calling_results.json
│       └── recovery_report.json
├── docs/                                # topic-by-topic documentation (see below)
├── llm_sdk/                             # provided model wrapper (copied, unmodified)
├── src/
│   ├── __main__.py            # CLI entry point — orchestration only
│   ├── env_setup.py           # configures HF_HOME before any ML import
│   ├── llm_engine.py          # model creation + prompt construction
│   ├── vocab_loader.py        # loads & pre-processes the model vocabulary
│   ├── validators.py          # pure functions: valid tokens per JSON state
│   ├── constrained_decoder.py # the state machine + generation loop
│   ├── recovery.py            # retry logic + recovery reporting (bonus)
│   ├── data_loader.py         # JSON I/O for inputs/outputs
│   └── schemas.py             # Pydantic models for all data structures
├── .flake8
├── .gitignore
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 📖 Reading Order — Where to Start

If you're reviewing this project for the first time, this is the recommended order:

1. **`schemas.py`** — the data shapes everything else is built on.
2. **`llm_engine.py`** — how the model is created and how a prompt is built for it.
3. **`vocab_loader.py`** — how the model's vocabulary is turned into a lookup structure.
4. **`validators.py`** — the "rules" — pure functions answering *"which tokens are legal right now?"* for each JSON value type.
5. **`constrained_decoder.py`** — where it all comes together: the state machine (`_get_current_state`, `_get_tokens_for_state`, `_apply_token`) and the main generation loop (`generate_function_call`).
6. **`recovery.py`** — the retry/recovery layer wrapping generation.
7. **`data_loader.py`** — reading inputs, writing outputs.
8. **`__main__.py`** — the thin orchestration layer tying everything together via the CLI.

---

## 📚 Documentation Index (`docs/`)

Each file in `docs/` covers one topic in isolation, so any concept used in the code can be looked up independently:

| File | Covers |
|---|---|
| `LLM_Guide.md` | What a Large Language Model is and how token-by-token generation works |
| `LLM_SDK_Hugging_face.md` | The `llm_sdk` wrapper, its public API, and its relationship to Hugging Face `transformers` |
| `Tokenization.md` | How text becomes tokens, BPE space markers (`Ġ`, `Ċ`), and vocabulary files |
| `Constrained_Decoding.md` | The core technique behind this project — masking logits to guarantee valid structured output |
| `Function_Calling.md` | What function calling is and why it matters for LLM-based systems |
| `Schemas.md` | The Pydantic models used to validate every input and output in this project |
| `Pydantic.md` | General reference for the Pydantic validation library |
| `argparse.md` | Reference for Python's `argparse`, used to build the CLI |
| `UV_Guide.md` | Reference for `uv`, the package manager used for this project |
| `UV_Makefile_Example.md` | How `uv` commands are wired into this project's `Makefile` |

---


## ⚙️ Installation

Requirements: Python 3.10+, and [`uv`](https://docs.astral.sh/uv/).

```bash
make install
```

This installs `uv` automatically if it isn't already available, then runs `uv sync` to install all dependencies (`pydantic`, `numpy`, and the local `llm_sdk` package) into a virtual environment.

> The `llm_sdk/` directory must sit alongside `src/` (already the case in this repository) — `uv` resolves it as a local path dependency, see `pyproject.toml`.

---

## ▶️ Usage

```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>] [--model <name>] [--verbose]
```

| Flag | Default | Description |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | Path to the JSON file describing available functions |
| `--input` | `data/input/function_calling_tests.json` | Path to the JSON file with prompts to process |
| `--output` | `data/output/function_calling_results.json` | Where the generated function calls are written |
| `--model` | `Qwen/Qwen3-0.6B` | Hugging Face model name to load |
| `--verbose` | off | Prints a detailed, per-prompt generation trace |

See all options at any time with:
```bash
uv run python -m src --help
```

---

## 🛠 The Makefile

| Command | What it does |
|---|---|
| `make install` | Installs `uv` (if missing) and syncs all project dependencies |
| `make run` | Runs the program with default arguments |
| `make run-verbose` | Same as `run`, with `--verbose` enabled |
| `make debug` | Runs the program under Python's `pdb` debugger |
| `make lint` | Runs `flake8` + `mypy` with the mandatory flag set |
| `make lint-strict` | Runs `flake8` + `mypy --strict` for stricter checking |
| `make clean` | Removes `__pycache__` and `.pyc` files |
| `make fclean` | `clean` + removes the virtual environment |
| `make re` | `fclean` followed by `install` |
| `make help` | Prints this command list, then runs `uv run python -m src --help` |

---

## 🧠 Algorithm Explanation — Constrained Decoding

For every token the model is about to generate, the pipeline is:

1. **Determine the current JSON state** (`_get_current_state`) from the string generated so far — are we writing the function name? A parameter key? A string value? Are we done?
2. **Compute the set of legal next tokens** for that state (`_get_tokens_for_state`), delegating to a dedicated validator per value type (`validators.py`):
   - `get_valid_name_value_tokens` — only tokens that continue a real function name from `functions_definition.json`.
   - `get_valid_param_key_tokens` — only tokens that continue an un-written parameter name of the selected function.
   - `get_valid_number_tokens` / `get_valid_integer_tokens` — only tokens keeping the value a valid JSON number/integer prefix, offering `,`/`}` once the value is syntactically complete.
   - `get_valid_bool_tokens` — only tokens continuing `true`/`false`, then the closing delimiter.
   - `get_valid_string_tokens` — an **anchored extraction** strategy: a token is only allowed if the resulting partial string is still an exact substring of the original prompt. This prevents the model from inventing content and grounds every string value in the user's actual request.
3. **Mask the logits**: every token outside that legal set is set to `-inf` before the highest-scoring token is picked (`masked_logits`), guaranteeing the model's choice is always structurally valid.
4. **Apply the chosen token** (`_apply_token`) to update the parsing state (which parameter is being written, whether inside a string, etc.).
5. Repeat until the JSON is `"complete"`, then parse and validate it against the selected function's schema.

A dedicated **regex-keyword shortcut** handles one unavoidable limitation of anchored extraction: parameters expecting a regex pattern (e.g. `"replace all vowels"`) can never be extracted from the prompt, because the regex syntax itself never appears there. For these cases, a small keyword-to-pattern map (`REGEX_KEYWORD_MAP`) injects the correct pattern directly.

---

## 🧭 Design Decisions

- **Separation of concerns**: token-validity rules (`validators.py`) are pure functions, independent of the decoder's internal state — easy to reason about and to test in isolation.
- **Anchored string extraction** over free generation: rather than trusting the model to reproduce prompt content verbatim (which small models do unreliably), every string token is checked against the original prompt text before being allowed.
- **Fail loudly, recover gracefully**: any point where the constrained vocabulary would be empty raises a clear `ValueError` immediately, rather than silently falling back to an arbitrary token — see [Performance Analysis](#-performance-analysis) and [Bonus Features](#-bonus-features-implemented).
- **`llm_sdk` isolation**: only the four public methods are used anywhere in this codebase; no private attributes, ever.

---

## 🎁 Bonus Features Implemented

| Feature | Status | Where |
|---|---|---|
| **Support for multiple LLM models** | ✅ | `--model` CLI flag, wired through `llm_engine.create_model()` |
| **Advanced error recovery** | ✅ | `recovery.py` — retries failed generations with an increasing token budget, logs every attempt, and exports a recovery report in the output folder |
| **Visualization of the generation process** | ✅ | `--verbose` flag — prints a formatted, per-prompt trace (state, token, token id, number of valid candidates) and a final success/failure summary on the terminal |

### ✅ How to verify

**Multiple LLM models**
```bash
uv run python -m src --model "Qwen/Qwen3-0.6B" --output data/output/test_default.json
```
The run should complete without crashing and produce valid, schema-compliant JSON in the output file. Any other model name can be passed the same way via `--model`.

**Advanced error recovery**

Force a recovery scenario with a prompt long enough to exceed the initial 200-token budget (the first attempt fails, the retry with a 400-token budget succeeds):

`data/input/recovery_test.json`:
```json
[
  {
    "prompt": "Reverse the string 'this is a very long string deliberately crafted to force the constrained decoder to exceed the initial two hundred token generation budget so that the automatic recovery mechanism kicks in and retries with a larger budget of four hundred tokens which should then be enough to complete the full valid json output successfully without any further issues at all please keep adding more and more words here because the first attempt clearly did not use enough tokens to trigger the failure so this sentence needs to be substantially longer than before with many additional words padding it out until it finally crosses the two hundred token threshold and forces the recovery mechanism to actually kick in in this constrained decoding engine we are testing today for the call me maybe project at fourty two school porto and since one hundred and forty tokens was still not enough we are now adding an entire additional paragraph of padding text to make absolutely certain that this string pushes the total token count comfortably past the two hundred token mark so the recovery mechanism has no choice but to trigger on the very first attempt before succeeding on the second retry with the doubled token budget of four hundred tokens which will definitely be sufficient to hold this much longer piece of text from start to finish without truncation'"
  }
]
```

```bash
uv run python -m src --input data/input/recovery_test.json --output data/output/recovery_test_output.json --verbose
cat data/output/recovery_report.json
```
The terminal should show a failed first attempt (`Generation did not complete within 200 tokens...`) followed by a successful retry, and `recovery_report.json` should list the failed attempt instead of the "no recovery needed" confirmation.

**Visualization of the generation process**
```bash
make run-verbose
```
Prints a per-token trace (`state`, `token_id`, `token`, `valid` candidate count) for every prompt, followed by a per-function success/failure summary.


---

## 📊 Performance Analysis

- **Accuracy**: all 11 example prompts produce a valid, schema-compliant function call.
- **Reliability**: constrained decoding significantly improves the valid JSON structure — the model physically cannot emit a token that breaks the schema, because invalid tokens are masked to `-inf` before selection.
- **Speed**: generation is dominated by the LLM forward pass, called once per generated token; typical prompts complete well within the 5-minute budget for the full test set on standard hardware.
- **Safety net**: if the valid-token set is ever empty (an edge case the state machine did not anticipate), generation fails immediately with a descriptive `ValueError` rather than silently degrading into meaningless output.

---

## 🧩 Challenges Faced

- **Greedy decoding degeneration**: with plain `argmax` token selection, the model would occasionally fall into repeating short token sequences indefinitely (observed on regex-pattern parameters). Anchoring string extraction to the prompt text — plus the regex-keyword shortcut for genuinely un-extractable content — resolved this without resorting to sampling strategies outside this project's scope.
- **State machine edge cases**: distinguishing "just opened a quote" from "just closed a quote", detecting when a structural `:` follows the `"parameters"` key versus a regular key, and preventing premature `}` before every required parameter was written all required careful, iteratively-tested logic in `_get_current_state` / `_get_tokens_for_state`.
- **Token boundary quirks**: the tokenizer sometimes merges a space with the following punctuation (e.g. `" ,"` as a single token), which required normalizing comparisons (`.strip()`) rather than exact-matching raw token text.

---

## 🧪 Testing Strategy

The implementation was validated by running the full `function_calling_tests.json` set end-to-end and manually inspecting `data/output/function_calling_results.json` for schema correctness (right function selected, all and only the expected parameters present, correct types). Constrained-decoding behavior was iteratively verified with `--verbose` tracing, inspecting the token-by-token state transitions for each prompt to confirm the state machine's decisions at every step.

---

## 💡 Example Usage

```bash
make install
make run
```

```bash
uv run python -m src --verbose
```

```
╭─── function 1/11 ──────────────────────────
│ prompt: 'What is the sum of 2 and 3?'
│ [name_value] token='fn'
│ [name_value] token='_add_numbers'
...
│ name: fn_add_numbers
│ parameters: {'a': 2.0, 'b': 3.0}
╰─── ✅ success
```

---

## 📎 Resources

**Function Calling**
- [OpenAI — Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

**JSON Schema**
- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/)

**`uv` Package Manager**
- [uv — Working on Projects](https://docs.astral.sh/uv/guides/projects/)
- [uv — Installation](https://docs.astral.sh/uv/getting-started/installation/)

**Python & JSON**
- [DataCamp — Working with JSON Data in Python](https://www.datacamp.com/tutorial/json-data-python)

**Prompt Engineering** *(used in `llm_engine.build_prompt_request()`)*
- [Hugging Face — Chat Templating: Tool Use & Function Calling](https://huggingface.co/docs/transformers/main/chat_templating#tool-use--function-calling)

**CLI Argument Parsing** *(`argparse`)*
- [Python `argparse` Glossary — Mimo](https://mimo.org/glossary/python/argparse)

**AI usage disclosure**: AI assistance (Claude) was used throughout this project's development to help debug the constrained-decoding state machine (iteratively identifying and fixing edge cases such as quote-parity handling, etc.), to review code structure, and to help draft the documentation for this project.
