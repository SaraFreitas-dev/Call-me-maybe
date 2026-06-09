## 📚 Resources

### Function Calling
- https://platform.openai.com/docs/guides/function-calling

### LLM Fundamentals
- https://huggingface.co/learn/llm-course

### JSON Schema
- https://json-schema.org/understanding-json-schema/

### Tokenization
- https://huggingface.co/docs/tokenizers/index

### Constrained Decoding & Structured Outputs
- https://platform.openai.com/docs/guides/structured-outputs


```text
call-me-maybe/
│
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   │
│   └── output/
│       └── function_calling_results.json
│
├── src/
│   ├── main.py
│   │
│   ├── models/
│   │   ├── function_definition.py
│   │   ├── function_call.py
│   │   └── prompt.py
│   │
│   ├── parser/
│   │   ├── json_loader.py
│   │   └── json_writer.py
│   │
│   ├── llm/
│   │   ├── llm_client.py
│   │   └── tokenizer.py
│   │
│   ├── decoder/
│   │   ├── constrained_decoder.py
│   │   ├── json_state_machine.py
│   │   └── token_validator.py
│   │
│   ├── services/
│   │   └── function_selector.py
│   │
│   └── utils/
│       └── exceptions.py
│
├── tests/
│
├── README.md
├── Makefile
└── pyproject.toml
```