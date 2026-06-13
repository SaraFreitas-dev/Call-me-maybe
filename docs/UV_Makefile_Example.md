# 📦 UV & Python Makefile Example


```text
UV = uv

FUNCTIONS = data/input/functions_definition.json
INPUT = data/input/function_calling_tests.json
OUTPUT = function_calling_results.json

all: install run

install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ uv is already installed ($$(uv --version))"; \
	else \
		echo "📦 uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

	@echo "📦 Syncing dependencies..."
	@uv sync
	@echo "✅ Dependencies ready"

run:
	@$(UV) run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	@$(UV) run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	@find . -name "__pycache__" -exec rm -rf {} +
	@find . -name "*.pyc" -delete

fclean: clean
	@rm -rf .venv

re: fclean install

.PHONY: all install run debug clean fclean re
```