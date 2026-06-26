UV = uv

HF_HOME = /sgoinfre/sarfreit/hf_cache
FUNCTIONS = data/input/functions_definition.json
INPUT = data/input/function_calling_tests.json
OUTPUT = function_calling_results.json

export HF_HOME

all: install run

# INSTALL ALL REQUIREMENTS
install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ uv is already installed ($$(uv --version))"; \
	elif [ -f "$$HOME/.local/bin/uv" ]; then \
		echo "✅ uv found in $$HOME/.local/bin"; \
	else \
		echo "📦 uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "📦 Syncing dependencies..."
	@PATH="$$HOME/.local/bin:$$PATH" uv sync
	@echo "✅ Dependencies ready"

# RUN THE PROGRAM
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


# CHECK FOR NORM ERRORS
lint:
	@echo "🔍 Running flake8 and mypy..."
	@$(UV) run flake8 .
	@$(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
	@echo "✅ Lint completed"

lint-strict:
	@echo "🧠 Running strict checks..."
	@$(UV) run flake8 .
	@$(UV) run mypy . --strict
	@echo "✅ Strict lint completed"

# CLEANERS
clean:
	@echo "\n🧹 Cleaning cache files..."
	@find . -name "__pycache__" -exec rm -rf {} +
	@find . -name "*.pyc" -delete
	@echo "\n✅ Partial clean complete\n"

fclean: clean
	@echo "\n💣 Removing virtual environment..."
	@rm -rf .venv
	@echo "\n✅ Full clean complete\n"

re: fclean install

.PHONY: all install run debug lint lint-strict clean fclean re
