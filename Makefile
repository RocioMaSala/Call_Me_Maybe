SRC_DIR := src
ARGS    ?=


.PHONY: install run debug clean lint lint-strict

all: install

install:
	@echo "Installing poetry..."
	uv sync

run:
	clear
	uv run python -m $(SRC_DIR) $(ARGS)

debug:
	@echo "Debugging simulation with arguments: $(ARGS)"
	uv run python -m pdb -m src $(ARGS)

clean:
	@echo "Cleaning up caches and temporary files..."
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete

lint:
	@echo "Running flake8..."
	uv run flake8 $(SRC_DIR)
	@echo "Running mypy..."
	uv run mypy $(SRC_DIR) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running strict checks..."
	uv run flake8 $(SRC_DIR)
	uv run mypy $(SRC_DIR) --strict
