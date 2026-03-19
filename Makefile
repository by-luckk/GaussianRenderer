.PHONY: install lint format typecheck test ci

install:
	uv sync --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/gaussian_renderer

test:
	uv run pytest

ci: lint typecheck test
	@echo "All CI checks passed!"
