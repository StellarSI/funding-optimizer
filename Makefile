.PHONY: install check lint format typecheck test clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check funding_optimizer tests

format:
	black funding_optimizer tests

typecheck:
	mypy funding_optimizer tests

test:
	pytest

check: lint typecheck test

clean:
	rm -rf .venv build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
