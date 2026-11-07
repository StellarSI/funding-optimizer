.PHONY: help install check lint format format-check typecheck test

help:
	@echo "Available targets:"
	@echo "  make install      - Install package and development dependencies"
	@echo "  make check        - Run full quality suite (lint, format-check, typecheck, test)"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Run black code formatter"
	@echo "  make format-check - Check formatting without modifying files"
	@echo "  make typecheck    - Run mypy static type checker"
	@echo "  make test         - Run pytest test suite"

install:
	python -m pip install -e ".[dev]"

check: lint format-check typecheck test

lint:
	ruff check funding_optimizer tests

format:
	black funding_optimizer tests

format-check:
	black --check funding_optimizer tests

typecheck:
	mypy funding_optimizer tests

test:
	pytest
