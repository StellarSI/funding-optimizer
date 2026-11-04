.PHONY: check lint test

# Run the full verification pipeline: linting, formatting check, type-checking,
# and the test suite. Equivalent to the Ruff, Black, mypy and pytest commands
# referenced by the optimizer issues until a formal `make check` is defined.
check: lint test

lint:
	ruff check funding_optimizer tests
	black --check funding_optimizer tests
	mypy funding_optimizer

test:
	pytest -q
