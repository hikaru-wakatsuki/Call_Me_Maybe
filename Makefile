NAME = call_me_maybe

install:
	uv sync

run:
	uv run python -m src

visualize:
	uv run python -m src --visualize

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest tests/ -s -v

.PHONY: install run debug clean lint lint-strict test
