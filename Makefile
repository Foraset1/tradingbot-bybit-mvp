.PHONY: install test lint typecheck check topics validate collect smoke audit

install:
	python3 -m pip install -e '.[dev]'

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

typecheck:
	python3 -m mypy

check: lint typecheck test

topics:
	python3 -m tradingbot show-topics

validate:
	python3 -m tradingbot validate-config

collect:
	python3 -m tradingbot collect

smoke:
	python3 -m tradingbot collect --run-seconds 90

audit:
	python3 -m tradingbot audit-data
