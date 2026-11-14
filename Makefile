PYTHON ?= python
SRC := src/fs_ml
TESTS := tests

setup:
	$(PYTHON) -m pip install -e ".[dev]"

ingest:
	$(PYTHON) -m fs_ml.data configs/ml.yaml

test:
	$(PYTHON) -m pytest $(TESTS)

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)
	$(PYTHON) -m mypy --strict $(SRC)

export:
	$(PYTHON) -m fs_ml.contracts
