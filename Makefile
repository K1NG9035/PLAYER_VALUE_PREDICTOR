.PHONY: data features train evaluate predict test lint all

PYTHON ?= python

data:
	$(PYTHON) -m fifa_value.cli data

features:
	$(PYTHON) -m fifa_value.cli features

train:
	$(PYTHON) -m fifa_value.cli train

evaluate:
	$(PYTHON) -m fifa_value.cli evaluate

predict:
	$(PYTHON) -m fifa_value.cli predict

test:
	pytest tests/

lint:
	ruff check src/ tests/

all: data features train evaluate predict test
