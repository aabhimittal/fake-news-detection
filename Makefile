.PHONY: help install install-dev data train predict simulate test clean

PY ?= python3

help:
	@echo "Targets:"
	@echo "  install      Install the package and core dependencies"
	@echo "  install-dev  Install with dev + optional extras"
	@echo "  data         Write the synthetic sample dataset to data/"
	@echo "  train        Train the classifier and save the model"
	@echo "  predict      Score an example headline"
	@echo "  simulate     Run the propagation / containment experiment"
	@echo "  test         Run the unit-test suite"
	@echo "  clean        Remove caches and generated artefacts"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev,viz,api]"

data:
	$(PY) -m fakenews.cli make-data

train:
	$(PY) -m fakenews.cli train

predict:
	$(PY) -m fakenews.cli predict "SHOCKING: doctors HATE this one weird trick, share before it is DELETED!!!" --explain

simulate:
	$(PY) -m fakenews.cli simulate

test:
	$(PY) -m pytest

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info build dist
	rm -f models/*.joblib
