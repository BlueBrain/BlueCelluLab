# Variables
PIP_INDEX_URL := https://pypi.python.org/simple
NEURON_MODULE_OPTIONS := '-nogui'
TESTS_DIR := tests
MECHANISMS_DIR := ./../mechanisms/
PYTEST_OPTS := --numprocesses=auto --timeout=600 -vsx --forked

# Default rule (similar to 'testenv' in tox)
default: install_deps compile_mod run_lint run_tests

# Rule for installing dependencies (similar to 'deps' in tox)
install_deps:
	python -m pip install --index-url $(PIP_INDEX_URL) -r test_requirements.txt

# Rule for compiling mod (similar to the first 'commands' in tox)
compile_mod:
	chmod +x ./.compile_mod.sh
	./.compile_mod.sh . $(MECHANISMS_DIR)

# Rule for running tests (similar to the second 'commands' in tox)
run_tests:
	NEURON_MODULE_OPTIONS=$(NEURON_MODULE_OPTIONS) pytest $(TESTS_DIR) $(PYTEST_OPTS)

run_lint:
	ruff check $(TESTS_DIR)
	mypy $(TESTS_DIR) --ignore-missing-imports
