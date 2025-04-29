# Adaptive Latent-Space Constraints in Personalized FL
This is the codebase used for our submission to NeurIPS 2025, titled "Adaptive Latent-Space Constraints in Personalized FL."

## Codebase Structure

The codebase consists of two main parts:

1. **FL4Health**: The main library used to implement our method. It is a fork of the original FL4Health library, a PyTorch-based framework for Federated Learning in healthcare.
2. **researh**: This folder contains training and evaluation scripts for each dataset used in the paper. Each dataset includes its own `README.md` file with instructions on how to run the experiments, including steps for downloading, generating, or partitioning the dataset, as well as running hyperparameter sweeps and evaluation scripts.

**Model checkpoints are available upon request and will be made publicly accessible after the paper is accepted.**

## Requirements
We use [Poetry](https://python-poetry.org/) for dependency management. The library dependencies and those for development and testing are listed in the `pyproject.toml` file. You may use whatever virtual environment management tool that you would like. These include conda, poetry itself, and virtualenv. Poetry is also used to produce our releases, which are managed and automated by GitHub.

The easiest way to create and activate a virtual environment is by using the [virtualenv](https://pypi.org/project/virtualenv/) package:
```bash
python -m venv <ENV_PATH>
source "ENV_PATH/bin/activate"
pip install --upgrade pip poetry
poetry install --with "dev, dev-local, test, codestyle"
```

Note that the with command is installing all libraries required for the full development workflow. See the `pyproject.toml` file for additional details as to what is installed with each of these options.

If you need to update the environment libraries, you should change the requirements in the `pyproject.toml` and then update the `poetry.lock` using the command `poetry update`

## Tests

All tests for the library are housed in the tests folder. The unit and integration tests are run using `pytest`. These tests are automatically run through GitHub integrations on PRs to the main branch of this repository. PRs that fail any of the tests will not be eligible to be merged until they are fixed.

To run all tests in the tests folder one only needs to run (with the venv active)
```bash
pytest .
```
To run a specific test with pytest, one runs
```bash
pytest tests/checkpointing/test_best_checkpointer.py
```
