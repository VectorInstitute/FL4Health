# Contributing

Thanks for your interest in contributing to the FL4Health library!

To submit PRs, please fill out the PR template along with the PR. If the PR fixes an issue, please include a link to the PR to the issue, if possible. Below are some details around important things to consider before contributing to the library. A table of contents also appears below for navigation.

- [Development Practices](#development-practices)
- [Development Requirements](#development-requirements)
- [Coding Guidelines, Formatters, and Checks](#coding-guidelines-formatters-and-checks)
- [Code Documentation](#code-documentation)
- [Tests](#tests)

## Development Practices

We use the standard git development flow of branch and merge to main with PRs on GitHub. At least one member of the core team needs to approve a PR before it can be merged into main. As mentioned above, tests are run automatically on PRs with a merge target of main. Furthermore, a suite of static code checkers and formatters are also run on said PRs. These also need to pass for a PR to be eligible for merging into the main branch of the library. Currently, such checks run on python3.9.

## Development Requirements

For development and testing, we use [Poetry](https://python-poetry.org/) for dependency management. The library dependencies and those for development and testing are listed in the `pyproject.toml` file. You may use whatever virtual environment management tool that you would like. These include conda, poetry itself, and virtualenv. Poetry is also used to produce our releases, which are managed and automated by GitHub.

The easiest way to create and activate a virtual environment is by using the [virtualenv](https://pypi.org/project/virtualenv/) package:
```bash
virtualenv "ENV_PATH"
source "ENV_PATH/bin/activate"
pip install --upgrade pip poetry
poetry install --with "dev, dev-local, test, codestyle"
```

Note that the with command is installing all libraries required for the full development workflow. See the `pyproject.toml` file for additional details as to what is installed with each of these options.

If you need to update the environment libraries, you should change the requirements in the `pyproject.toml` and then update the `poetry.lock` using the command `poetry update`

## Coding Guidelines, Formatters, and Checks

For code style, we recommend the [google style guide](https://google.github.io/styleguide/pyguide.html).

Pre-commit hooks apply [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code formatting.

We also use [flake8](https://flake8.pycqa.org/en/latest/) and [pylint](https://pylint.pycqa.org/en/stable/) for further static code analysis. The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not least, we use type hints in our code which are checked using [mypy](https://mypy.readthedocs.io/en/stable/). The mypy checks are strictly enforced. That is, all mypy checks must pass or the associated PR will not be merge-able.

The settings for `mypy` are in the `mypy.ini`, settings for `flake8` are contained in the `.flake8` file. Settings for `black` and `isort` come from the `pyproject.toml` and some standard checks are defined directly in the `.pre-commit-config.yaml` settings.

All of these checks and formatters are invoked by pre-commit hooks. These hooks are run remotely on GitHub. In order to ensure that your code conforms to these standards, and, therefore, passes the remote checks, you can install the pre-commit hooks to be run locally. This is done by running (with your environment active)

**Note**: We use the modern mypy types introduced in Python 3.10 and above. See some of the [documentation here](https://mypy.readthedocs.io/en/stable/builtin_types.html)

For example, this means that we're using `list[str], tuple[int, int], tuple[int, ...], dict[str, int], type[C]` as built-in types and `Iterable[int], Sequence[bool], Mapping[str, int], Callable[[...], ...]` from collections.abc (as now recommended by mypy).

We are also moving to the new Optional and Union specification style:
```python
Optional[typing_stuff] -> typing_stuff | None
Union[typing1, typing2] -> typing1 | typing2
Optional[Union[typing1, typing2]] -> typing1 | typing2 | None
```

```bash
pre-commit install
```

To run the checks, some of which will automatically re-format your code to fit the standards, you can run
```bash
pre-commit run --all-files
```
It can also be run on a subset of files by omitting the `--all-files` option and pointing to specific files or folders.

If you're using VS Code for development, pre-commit should setup git hooks that execute the pre-commit checks each time you check code into your branch through the integrated source-control as well. This will ensure that each of your commits conform to the desired format before they are run remotely and without needing to remember to run the checks before pushing to a remote. If this isn't done automatically, you can find instructions for setting up these hooks manually online.

## Code Documentation

For code documentation, we try to adhere to the Google docstring style (See [here](https://google.github.io/styleguide/pyguide.html), Section: Comments and Doc-strings). The implementation of an extensive set of comments for the code in this repository is a work-in-progress. However, we are continuing to work towards a better commented state for the code. For development, as stated in the style guide, __any non-trivial or non-obvious methods added to the library should have a doc string__. For our library this applies only to code added to the main library in `fl4health`. Examples, research code, and tests need not incorporate the strict rules of documentation, though clarifying and helpful comments in that code is also __strongly encouraged__.

> As a matter of convention choice, classes are documented through their `__init__` functions rather than at the "class" level.

If you are using VS Code a very helpful integration is available to facilitate the creation of properly formatted doc-strings called autoDocstring [VS Code Page](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) and [Documentation](https://github.com/NilsJPWerner/autoDocstring). This tool will automatically generate a docstring template when starting a docstring with triple quotation marks (`"""`). To get the correct format, the following settings should be prescribed in your VS Code settings JSON:

```json
{
    "autoDocstring.customTemplatePath": "",
    "autoDocstring.docstringFormat": "google",
    "autoDocstring.generateDocstringOnEnter": true,
    "autoDocstring.guessTypes": true,
    "autoDocstring.includeExtendedSummary": false,
    "autoDocstring.includeName": false,
    "autoDocstring.logLevel": "Info",
    "autoDocstring.quoteStyle": "\"\"\"",
    "autoDocstring.startOnNewLine": true
}
```

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

If you use VS Code for development, you can setup the tests with the testing integration so that you can run debugging and other IDE features. Setup will vary depending on your VS Code environment, but in your .vscode folder your `settings.json` might look something like

``` JSON
{
    "python.testing.unittestArgs": [
        "-v",
        "-s",
        ".",
        "-p",
        "test_*.py"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "."
    ]
}
```

In addition to the unit and integration tests through `pytest` a number of **smoke tests** have been implemented and are run remotely on github as well. These tests are housed in `tests/smoke_tests`. All of these smoke tests must also pass for a PR to be eligible for merging to main. These smoke tests ensure that changes to note unintentionally break or alter the current functionality of many of our examples. This helps to ensure that code changes do not have unintended side-effects on already tested and/or working code.

### Code Coverage

For code coverage, we use [Codecov](https://about.codecov.io/) (by Sentry) and have configured this tool to pass only if a PR's overall code coverage is above 80%.

> The contents of the tests folder is not packed with the FL4Health library on release to PyPi
