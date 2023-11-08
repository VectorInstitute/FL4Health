

# FL4Health

Principally, this repository contains the federated learning (FL) engine aimed at facilitating FL research, experimentation, and exploration, specifically targetting health applications.
<!-- TOC -->

- [FL4Health](#fl4health)
    - [Summary of Approaches](#summary-of-approaches)
    - [Privacy Capabilities](#privacy-capabilities)
    - [Components](#components)
        - [Checkpointing](#checkpointing)
        - [Client Managers](#client-managers)
        - [Clients](#clients)
        - [Model Bases](#model-bases)
        - [Parameter Exchange](#parameter-exchange)
        - [Privacy](#privacy)
        - [Reporting](#reporting)
        - [Server](#server)
        - [Strategies](#strategies)
    - [Examples](#examples)
    - [Research Code](#research-code)
    - [Tests](#tests)
    - [Development Practices](#development-practices)
        - [Development Requirements](#development-requirements)
        - [Coding Guidelines, Formatters, and Checks](#coding-guidelines-formatters-and-checks)
        - [Running Tests](#running-tests)
    - [Citation](#citation)

<!-- /TOC -->

The library source code is housed in the `fl4health` folder. This library is built on the foundational components of Flower, an open-source FL library in its own right. The documentation is [here](https://flower.dev/docs/framework/index.html). This library contains a number of unique components that extend the functionality of Flower in a number of directions.

## Summary of Approaches

The present set of FL approaches implemented in the library are:

- [FedAvg](https://arxiv.org/abs/1602.05629)
    - Weighted
    - Unweighted
- [FedOpt](https://arxiv.org/abs/2003.00295)
    - FedAdam
    - FedAdaGrad
    - FedYogi
- [FedProx](https://arxiv.org/abs/1812.06127)
    - Adaptive
    - Uniform
- [SCAFFOLD](https://arxiv.org/abs/1910.06378)
    - Standard
    - [With Warmup](https://arxiv.org/abs/2111.09278)
    - [DP-Scaffold](https://arxiv.org/abs/2111.09278)
- [Personal FL (Continued Local Training)](https://arxiv.org/abs/2205.13692)
- [APFL](https://arxiv.org/abs/2003.13461)
- [FENDA-FL]()

More approaches are being implemented as they are prioritized. However, the library also provides significant flexibiltiy to implement strategies of your own.

## Privacy Capabilities

In addition to the FL strategies, we also support several differentially private FL training approaches. These include:

- [Instance-level FL privacy](https://arxiv.org/abs/1607.00133)
- [Client-level FL privacy with Adaptive Clipping](https://arxiv.org/abs/1905.03871)
    - Weighted and Unweighted FedAvg

## Components

### Checkpointing

Contains modules associated with basic checkpointing. Currently only supports checkpointing of pytorch models.

### Client Managers

Houses modules associated with custom functionality on top of Flower's client managers. Client managers are responsible for, among other things, coordinating and sampling clients to participate in server rounds. We support several ways to sample clients in each round.

### Clients

Here, implementations for specific FL strategies that affect client-side training or enforce certrain properties during training are housed. There is also a basic client that implements standard client-side optimization flows for convenience. For example, the FedProxClient adds the requisite proximal loss term to a provided standard loss prior to performing optimization.

### Model Bases

Certain methods require special model architectures. For example APFL has twin models and separate global and personal forward passes. It also has a special update function associated with the convex combination parameter $\alpha$. This folder houses special code to facilitate use of these customizations to the neural network architectures.

### Parameter Exchange

In vanilla FL, all model weights are exchanged between the server and clients. However, in many cases, either more or less information needs to be exchanged. SCAFFOLD requires that both weights and associated "control variates" be exchanged between the two entities. On the other hand, APFL only exchanges a subset of the parameters. The classes in this folder facilitate the proper handling of both of these situtations. More complicated [adaptive parameter exchange](https://arxiv.org/abs/2205.01557) techniques are also considered here. There is an example of this type of approach in the Examples folder under the [partial_weight_exchange_example](examples/partial_weight_exchange_example).

### Privacy

This folder holds the current differential privacy accountants for the instance and client-level DP methods that have been implemented. They are based on the established "Moments Accountants." However, we are working to move these to the new "PRV Accountants."

### Reporting

Currently, this holds the reporting integrations with Weights and Biases for experiment logging. It is capable of capturing both Server and Client metrics. For an example of using this integration, see the [fedprox_example](examples/fedprox_example).

### Server

Certain FL methods, such as Client-Level DP and SCAFFOLD with Warm Up, require special server-side flows to ensure that everything is properly handled. This code also establishes initialization communication between the client and server. For example, one can poll each of the clients to obtain the size of each client's dataset before proceeding to FL training.

### Strategies

This folder contains implementations of distinct strategies going beyond those implemented in the standard Flower library. Certain methods require distinct aggregation procedures, such as Client-level differential privacy with adaptive clipping where a noisy aggregation must take place and special considerations are required for the clipping bits. Implementation of new strategies here allows one to customize the way in which parameters and other information communicated between a server and the clients is aggregated.

## Examples

The examples folder contains an extensive set of ways to use the various components of the library, setup the different strategies implemented in the library, and how to run federated learning in general. These examples are an accessbile way to learn what is required to experiment with different FL capabilties. Each example has some documentation describing what is being implemented and how to run the code to see it in action. The examples span basic FedAvg implementations to differentially private SCAFFOLD.

__NOTE__: The contents of the examples folder is not packed with the FL4Health library on release to PyPi

## Research Code

The research folder houses code associated with various research being conducted by the team at Vector. It may be used to perform experiments on the Cluster or to reproduce experiments from our research. The current research is:

- [FENDA-FL]() FLamby Experiments. There is a README in that folder that provides details on how to run the hyper-parameter sweeps, evaluations, and other experiments.

__NOTE__: The contents of the research folder is not packed with the FL4Health library on release to PyPi

## Tests

All tests for the library are housed in the tests folder. These are run using `pytest`, see [Running Tests](#running-tests) below. These tests are automatically run through GitHub integrations on PRs to the main branch of this repository. PRs that fail any of the tests will not be eligible to be merged until they are are fixed.

If you use VSCode for development, you can setup the tests with the testing integration so that you can run debugging and other IDE features. Setup will vary depending on your VSCode environment, but in your .vscode folder your `settings.json` might look something like

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

__NOTE__: The contents of the tests folder is not packed with the FL4Health library on release to PyPi

## Development Practices

We use the standard git development flow of branch and merge to main with PRs on GitHub. At least one member of the core team needs to approve a PR before it can be merged into main. As mentioned above, tests are run automatically on PRs with a merge target of main. Furthermore, a suite of static code checkers and formatters are also run on said PRs. These also need to pass for a PR to be eligible for merging into the main branch of the library. Currently, such checks run on python3.9.

### Development Requirements

The library dependencies and those for development are listed in the `pyproject.toml` and `requirements.txt` files. You may use whatever virtual environment management tool that you would like. These include conda, poetry, and virtualenv. Poetry is used to produce our releases, which are managed and automated by GitHub.

The easiest way to create and activate a virtual environment is
```bash
virtualenv "ENV_PATH"
source "ENV_PATH/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
```

### Coding Guidelines, Formatters, and Checks

For code style, we recommend the [google style guide](https://google.github.io/styleguide/pyguide.html).

Pre-commit hooks apply [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code formatting.

We also use [flake8](https://flake8.pycqa.org/en/latest/) and [pylint](https://pylint.pycqa.org/en/stable/) for further static code analysis. The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not the least, we use type hints in our code which are checked using [mypy](https://mypy.readthedocs.io/en/stable/). The mypy checks are strictly enforced. That is, all mypy checks must pass or the associated PR will not be mergeable.

The settings for `mypy` are in the `mypy.ini`, settings for `flake8` are contained in the `.flake8` file. Settings for `black` and `isort` come from the `pyproject.toml` and some standard checks are defined directly in the `.pre-commit-config.yaml` settings.

All of these checks and formatters are invoked by pre-commit hooks. These hooks are run remotely on GitHub. In order to ensure that your code conforms to these standards and therefore passes the remote checks, you can install the pre-commit hooks to be run locally. This is done by running (with your environment active)

```bash
pre-commit install
```

To run the checks, some of which will automatically re-format your code to fit the standards, you can run
```bash
pre-commit run --all-files
```
It can also be run on a subset of files by omitting the `--all-files` option and pointing to specific files or folders.

If you're using VSCode for development, pre-commit should setup git hooks that execute the pre-commit checks each time you check code into your branch through the integrated source-control as well. This will ensure that each of your commits conform to the desired format before they are run remotely and without needing to remember to run the checks before pushing to a remote. If this isn't done automatically, you can find instructions for setting up these hooks manually online.

### Code Documentation

For code documentation, we try to adhere to the Google docstring style (See [Here](https://google.github.io/styleguide/pyguide.html), Section: Comments and Docstrings). The implementation of an extensive set of comments for the code in this repository is a work-in-progress. However, we are continuing to work towards a better commented state for the code. For development, as stated in the style guide, __any non-trivial or non-obvious methods added to the library should have a doc string__. For our library this applies only to code added to the main library in `fl4health`. Examples, research code, and tests need not incorporate the strict rules of documentation, though clarifying and helpful comments in those code is __strongly encouraged__.

__NOTE__: As a matter of convention choice, classes are documented through their `__init__` functions rather than at the "class" level.

If you are using VS Code a very helpful integration is available to facilitate the creation of properly formatted docstrings called autoDocstring [VS Code Page](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) and [Documentation](https://github.com/NilsJPWerner/autoDocstring). This tool will automatically generate a docstring template when starting a docstring with triple quotation marks ("""). To get the correct format, the following settings should be prescribed:

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

### Running Tests

We use pytest for our unit and integration testing in the tests folder. These tests are automatically run on GitHub for PRs targeting the main branch. All tests need to pass before merging can happen. To run all tests in the tests folder one only run (with the venv active)

```bash
pytest .
```
To run a specific test with pytest, one runs
```bash
pytest tests/checkpointing/test_best_checkpointer.py
```
where the path is the relative one from the root directory. If you're using VSCode, you can use the integrated debugger from the test suite if you properly configure your project. The settings will depend on your specific environment, but a potential setup is shown above in the [Tests Section](#tests).

## Citation

Reference to cite when you use FL4Health in a project or a research paper:
```
D.B. Emerson, J. Jewell, F. Tavakoli, Y. Zhang, S. Ayromlou, and A. Krishnan (2023). FL4Health. https://github.com/vectorInstitute/FL4Health/. Computer Software, Vector Institute for Artificial Intelligence.

```
