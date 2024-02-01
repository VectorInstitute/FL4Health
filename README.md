

# FL4Health

Principally, this repository contains the federated learning (FL) engine aimed at facilitating FL research, experimentation, and exploration, with a specific focus on health applications.
<!-- TOC -->
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
    - [Citation](#citation)

The library source code is housed in the `fl4health` folder. This library is built on the foundational components of [Flower](https://flower.dev/), an open-source FL library in its own right. The documentation is [here](https://flower.dev/docs/framework/index.html). This library contains a number of unique components that extend the functionality of Flower in a number of directions.

## Summary of Approaches

The present set of FL approaches implemented in the library are:
<table>
<tr>
<th style='text-align: center'> Non-Personalized FL </th>
<th style='text-align: center'> <div style="width:50px"></div> </th>
<th style='text-align: center'> Personal FL </th>
</tr>
<tr style="border-left: none; border-right: none; border-collapse: collapse;">
<td>

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
- [MOON](https://arxiv.org/abs/2103.16257)
</td>
<td"></td>
<td>

- [Personal FL](https://arxiv.org/abs/2205.13692)
- [FedBN](https://arxiv.org/abs/2102.07623)
- [FedPer](https://arxiv.org/abs/1912.00818)
- [APFL](https://arxiv.org/abs/2003.13461)
- [PerFCL](https://ieeexplore.ieee.org/document/10020518/)
- [FENDA-FL](https://arxiv.org/pdf/2309.16825.pdf)
<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br style="line-height:0px;"/>
</td>
</tr>
</table>

More approaches are being implemented as they are prioritized. However, the library also provides significant flexibiltiy to implement strategies of your own.

## Privacy Capabilities

In addition to the FL strategies, we also support several differentially private FL training approaches. These include:

- [Instance-level FL privacy](https://arxiv.org/abs/1607.00133)
- [Client-level FL privacy with Adaptive Clipping](https://arxiv.org/abs/1905.03871)
    - Weighted and Unweighted FedAvg

The addition of Distributed Differential Privacy (DDP) with Secure Aggregation is also anticipated very soon.

## Components

### Checkpointing

Contains modules associated with basic checkpointing. Currently only supports checkpointing of pytorch models. There are two basic forms of checkpointing available. The first is simply "latest" checkpointing. The second is "best" checkpointing based on a metric value compared with past metrics seen during training. The current implementations support both server-side and client-side checkpointing based on these modules. This allows for what we refer to as "Federated Checkpointing" where, given a validation set on each client, models can be checkpointed at any point during the federated training run, rather than just at the end of the server rounds. This can often significantly improve federally trained model performance. See the experiments implemented in `research/flamby` for an example of using federated checkpointing.

### Client Managers

Houses modules associated with custom functionality on top of Flower's client managers. Client managers are responsible for, among other things, coordinating and sampling clients to participate in server rounds. We support several ways to sample clients in each round, including Poisson based sampling.

### Clients

Here, implementations for specific FL strategies that affect client-side training or enforce certrain properties during training are housed. There is also a basic client that implements standard client-side optimization flows for convenience. For example, the FedProxClient adds the requisite proximal loss term to a provided standard loss prior to performing optimization.

### Feature Alignment

A common problem when working with distributed datasets is a lack of feature alignment. That is, some datasets may contain extra features that others do not, or vice versa. Even if the feature columns are completely shared, there may be issues associated with disjoint feature values. Consider the setting of a categorical feature where a client has more categoires than another. When one-hot encoding these features, there can be a dimensionality mismatch or ordering mismatch in the one-hot representations if the clients are left to independently encode these features. The code in this folder facilitates automatic alignment through an initial communication round between the server and clients. Current there are two supported alignment approaches.
1) The server can provide "oracle" instructions to the clients around how to encode their features.
2) The server samples a client from the pool to provide such a mapping for the feature space and uses that to inform other clients about how to map their own features.

See [Feature Alignment Example](examples/feature_alignment_example) for an example of this process

### Model Bases

Certain methods require special model architectures. For example APFL has twin models and separate global and personal forward passes. It also has a special update function associated with the convex combination parameter $\alpha$. This folder houses special code to facilitate use of these customizations to the neural network architectures.

An interesting model base is the `ensemble_base` which facilitates federally training an ensemble of models rather than just one model at a time. See the [Ensemble Example](examples/ensemble_example) for an example of this usage.

### Parameter Exchange

In vanilla FL, all model weights are exchanged between the server and clients. However, in many cases, either more or less information needs to be exchanged. SCAFFOLD requires that both weights and associated "control variates" be exchanged between the two entities. On the other hand, APFL only exchanges a subset of the parameters. The classes in this folder facilitate the proper handling of both of these situtations. More complicated [adaptive parameter exchange](https://arxiv.org/abs/2205.01557) techniques are also considered here. There is an example of this type of approach in the Examples folder under the [partial_weight_exchange_example](examples/partial_weight_exchange_example).

### Preprocessing

Currently a fairly small module, but will be expanding. This folder contains functionality that can be used to perform steps prior to FL training. This includes dimensionality reduction through a previously trained PCA module or pre-loading weights into models to be trained via FL. The warm-up module supports various kinds of model surgery to inject pre-existing weights into different components of a target pytorch model.

### Privacy

This folder holds the current differential privacy accountants for the instance and client-level DP methods that have been implemented. They are based on the established "Moments Accountants." However, we are working to move these to the new "PRV Accountants."

### Reporting

Currently, this holds the reporting integrations with Weights and Biases for experiment logging. It is capable of capturing both Server and Client metrics. For an example of using this integration, see the [fedprox_example](examples/fedprox_example).

This section also contains functionality associated with metrics tracking during FL training.

### Server

Certain FL methods, such as Client-Level DP and SCAFFOLD with Warm Up, require special server-side flows to ensure that everything is properly handled. This code also establishes initialization communication between the client and server. For example, one can poll each of the clients to obtain the size of each client's dataset before proceeding to FL training. More complex examples of this communication are found in implementations like the feature alignment server.

This section also contains functionality that facilitates running **evaluation only** FL (Federated Evaluation) without performing any training etc. That is useful, for example, if you want to consider the generalization performance across distributed datasets of a model.

Note that Secure Aggregation needs even more complex initial communication, which will be showcased when that functionality is merged.

### Strategies

This folder contains implementations of distinct strategies going beyond those implemented in the standard Flower library. Certain methods require distinct aggregation procedures, such as Client-level differential privacy with adaptive clipping where a noisy aggregation must take place and special considerations are required for the clipping bits. Implementation of new strategies here allows one to customize the way in which parameters and other information communicated between a server and the clients is aggregated.

Note that these strategies are also responsible for unpacking and repacking information that will be sent to the clients. That is, **strategy implementations** are responsible for weight exchange functionality, not the servers.

## Examples

The examples folder contains an extensive set of ways to use the various components of the library, setup the different strategies implemented in the library, and how to run federated learning in general. These examples are an accessbile way to learn what is required to experiment with different FL capabilties. Each example has some documentation describing what is being implemented and how to run the code to see it in action. The examples span basic FedAvg implementations to differentially private SCAFFOLD and beyond.

__NOTE__: The contents of the examples folder is not packed with the FL4Health library on release to PyPi

## Research Code

The research folder houses code associated with various research being conducted by the team at Vector. It may be used to perform experiments on the Cluster or to reproduce experiments from our research. The current research is:

- [FENDA-FL](https://arxiv.org/pdf/2309.16825.pdf) FLamby Experiments. There is a README in that folder that provides details on how to run the hyper-parameter sweeps, evaluations, and other experiments.

__NOTE__: The contents of the research folder is not packed with the FL4Health library on release to PyPi

## Tests

All tests for the library are housed in the tests folder. The unit and integration tests are run using `pytest`, see [Running Tests](./CONTRIBUTING.MD#running-tests) in the contribution markdown. These tests are automatically run through GitHub integrations on PRs to the main branch of this repository. PRs that fail any of the tests will not be eligible to be merged until they are are fixed.

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

We also have automatic smoke tests that run remotely on github in `tests/smoke_tests`. These tests ensure that unintended side-effects are not merged into the library.

__NOTE__: The contents of the tests folder is not packed with the FL4Health library on release to PyPi

## Citation

We hope that the libary will be useful to both FL practioners and researchers working on cutting edge FL applications, with a specific interest in FL for healthcare. If you use FL4Health in a project or in your research, the citation below should be used.
```
D. B. Emerson, J. Jewell, F. Tavakoli, Y. Zhang, S. Ayromlou, M. Lotif, and A. Krishnan (2023). FL4Health. https://github.com/vectorInstitute/FL4Health/. Computer Software, Vector Institute for Artificial Intelligence.

```
