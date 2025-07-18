

# FL4Health

![FL4Health Logo](./assets/fl4health_rect_logo_no_background.png)
[![Static Code Checks](https://github.com/VectorInstitute/FL4Health/actions/workflows/static_code_checks.yaml/badge.svg)](https://github.com/VectorInstitute/FL4Health/actions/workflows/static_code_checks.yaml)
[![Unit Tests](https://github.com/VectorInstitute/FL4Health/actions/workflows/tests.yaml/badge.svg)](https://github.com/VectorInstitute/FL4Health/actions/workflows/tests.yaml)
[![Smoke tests](https://github.com/VectorInstitute/FL4Health/actions/workflows/smoke_tests.yaml/badge.svg)](https://github.com/VectorInstitute/FL4Health/actions/workflows/smoke_tests.yaml)
[![docs](https://github.com/VectorInstitute/FL4Health/actions/workflows/docs_build.yml/badge.svg)](https://github.com/VectorInstitute/FL4Health/actions/workflows/docs_build.yml/)
[![codecov](https://codecov.io/github/VectorInstitute/FL4Health/branch/main/graph/badge.svg?token=Ty9hHLCF5g)](https://codecov.io/github/VectorInstitute/FL4Health)
![License](https://img.shields.io/github/license/VectorInstitute/FL4Health)

Principally, this repository contains the federated learning (FL) engine aimed at facilitating FL research, experimentation, and exploration, with a specific focus on health applications. **Documentation is hosted at**: https://vectorinstitute.github.io/FL4Health/index.html

- [Summary of Approaches](#summary-of-approaches)
- [Privacy Capabilities](#privacy-capabilities)
- [Components](#components)
- [Examples](#examples)
- [Research Code](#research-code)
- [Contributing](#contributing)
- [Citation](#citation)

The library source code is housed in the `fl4health` folder. This library is built on the foundational components of [Flower](https://flower.dev/), an open-source FL library in its own right. The documentation is [here](https://flower.dev/docs/framework/index.html). This library contains a number of unique components that extend the functionality of Flower in a number of directions.

If you have questions about the library or suggestions for new additions feel free to reach out to david.emerson@vectorinstitute.ai

## Summary of Currently Implemented Approaches

The present set of FL approaches implemented in the library are:
<table>
<tr>
<th style="text-align: left; width: 250px"> Non-Personalized Methods </th>
<th style="text-align: center; width: 350px"> Notes </th>
</tr>
<tr>
<td>

[FedAvg](https://arxiv.org/abs/1602.05629)
</td>
<td>
Weights are aggregated on the server-side through averaging. Weighted and unweighted averaging is available.
</td>
</tr>
<tr>
<td>

[FedOpt](https://arxiv.org/abs/2003.00295)
</td>
<td>
A recent extension of FedAvg that includes adaptive optimization on the server-side aggregation. Implementations through Flower include FedAdam, FedAdaGrad, and FedYogi.
</td>
</tr>
<tr>
<td>

[FedProx](https://arxiv.org/abs/1812.06127)
</td>
<td>
An extension of FedAvg that attempts to control local weight drift through a penalty term added to each client's local loss function. Both fixed and adaptive FedProx implementations are available.
</td>
</tr>
<tr>
<td>

[SCAFFOLD](https://arxiv.org/abs/1910.06378)
</td>
<td>
Another extension of FedAvg that attempts to correct for local gradient drift due to heterogenous data representations. This is done through the calculation of control variates used to modify the weight updates. In addition to standard SCAFFOLD, a version with warm-up is implemented, along with DP-SCAFFOLD.
</td>
</tr>
<tr>
<td>

[MOON](https://arxiv.org/abs/2103.16257)
</td>
<td>
MOON adds a contrastive loss function that attempts to ensure that the feature representations learned on the client-side do not significantly drift from those of the previous server model.
</td>
</tr>
<tr>
<td>

[FedDG-GA](https://arxiv.org/abs/2103.06030)
</td>
<td>
FedDG-GA is a domain generalization approach that aims to ensure that the models trained during FL generalize well to unseen domains, potentially outside of the training distribution. The method applies an adjustment algorithm which modifies the client coefficients used during weighted averaging on the server-side.
</td>
</tr>
<tr>
<td>

[FLASH](https://proceedings.mlr.press/v202/panchal23a/panchal23a.pdf)
</td>
<td>
FLASH incorporates a modification to the server-side aggregation algorithm, adding an additional term that is meant to modify the server side learning rate if a data distribution shift occurs during training. In the absence of distribution shifts, the modified aggregation approach is nearly equivalent to the existing FedAvg or FedOpt approaches.
</td>
</tr>
<tr>
<td>

[FedPM](https://arxiv.org/pdf/2209.15328)
</td>
<td>
FedPM is a recent sparse, communication efficient approach to federated learning. The method has been shown to have exceptional information compression while maintaining good performance. Interestingly, it is also connected to the Lottery Ticket Hypothesis.
</td>
</tr>

<tr>
<th style="text-align: left; width: 250px"> Personalized Methods </th>
<th style="text-align: center; width: 350px"> Notes </th>
</tr>
<tr>
<td>

[Personal FL](https://arxiv.org/abs/2205.13692)
</td>
<td>
This method strictly considers the effect of continuing local training on each client model, locally, after federated training has completed.
</td>
</tr>
<tr>
<td>

[FedBN](https://arxiv.org/abs/2102.07623)
</td>
<td>
FedBN implements a very light version of personalization wherein clients exchange all parameters in their model except for anything related to batch normalization layers, which are only learned locally.
</tr>
<tr>
<td>

[FedPer](https://arxiv.org/abs/1912.00818)
</td>
<td>
Trains a global feature extractor shared by all clients through FedAvg and a private classifier that is unique to each client.
</td>
</tr>
<tr>
<td>

[FedRep](https://arxiv.org/abs/2303.05206)
</td>
<td>
Similar to FedPer, FedRep trains a global feature extractor shared by all clients through FedAvg and a private classifier that is unique to each client. However, FedRep breaks up the client-side training of these components into two phases. First the local classifier is trained with the feature extractor frozen. Next, the classifier is frozen and the feature extractor is trained.
</td>
</tr>
<tr>
<td>

[Ditto](https://arxiv.org/abs/2012.04221)
</td>
<td>
Trains a global model with FedAvg and a personal model that is constrained by the l2-norm of the difference between the personal model weights and the previous global model.
</td>
</tr>
<tr>
<td>

[MR-MTL](https://arxiv.org/abs/2206.07902)
</td>
<td>
Trains a personal model that is constrained by the l2-norm of the difference between the personal model weights and the previous aggregation of all client's models. Aggregation of the personal models is done through FedAvg. Unlike Ditto, no global model is optimized during client-side training.
</td>
</tr>
<tr>
<td>

[APFL](https://arxiv.org/abs/2003.13461)
</td>
<td>
Twin models are trained. One of them is globally shared by all clients and aggregated on the server. The other is strictly trained locally by each client. Predictions are made by a convex combination of the models.
</td>
</tr>
<tr>
<td>

[PerFCL](https://ieeexplore.ieee.org/document/10020518/)
</td>
<td>
PerFCL extends MOON to consider separate globally and locally trained feature extractors and a locally trained classifier. Contrastive loss functions are used to ensure that, during client training, the global features stay close to the original server model features and that the local features are not close to the global features.
</td>
</tr>
<tr>
<td>

[FENDA-FL](https://arxiv.org/pdf/2309.16825.pdf)
</td>
<td>
FENDA is an ablation of PerFCL that strictly considers globally and locally trained feature extractors and a locally trained classifier. The contrastive loss functions are removed from the training procedure to allow for less constrained feature learning and more flexible model architecture design.
</td>
</tr>
<tr>
<td>

FENDA+Ditto
</td>
<td>
This is a combination of two state-of-the-art approaches above: FENDA-FL and Ditto. The idea is to merge the two approaches to yield a "best of both" set of modeling with the flexibility of FENDA-FL for local adaptation and the global-model constrained optimization of Ditto.
</td>
</tr>
<tr>
<td>

[GPFL](https://arxiv.org/pdf/2308.10279)
</td>
<td>
GPFL trains a global feature extractor and a local classifier head. A global mapping model transforms the feature extractor’s output into both global and personalized feature representations. Global feature representations are trained to be close to their respective class embeddings stored in a global lookup table, whereas personalized feature embeddings are learned for prediction.
</td>
</tr>
</table>

More approaches are being implemented as they are prioritized. However, the library also provides significant flexibility to implement strategies of your own.

## Privacy Capabilities

In addition to the FL strategies, we also support several differentially private FL training approaches. These include:

- [Instance-level FL privacy](https://arxiv.org/abs/1607.00133)
- [Client-level FL privacy with Adaptive Clipping](https://arxiv.org/abs/1905.03871)
    - Weighted and Unweighted FedAvg

The addition of Distributed Differential Privacy (DDP) with Secure Aggregation is also anticipated soon.

## Components

### Checkpointing

Contains modules to support checkpointing strategies in federated learning. Currently only supports checkpointing of pytorch models. Generic scoring functions for determining whether to checkpoint a model are supported. There are two basic forms of checkpointing implemented out of the box for convenience. The first is simply "latest" checkpointing. The second is "best" checkpointing based on a metric value compared with past metrics seen during training. The current implementations support both server-side and client-side checkpointing based on these modules. This allows for what we refer to as "Federated Checkpointing" where, given a validation set on each client, models can be checkpointed at any point during the federated training run, rather than just at the end of the server rounds. This can often significantly improve federally trained model performance. See the experiments implemented in `research/flamby` for an example of using federated checkpointing. The library currently supports server-side checkpointing of global models after weight aggregation. On the client-side, we support checkpointing local models on each client during local training and/or after weight aggregation from the server. These are referred to as pre- and post-aggregation checkpointing modules.

### Client Managers

Houses modules associated with custom functionality on top of Flower's client managers. Client managers are responsible for, among other things, coordinating and sampling clients to participate in server rounds. We support several ways to sample clients in each round, including Poisson based sampling.

### Clients

Here, implementations for specific FL strategies that affect client-side training or enforce certain properties during training are housed. There is also a basic client that implements standard client-side optimization flows for convenience. For example, the FedProxClient adds the requisite proximal loss term to a provided standard loss prior to performing optimization.

### Feature Alignment

A common problem when working with distributed datasets is a lack of feature alignment. That is, some datasets may contain extra features that others do not, or vice versa. Even if the feature columns are completely shared, there may be issues associated with disjoint feature values. Consider the setting of a categorical feature where a client has more categories than another. When one-hot encoding these features, there can be a dimensionality mismatch or ordering mismatch in the one-hot representations if the clients are left to independently encode these features. The code in this folder facilitates automatic alignment through an initial communication round between the server and clients. Current there are two supported alignment approaches.
1) The server can provide "oracle" instructions to the clients around how to encode their features.
2) The server samples a client from the pool to provide such a mapping for the feature space and uses that to inform other clients about how to map their own features.

See [Feature Alignment Example](examples/feature_alignment_example) for an example of this process

### Model Bases

Certain methods require special model architectures. For example APFL has twin models and separate global and personal forward passes. It also has a special update function associated with the convex combination parameter $\alpha$. This folder houses special code to facilitate use of these customizations to the neural network architectures.

An interesting model base is the `ensemble_base` which facilitates federally training an ensemble of models rather than just one model at a time. See the [Ensemble Example](examples/ensemble_example) for an example of this usage.

### Parameter Exchange

In vanilla FL, all model weights are exchanged between the server and clients. However, in many cases, either more or less information needs to be exchanged. SCAFFOLD requires that both weights and associated "control variates" be exchanged between the two entities. On the other hand, APFL only exchanges a subset of the parameters. The classes in this folder facilitate the proper handling of both of these situations. More complicated [adaptive parameter exchange](https://arxiv.org/abs/2205.01557) techniques are also considered here. There is an example of this type of approach in the Examples folder under the [partial_weight_exchange_example](examples/partial_weight_exchange_example).

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

The examples folder contains an extensive set of ways to use the various components of the library, setup the different strategies implemented in the library, and how to run federated learning in general. These examples are an accessible way to learn what is required to experiment with different FL capabilities. Each example has some documentation describing what is being implemented and how to run the code to see it in action. The examples span basic FedAvg implementations to differentially private SCAFFOLD and beyond.

> [!NOTE]
> The contents of the examples folder is not packed with the FL4Health library on release to PyPi

## Research Code

The research folder houses code associated with various research being conducted by the team at Vector. It may be used to perform experiments on the Cluster or to reproduce experiments from our research. The current research is:

- [FENDA-FL](https://arxiv.org/pdf/2309.16825.pdf) FLamby and GEMINI Experiments (training scripts and model code only). There is a README in these folders that provide details on how to run the hyper-parameter sweeps, evaluations, and other experiments. The GEMINI experiments are not runnable in a general environment, as they require access to GEMINI data, which is securely held, due to patient privacy.

> [!NOTE]
> The contents of the research folder is not packed with the FL4Health library on release to PyPi

## Contributing

If you are interested in contributing to the library, please see [CONTRIBUTION.MD](CONTRIBUTING.MD). This file contains many details around contributing to the code base, including are development practices, code checks, tests, and more.

## Citation

We hope that the library will be useful to both FL practitioners and researchers working on cutting edge FL applications, with a specific interest in FL for healthcare. If you use FL4Health in a project or in your research, the citation provided by the "Cite this Repository" widget should be used.

As examples (the version may be out of date)

```
# APA Style
Emerson, D.B., Jewell, J., Ayromlou, S., Carere, S., Tavakoli, F., Zhang, Y., Lotif, M., Fajardo, V. A., & Krishnan, A. FL4Health (Version 0.3.0) [Computer software]. https://doi.org/10.5281/zenodo.1234
```

```bibtex
# Bibtex Entry
@software{Emerson_FL4Health,
author = {Emerson, David B. and Jewell, John and Ayromlou, Sana and Carere, Shawn and Tavakoli, Fatemeh and Zhang, Y. and Lotif, Marcello and Fajardo, Val Andrei and Krishnan, A.},
doi = {10.5281/zenodo.1234},
license = {Apache-2.0},
title = {{FL4Health}},
url = {https://github.com/vectorInstitute/FL4Health/},
version = {0.3.0}
}
```
