![FL4Health Logo](assets/fl4health_rect_logo_no_background.png)

# Welcome to FL4Health ✨

A flexible, modular, and easy to use library to facilitate federated learning research and development in healthcare settings.

## Introduction

Principally, this library contains the federated learning (FL) engine aimed at facilitating FL research, experimentation, and exploration, with a specific focus on health applications.

This library is built on the foundational components of [Flower](https://flower.dev/), an open-source FL library in its own right. The documentation is [here](https://flower.dev/docs/framework/index.html). This library contains a number of unique components that extend the functionality of Flower in a number of directions.

## Summary of Currently Implemented Approaches

The present set of FL approaches implemented in the library are:

### Non-Personalized Methods

| Method | Notes |
|--------|-------|
| [FedAvg](https://arxiv.org/abs/1602.05629) | Weights are aggregated on the server-side through averaging. Weighted and unweighted averaging is available. |
| [FedOpt](https://arxiv.org/abs/2003.00295) | A recent extension of FedAvg that includes adaptive optimization on the server-side aggregation. Implementations through Flower include FedAdam, FedAdaGrad, and FedYogi. |
| [FedProx](https://arxiv.org/abs/1812.06127) | An extension of FedAvg that attempts to control local weight drift through a penalty term added to each client's local loss function. Both fixed and adaptive FedProx implementations are available. |
| [SCAFFOLD](https://arxiv.org/abs/1910.06378) | Another extension of FedAvg that attempts to correct for local gradient drift due to heterogenous data representations. This is done through the calculation of control variates used to modify the weight updates. In addition to standard SCAFFOLD, a version with warm-up is implemented, along with DP-SCAFFOLD. |
| [MOON](https://arxiv.org/abs/2103.16257) | MOON adds a contrastive loss function that attempts to ensure that the feature representations learned on the client-side do not significantly drift from those of the previous server model. |
| [FedDG-GA](https://arxiv.org/abs/2103.06030) | FedDG-GA is a domain generalization approach that aims to ensure that the models trained during FL generalize well to unseen domains, potentially outside of the training distribution. The method applies an adjustment algorithm which modifies the client coefficients used during weighted averaging on the server-side. |
| [FLASH](https://proceedings.mlr.press/v202/panchal23a/panchal23a.pdf) | FLASH incorporates a modification to the server-side aggregation algorithm, adding an additional term that is meant to modify the server side learning rate if a data distribution shift occurs during training. In the absence of distribution shifts, the modified aggregation approach is nearly equivalent to the existing FedAvg or FedOpt approaches. |
| [FedPM](https://arxiv.org/pdf/2209.15328) | FedPM is a recent sparse, communication efficient approach to federated learning. The method has been shown to have exceptional information compression while maintaining good performance. Interestingly, it is also connected to the Lottery Ticket Hypothesis. |

### Personalized Methods

| Method | Notes |
|--------|-------|
| [Personal FL](https://arxiv.org/abs/2205.13692) | This method strictly considers the effect of continuing local training on each client model, locally, after federated training has completed. |
| [FedBN](https://arxiv.org/abs/2102.07623) | FedBN implements a very light version of personalization wherein clients exchange all parameters in their model except for anything related to batch normalization layers, which are only learned locally. |
| [FedPer](https://arxiv.org/abs/1912.00818) | Trains a global feature extractor shared by all clients through FedAvg and a private classifier that is unique to each client. |
| [FedRep](https://arxiv.org/abs/2303.05206) | Similar to FedPer, FedRep trains a global feature extractor shared by all clients through FedAvg and a private classifier that is unique to each client. However, FedRep breaks up the client-side training of these components into two phases. First the local classifier is trained with the feature extractor frozen. Next, the classifier is frozen and the feature extractor is trained. |
| [Ditto](https://arxiv.org/abs/2012.04221) | Trains a global model with FedAvg and a personal model that is constrained by the l2-norm of the difference between the personal model weights and the previous global model. |
| [MR-MTL](https://arxiv.org/abs/2206.07902) | Trains a personal model that is constrained by the l2-norm of the difference between the personal model weights and the previous aggregation of all client's models. Aggregation of the personal models is done through FedAvg. Unlike Ditto, no global model is optimized during client-side training. |
| [APFL](https://arxiv.org/abs/2003.13461) | Twin models are trained. One of them is globally shared by all clients and aggregated on the server. The other is strictly trained locally by each client. Predictions are made by a convex combination of the models. |
| [PerFCL](https://ieeexplore.ieee.org/document/10020518/) | PerFCL extends MOON to consider separate globally and locally trained feature extractors and a locally trained classifier. Contrastive loss functions are used to ensure that, during client training, the global features stay close to the original server model features and that the local features are not close to the global features. |
| [FENDA-FL](https://arxiv.org/pdf/2309.16825.pdf) | FENDA is an ablation of PerFCL that strictly considers globally and locally trained feature extractors and a locally trained classifier. The contrastive loss functions are removed from the training procedure to allow for less constrained feature learning and more flexible model architecture design. |
| FENDA+Ditto | This is a combination of two state-of-the-art approaches above: FENDA-FL and Ditto. The idea is to merge the two approaches to yield a "best of both" set of modeling with the flexibility of FENDA-FL for local adaptation and the global-model constrained optimization of Ditto. |
| [GPFL](https://arxiv.org/pdf/2308.10279) | GPFL trains a global feature extractor and a local classifier head. A global mapping model transforms the feature extractor's output into both global and personalized feature representations. Global feature representations are trained to be close to their respective class embeddings stored in a global lookup table, whereas personalized feature embeddings are learned for prediction. |

More approaches are being implemented as they are prioritized. However, the library also provides significant flexibility to implement strategies of your own.

## Privacy Capabilities

In addition to the FL strategies, we also support several differentially private FL training approaches. These include:

- [Instance-level FL privacy](https://arxiv.org/abs/1607.00133)
- [Client-level FL privacy with Adaptive Clipping](https://arxiv.org/abs/1905.03871)
    - Weighted and Unweighted FedAvg

The addition of Distributed Differential Privacy (DDP) with Secure Aggregation is also anticipated soon.

## Community

Need a specific FL algorithm implemented? Submit an issue in our Github, or
even better contribute to our open-source project!

- [FL4Health Python Github](https://github.com/VectorInstitute/FL4Health)
- [FL4Health on PyPI](https://pypi.org/project/fl4health/)
- [FL4Health Contributing Guide](https://github.com/VectorInstitute/FL4Health/blob/main/CONTRIBUTING.MD)

## What's Next?

- [Quick Start](quickstart.md) - Get started with simple federated system in a few lines of code.
- [Module Guides](module_guides/index.md) - Explore the various modules of `fl4health`!
- [Examples](examples/index.md) - A wide-ranging set of FL example tasks.
- [API Reference](api.md) - Thorough documentation of our classes.
