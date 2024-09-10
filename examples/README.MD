## FL4Health Examples

This folder contains a large collection of example implementations of the various methods and capabilities of the FL4Health library. There are several broad categories covered by these examples, including different state-of-the-art global and personalized FL (pFL) methods, aggregation strategies, pre-processing techniques and more. For deeper descriptions of the methods supported and components of the library, see [README.MD](../README.md).

Below is an index of these examples, followed by a brief description of what they demonstrate. A README is included in each example folder with instructions on how to run the example and a bit more detail on what the example implements. The examples are listed in alphabetical order. For more details on the library, installing the appropriate environment etc., see [README.MD](../README.md)

<table>
<tr>
<th style="text-align: left; width: 250px"> Example Folder </th>
<th style="text-align: center; width: 350px"> Notes </th>
</tr>
<tr>
<td>

[ae_examples](ae_examples)
</td>
<td>
This folder contains examples implementing federated training of auto-encoders (AEs), variational auto-encoders (VAEs), and conditional variational auto-encoders (CVAEs).
</td>
</tr>
<tr>
<td>

[apfl_example](apfl_example)
</td>
<td>
This implements an example of training a model using the pFL approach, APFL. Server-side modifications are minimal, but the client-side training modifications are handled by the appropriate parent class.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[basic_example](basic_example)
</td>
<td>
This is one of the simplest FL examples. It demonstrates federally training a model with basic FedAvg. It also provides an example of using server-side checkpointing and evaluating on a client-side test dataset in every round.

Dataset: CIFAR 10
</td>
</tr>
<tr>
<td>

[ditto_example](ditto_example)
</td>
<td>
This example implements the pFL method Ditto to train individual client models on a heterogeneous version of MNIST. Ditto is one of a small set of methods that actually requires two optimizers for training.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[dp_fed_examples](dp_fed_examples)
</td>
<td>
The examples in this folder are basic implementations of differentially private (DP) FL training. Depending on the threat model, DP ensures that inferring participation of an entity in training (a data point, group of data points, or an entire client) by an attacker is statistically challenging. There are two levels of DP guarantees. Client-level DP refers to guarantees protecting <b>client</b> participation in FL training. Instance-level DP refers to guarantees protecting <b>datapoint</b> participation in FL training. Instance-level DP is classically achieved using DP-SGD on the client-side. The client-level examples include adaptive clipping implementations.

Dataset: CIFAR-10, Medical EHR
</td>
</tr>
<tr>
<td>

[dp_scaffold_example](dp_scaffold_example)
</td>
<td>
This implementation is an example of using the global FL SCAFFOLD method with instance-level differential privacy guarantees (DP-SGD on the client-side). It also leverages the "warm-start" approach to SCAFFOLD.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[dynamic_layer_exchange_example](dynamic_layer_exchange_example)
</td>
<td>
This example demonstrates how one might use the dynamic <b>layer</b> exchanger. This exchanger allows a user to adaptively exchange only a subset of layers that meet certain criterion (set by the user). Facilitates communication efficiency at the cost of some performance.

Dataset: CIFAR-10
</td>
</tr>
<tr>
<td>

[ensemble_example](ensemble_example)
</td>
<td>
Training an ensemble of models is common in many applications. This example provides an implementation of training multiple models individually and simultaneously while measuring ensembled performance at prediction time.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[feature_alignment_example](feature_alignment_example)
</td>
<td>
Provides an implementation of automatic feature alignment for tabular datasets. When data is represented in dataframes, this implementation automatically unifies the shared features and feature spaces to prepare the datasets for FL prior to training.

Dataset: MIMIC-III Tabular Dataset
</td>
</tr>
<tr>
<td>

[fedbn_example](fedbn_example)
</td>
<td>
This is an example of implementing the FedBN approach. The implementation is quite simple in that it is essentially the basic_example but with an exclusion parameter exchanger to exclude batch normalization layers when exchanging weights between the clients and the server.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[feddg_ga_example](feddg_ga_example)
</td>
<td>
This example is a demonstration of using the FedDG-GA strategy. It is a server-side aggregation strategy aimed at providing better model generalization. Clients are essentially unchanged, but the server uses this custom strategy.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[federated_eval_example](federated_eval_example)
</td>
<td>
The example in this folder provides an example of an implementation of federated evaluation, which bypasses federated training. The server is simply used to orchestrate and aggregate client-side evaluation results, where the clients just load a model to be evaluated and an evaluation dataset.

Dataset: CIFAR-10
</td>
</tr>
<tr>
<td>

[fedopt_example](fedopt_example)
</td>
<td>
A fairly simple example, similar to the Basic Example, but demonstrating the use of aggregation strategies in the FedOpt family.

Dataset: AGs News
</td>
</tr>
<tr>
<td>

[fedpca_examples](fedpca_examples)
</td>
<td>
The examples in this folder provide demonstrations of performing federated PCA and subsequently using those principal components for <b>dimensionality reduction</b>. For many applications dimensionality reduction is helpful in improving model performance.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[fedper_example](fedper_example)
</td>
<td>
The implementation in this example provides a demonstration of using the pFL method FedPer to train individual models on each client with global feature extractors and personal classification layers.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[fedpm_example](fedpm_example)
</td>
<td>
The method demonstrated in this example is FedPM, a communication efficient federated learning method for training sparse models. Its success is founded on the Lottery Ticket Hypothesis and related to work on Super Masks for random networks.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[fedprox_example](fedprox_example)
</td>
<td>
This folder contains a fairly straightforward implementation of the classical global FL algorithm FedProx. The implementation here uses the adaptive form of FedProx. It also provides an example of logging FL results to Weights and Biases.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[fedrep_example](fedrep_example)
</td>
<td>
FedRep, which is closely related to FedPer but slightly different, is implemented in this example. It is a pFL method with similar performance to FedPer.

Dataset: CIFAR-10
</td>
</tr>
<tr>
<td>

[fedsimclr_example](fedsimclr_example)
</td>
<td>
In this folder, an example of federated, unsupervised representation learning is implemented. The example includes implementations of the pre-training and fine-tuning phases of such models. Most of the FL components are unchanged in the implementation. However, a special model type is leveraged to facilitate the unsupervised pre-training and supervised fine-tuning stages.

Dataset: CIFAR-10
</td>
</tr>
<tr>
<td>

[fenda_ditto_example](fenda_ditto_example)
</td>
<td>
This example provides a small implementation combining the FENDA-FL and Ditto approaches into a single training approach.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[fenda_example](fenda_example)
</td>
<td>
This is an example of how to use the pFL FENDA-FL approach. The method falls under the category of weight decoupling methods in FL. It uses fairly standard components, except that it has a special model structure and a fixed partial weight exchange (i.e. only a subset of weights are sent to the server).

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[fl_plus_local_ft_example](fl_plus_local_ft_example)
</td>
<td>
This example is a <b>very</b> simple extension of the basic_example, wherein a global model is trained with FedAvg, but then each client does a final epoch of training on their local dataset. This is the "simplest" way of achieving personalized models.

Dataset: CIFAR-10
</td>
</tr>
<tr>
<td>

[flash_example](flash_example)
</td>
<td>
In the FLASH example, an implementation of the aggregation strategy FLASH is used. In general, the implementation is similar to the basic example, but the aggregation strategy is replaced with FLASH.

Dataset: CIFAR-10
</td>
</tr>
<tr>
<td>

[model_merge_example](model_merge_example)
</td>
<td>
Similar to the Federated Evaluation Example, this example implements model merging without any client-side training. In this case, each client simply loads its local model, sends the parameters to the server to be aggregated (with FedAvg), and perhaps evaluate the merged model on their local datasets.

Dataset: N/A
</td>
</tr>
<tr>
<td>

[moon_example](moon_example)
</td>
<td>
An illustrative implementation using the global non-IID FL method MOON is housed in this folder. On the server-side, very little change is required. On the client-side, the MoonClient base class handles the required training modifications, provided the correct model type.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[mr_mtl_example](mr_mtl_example)
</td>
<td>
This example provides an implementation using the MR-MTL pFL approach. As with other examples, the server-side implementation requires little modification, but the client-side training is modified in the appropriate parent class.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[nnunet_example](nnunet_example)
</td>
<td>
nnUNet is a common and effective library for training centralized segmentation models, especially in clinical settings. This example demonstrates how the library can leverage the nnUNet training approach in a federated setting to train good models in a decentralized setting.

Dataset: MSD Dataset: Task04_Hippocampus
</td>
</tr>
<tr>
<td>

[perfcl_example](perfcl_example)
</td>
<td>
A set of code demonstrating the use of PerFCL to federally training personalized models is applied. PerFCL is a personalized extension of MOON and is essentially a loss constrained version of FENDA-FL.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[scaffold_example](scaffold_example)
</td>
<td>
This folder provides an example of using the global SCAFFOLD method to federally train models. The approach requires use of specialized server and client classes to facilitate the necessary gradient modifications and information exchanges/aggregation.

Dataset: MNIST
</td>
</tr>
<tr>
<td>

[sparse_tensor_partial_exchange_example](sparse_tensor_partial_exchange_example)
</td>
<td>
This is an example demonstrating how to use the sparse tensor exchanger implemented in the library. This exchanger allows each client to exchange individual parameters (not just full layers) with the server based on a user specified criteria function. Parameters are aggregated on the server-side using a special form of FedAvg specifically designed to handle heterogeneity in parameters exchanged across clients.

Dataset: CIFAR-10
</td>
</tr>
<tr>
<td>

[warm_up_example](warm_up_example)
</td>
<td>
This folder provides several implementations of what we refer to as warm-start FL. That is, a model is federally trained using some mechanism, perhaps FedAvg. Thereafter, that pre-trained model is used to initialize further FL training using a different method. The examples herein also provide illustrations of "model surgery" wherein only portions of the warm model are transferred into the new model that will receive continued FL training.

Dataset: MNIST
</td>
</tr>
<tr>
<td>
</table>
