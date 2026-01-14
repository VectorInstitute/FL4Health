from collections.abc import Sequence
from logging import INFO, WARNING
from pathlib import Path

import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.functional import one_hot
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.gpfl_base import Gce, GpflModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import clone_and_freeze_model
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class GpflClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
        lam: float = 0.01,
        mu: float = 0.01,
    ) -> None:
        """
        This client is used to perform client-side training associated with the GPFL method described in
        https://arxiv.org/abs/2308.10279.

        In this approach, the client's model is sequentially split into a feature extractor and a head module.
        The client also has two extra modules that are trained alongside the main model: a CoV (Conditional Value),
        and a GCE (Global Category Embedding) module. These sub-modules are trained in the client and shared
        with the server alongside the feature extractor. In simple words, CoV takes in the output of the
        feature extractor (feature_tensor) and maps it into two feature tensors (personal f_p and general f_g)
        computed through affine mapping. `f_p`is fed into the head module for classification, while `f_g` is used
        to train the GCE module. GCE is a lookup table that stores a global representative embedding for each class.
        The GCE module is used to generate two conditional tensors: ``global_conditional_input`` and
        ``personalized_conditional_input`` referred to in the paper as g and p_i, respectively.
        These conditional inputs are then used in the CoV module. All the components are trained simultaneously via
        a combined loss.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda"
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. The module, and its underlying model and state checkpointing
                components will determine when and how to do checkpointing during client-side training.
                No checkpointing (state or model) is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            progress_bar (bool, optional): Whether or not to display a progress bar during client training and
                validation. Uses ``tqdm``. Defaults to False
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
            lam (float, optional): A hyperparameter that controls the weight of the GCE magnitude-level
                global loss. Defaults to 0.01.
            mu (float, optional): A hyperparameter that acts as the weight of the L2 regularization on the GCE and CoV
                modules. This value is used as the optimizers' weight decay parameter. This can be set in
                ``get_optimizer`` function defined by the client user, or if it is not set by the user, it will be
                set in ``set_optimizer`` method. Defaults to 0.01.
        """
        self.model: GpflModel
        self.gce_frozen: Gce

        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
        self.lam = lam
        self.mu = mu
        if self.lam == 0.0:
            log(
                WARNING,
                "Lambda parameter is set to 0.0, which means that the magnitude-level global loss will not be used.",
            )
        # If self.mu is set to 0.0, it means user does not want to use L2 regularization.
        if self.mu == 0.0:
            log(
                WARNING,
                "Mu parameter is set to 0.0, which means that the GCE and CoV modules will not be regularized.",
            )

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        """
        Returns a dictionary with model, gce, and cov optimizers with string keys "model", "gce",
        and "cov" respectively.

        Args:
            config (Config): The config from the server.

        Returns:
            (dict[str, Optimizer]): A dictionary of optimizers defined by the user
        """
        raise NotImplementedError(
            "User Clients must define a function that returns a dict[str, Optimizer] with keys 'model',"
            " 'gce', and 'cov',"
            "defining separate optimizers for different modules of the client."
        )

    def set_optimizer(self, config: Config) -> None:
        """
        This function simply ensures that the optimizers setup by the user have the proper keys
        and that there are three optimizers.

        Args:
            config (Config): The config from the server.
        """
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict) and {"model", "gce", "cov"} == set(optimizers.keys()), (
            "Three optimizers must be defined with keys 'model', 'gce', and 'cov'. Now, only "
            f"{optimizers.keys()} optimizers are defined."
        )
        # If user has specified weight decay for the GCE or CoV optimizers,
        # we will log a warning before overwriting these values with mu.
        user_gce_weight_decay: float = optimizers["gce"].param_groups[0].get("weight_decay", 0.0)
        user_cov_weight_decay: float = optimizers["cov"].param_groups[0].get("weight_decay", 0.0)
        if user_gce_weight_decay != 0.0 or user_cov_weight_decay != 0.0:
            log(
                WARNING,
                "Your gce or cov optimizer weight decay will be overwritten by the mu parameter.",
            )
        # Set the weight decay for the GCE and CoV optimizers to self.mu to enable
        # L2 regularization in the loss.
        log(INFO, f"Setting the GCE optimizer's weight decay to mu = {self.mu}")
        for param_group in optimizers["gce"].param_groups:
            param_group["weight_decay"] = self.mu

        log(INFO, f"Setting the CoV optimizer's weight decay to my = {self.mu}")
        for param_group in optimizers["cov"].param_groups:
            param_group["weight_decay"] = self.mu
        self.optimizers = optimizers

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        GPFL client uses a fixed layer exchanger to exchange layers in three sub-modules.
        Sub-modules to be exchanged are defined in the ``GpflModel`` class.

        Args:
            config (Config): Config from the server..

        Returns:
            (ParameterExchanger): FixedLayerExchanger used to exchange a set of fixed and specific layers.
        """
        assert isinstance(self.model, GpflModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def calculate_class_sample_proportions(self) -> torch.Tensor:
        """
        This method is used to compute the class sample proportions based on the training data.
        It computes the proportion of samples for each class in the training dataset.

        Returns:
            (torch.Tensor): A tensor containing the proportion of samples for each class.
        """
        class_sample_proportion = torch.zeros(self.num_classes, device=self.device)
        one_hot_n_dim = 2  # To avoid having magic numbers
        for _, target in self.train_loader:
            if target.dim() == one_hot_n_dim:  # Target is one-hot encoded
                assert target.shape[1] == self.num_classes, (
                    "Shape of the one-hot encoded labels should be (batch_size, num_classes)."
                )
            else:  # Target is not one-hot encoded
                target = one_hot(target, num_classes=self.num_classes).to(self.device)

            # Compute the proportion of samples for each class by summing the one-hot encoded targets along each column
            # which gives the count of samples per class.
            class_sample_proportion += target.sum(0)

        # Divide the number of samples per class by the total number of samples (sum of all ones).
        class_sample_proportion /= class_sample_proportion.sum()
        return class_sample_proportion

    def setup_client(self, config: Config) -> None:
        """
        In addition to dataloaders, optimizers, parameter exchangers, a few GPFL specific parameters
        are set up in this method. This includes the number of classes, feature dimension,
        and the sample per class tensor. The global and personalized conditional inputs are also initialized.

        Args:
            config (Config): The config from the server.
        """
        super().setup_client(config)

        # Initiate some parameters related to GPFL.
        # ``num_class`` and ``feature_dim`` are essential parts of the GPFL model construction.
        self.num_classes = self.model.num_classes
        self.feature_dim = self.model.feature_dim
        # class_sample_proportion tensor is used to compute personalized conditional input.
        self.class_sample_proportion = self.calculate_class_sample_proportions()

    def compute_conditional_inputs(self) -> None:
        """
        Calculates the conditional inputs (p_i and g) for the CoV module based on the new GCE from the server.
        The ``self.global_conditional_input`` and ``self.personalized_conditional_input`` tensors are computed
        based on a frozen GCE model and the sample per class tensor. These tensors are fixed in each client round,
        and are recomputed when a new GCE module is shared by the server in every client round.
        """
        # Initiate g(global_conditional_input) and p_i(personalized_conditional_input) tensors to zeros.
        self.global_conditional_input = torch.zeros(self.feature_dim).to(self.device)
        self.personalized_conditional_input = torch.zeros(self.feature_dim).to(self.device)

        embeddings = self.gce_frozen.embedding.weight
        for i, embedding in enumerate(embeddings):
            self.global_conditional_input += embedding
            self.personalized_conditional_input += embedding * self.class_sample_proportion[i]

        self.global_conditional_input = embeddings.sum(0) / self.num_classes
        self.personalized_conditional_input = (
            torch.matmul(embeddings.T, self.class_sample_proportion) / self.num_classes
        )

    def update_before_train(self, current_server_round: int) -> None:
        """
        Updates the frozen GCE model and computes the conditional inputs before training starts.

        Args:
            current_server_round (int): The number of current server round.
        """
        # Update the frozen GCE
        cloned_model = clone_and_freeze_model(self.model.gce)
        assert isinstance(cloned_model, Gce)
        self.gce_frozen = cloned_model
        # Update conditional inputs before training
        self.compute_conditional_inputs()

        return super().update_before_train(current_server_round)

    def transform_input(self, input: TorchInputType) -> TorchInputType:
        """
        Extend the input dictionary with ``global_conditional_input`` and ``personalized_conditional_input``
        tensors. This let's use provide these additional tensor to the GPFL model .

        Args:
            input (TorchInputType): Input tensor.

        Returns:
            (TorchInputType): Transformed input tensor.
        """
        # Attach the global and personalized conditional inputs to the input
        if isinstance(input, torch.Tensor):
            return {
                "input": input,
                "global_conditional_input": self.global_conditional_input.detach(),
                "personalized_conditional_input": self.personalized_conditional_input.detach(),
            }
        assert isinstance(input, dict)
        input.update(
            {
                "global_conditional_input": self.global_conditional_input.detach(),
                "personalized_conditional_input": self.personalized_conditional_input.detach(),
            }
        )
        return input

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[TrainingLosses, TorchPredType]:
        """
        Given a single batch of input and target data, generate predictions, compute loss, update parameters and
        optionally update metrics if they exist. (i.e. backprop on a single batch of data).
        Assumes ``self.model`` is in train mode already.

        Args:
            input (TorchInputType): The input to be fed into the model.
            target (TorchTargetType): The target corresponding to the input.

        Returns:
            (tuple[TrainingLosses, TorchPredType]): The losses object from the train step along with
        `            a dictionary of any predictions produced by the model.
        """
        # Clear gradients from the optimizers if they exist
        self.optimizers["model"].zero_grad()
        self.optimizers["gce"].zero_grad()
        self.optimizers["cov"].zero_grad()

        # Call user defined methods to get predictions and compute loss
        input = self.transform_input(input)
        preds, features = self.predict(input)
        target = self.transform_target(target)
        losses = self.compute_training_loss(preds, features, target)

        # Compute backward pass and update parameters with optimizer
        losses.backward["backward"].backward()
        self.transform_gradients(losses)
        self.optimizers["model"].step()
        self.optimizers["gce"].step()
        self.optimizers["cov"].step()

        return losses, preds

    def compute_magnitude_level_loss(
        self,
        global_features: torch.Tensor,
        target: TorchTargetType,
    ) -> torch.Tensor:
        """
        Computes magnitude level loss corresponds to \\(\\mathcal{L}_i^{\text{mlg}}\\) in the paper.

        Args:
            global_features (torch.Tensor): global features computed in this client.
            target (TorchTargetType): Either a tensor of class indices or one-hot encoded tensors.

        Returns:
            (torch.Tensor): L2 norm loss between the global features and the frozen GCE's global features.
        """
        # In magnitude level loss, GCE's embedding table is frozen, and the goal is to train
        # the model to generate good global features by making the generated embeddings closer to
        # frozen GCE's global embeddings.
        assert isinstance(target, torch.Tensor), "GPFL clients take only tensor targets."
        return torch.norm(global_features - self.gce_frozen.lookup(target).detach(), 2)

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes the combined training loss given predictions, global features of the model, and ground truth data.
        GPFL loss is a combined loss and is defined as ``prediction_loss + gce_softmax_loss + magnitude_level_loss``.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (TrainingLosses): An instance of ``TrainingLosses`` containing backward loss and additional losses
                indexed by name.
        """
        # The loss used during training is a combination of the prediction loss (CrossEntropy used in the paper),
        # angel-level (GCE loss) and magnitude-level global losses.
        prediction_loss, _ = self.compute_loss_and_additional_losses(preds, features, target)
        # ``gce_softmax_loss`` corresponds to \mathcal{L}_i^{\text{alg}} in the paper.
        gce_softmax_loss = self.model.gce(features["global_features"], target)
        # ``magnitude_level_loss`` corresponds to \mathcal{L}_i^{\text{mlg}} in the paper.
        magnitude_level_loss = self.compute_magnitude_level_loss(features["global_features"], target)
        # Note that L2 regularization terms are included in the optimizers.
        loss = prediction_loss + gce_softmax_loss + magnitude_level_loss * self.lam
        additional_losses = {
            "prediction_loss": prediction_loss.clone(),
            "gce_softmax_loss": gce_softmax_loss.clone(),
            "magnitude_level_loss": magnitude_level_loss.clone(),
        }
        return TrainingLosses(backward=loss, additional_losses=additional_losses)

    def val_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[EvaluationLosses, TorchPredType]:
        """
        Before performing validation, we need to transform the input and attach the global and personalized
        conditional tensors to the input.

        Args:
            input (TorchInputType): Input based on the training data.
            target (TorchTargetType): The target corresponding to the input..

        Returns:
            (tuple[EvaluationLosses, TorchPredType]: tuple[EvaluationLosses, TorchPredType]):
                The losses object from the val step along with a dictionary of the predictions produced
                by the model.
        """
        input = self.transform_input(input)
        return super().val_step(input, target)
