from collections import defaultdict
from functools import reduce
from logging import WARNING
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple, Union

import torch
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from torch import Tensor

from fl4health.parameter_exchange.parameter_packer import SparseCooParameterPacker
from fl4health.strategies.basic_fedavg import BasicFedAvg


class FedAvgSparseCooTensor(BasicFedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
    ) -> None:
        """
        A generalization of the FedAvg strategy where the server can receive any arbitrary subset of parameters from
        any arbitrary subset of the clients.
        Weighted average for parameters belonging to each received tensor is performed independently.

        Note that this strategy differs from FedAvgDynamicLayer in that it does not
        require clients to send entire layers (tensors).
        A client can send an arbitrary set of parameters within a certain tensor,
        and these parameters are packed according to the sparse COO format.

        For more information on the sparse COO format and sparse tensors in PyTorch, please see the following
        two pages:
            1. https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
            2. https://pytorch.org/docs/stable/sparse.html


        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_fit_clients (int, optional): _description_. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            evaluate_fn (Optional[
                Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
            ]):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure server-side central validation by providing a Config dictionary.
                Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Optional[Parameters], optional): Initial global model parameters. Defaults to None.
            fit_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
        )
        self.parameter_packer = SparseCooParameterPacker()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the results from the federated fit round. The aggregation requires some special treatment, as the
        participating clients are allowed to exchange an arbitrary set of parameters. So before aggregation takes place
        alignment must be done using the tensor names packed in along with the weights in the client results.

        More precisely, this method performs the following steps:
            1. Align all tensors according to their names.
            2. For tensors that have the same name, construct the sparse COO tensors and convert them to dense tensors.
            3. Perform averaging on the dense tensors (can be weighted or unweighted).
            4. For every aggregated dense tensor, discard the zero values and retain all information needed
            to represent it in the sparse COO format.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. In this scheme, the clients pack the tensor names into
                the results object along with the weight values to allow for alignment during aggregation.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated model weights and the metrics dictionary.
                For sparse tensor exchange we also pack in the names of all of the tensors that were aggregated in this
                phase to allow clients to insert the values into the proper areas of their models.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert client tensor weights and names into ndarrays
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]

        # For each tensor of the model, perform weighted average of all received weights from clients
        aggregated_tensors = self.aggregate(weights_results)

        tensor_names = []
        selected_parameters_all_tensors = []
        selected_indices_all_tensors = []
        tensor_shapes = []

        for tensor_name, aggregated_tensor in aggregated_tensors.items():
            selected_parameters, selected_indices, tensor_shape = self.parameter_packer.extract_coo_info_from_dense(
                aggregated_tensor
            )
            tensor_names.append(tensor_name)
            selected_parameters_all_tensors.append(selected_parameters)
            selected_indices_all_tensors.append(selected_indices)
            tensor_shapes.append(tensor_shape)

        packed_parameters = self.parameter_packer.pack_parameters(
            selected_parameters_all_tensors, (selected_indices_all_tensors, tensor_shapes, tensor_names)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(packed_parameters), metrics_aggregated

    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, Tensor]:
        """
        Aggregate the different tensors across clients that have contributed to a certain tensor.
        This aggregation may be weighted or unweighted.
        The called functions handle tensor alignment.

        Args:
            results (List[Tuple[NDArrays, int]]): The weight results from each client's local training
            that need to be aggregated on the server-side and the number of training samples
            held on each client.

                In this scheme, the clients pack the tensor names into the results object along with
                the weight values to allow for alignment during aggregation.

        Returns:
            Dict[str, Tensor]: A dictionary mapping the name of the tensor that was aggregated to the aggregated
                weights.
        """
        if self.weighted_aggregation:
            return self.weighted_aggregate(results)
        else:
            return self.unweighted_aggregate(results)

    def weighted_aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, Tensor]:
        """
        "results" consist of four parts: the exchanged (nonzero) parameter values,
        their coordinates within the tensor to which they belong,
        the shape of that tensor, and finally the name of that tensor.

        The first three items constitute the information that is needed
        to construct the tensor in the sparse COO format and convert
        it to a regular dense tensor.
        The tensor name is used to align tensors to ensure that averaging is performed only
        among tensors with the same name.

        This method performs the following steps:
            1. Align all tensors according to their names.
            2. For tensors that have the same name, construct the sparse COO tensors and convert them to dense tensors.
            3. Perform weighted averaging on the dense tensors according to
            the number of training examples each client has.

        Note: this method performs weighted averaging.

        Args:
            results (List[Tuple[NDArrays, int]]): The weight results from each client's local training that need to be
                aggregated on the server-side and the number of training samples held on each client.
                The weight results consist of four parts, as detailed above.
                In this scheme, the clients pack the layer names into the results object along with the weight values
                to allow for alignment during aggregation.

        Returns:
            Dict[str, Tensor]: A dictionary mapping the name of the tensor that was aggregated to the aggregated
                weights.
        """
        names_to_dense_tensors: DefaultDict[str, List[Tensor]] = defaultdict(list)
        total_num_examples: DefaultDict[str, int] = defaultdict(int)

        for packed_parameters, num_examples in results:
            nonzero_parameter_values, additional_info = self.parameter_packer.unpack_parameters(packed_parameters)
            parameter_indices, tensor_shapes, tensor_names = additional_info

            # Sanity check to ensure that they all have the same length and the length is > 0.
            assert (
                len(nonzero_parameter_values) == len(parameter_indices) == len(tensor_shapes) == len(tensor_names)
                and len(tensor_names) > 0
            )
            for tensor_params, tensor_param_indices, tensor_shape, tensor_name in zip(
                nonzero_parameter_values, parameter_indices, tensor_shapes, tensor_names
            ):
                coo_tensor = torch.sparse_coo_tensor(
                    indices=torch.tensor(tensor_param_indices.T),
                    values=torch.tensor(tensor_params),
                    size=torch.Size(tensor_shape),
                )
                dense_tensor = coo_tensor.to_dense()
                names_to_dense_tensors[tensor_name].append(dense_tensor * num_examples)
                total_num_examples[tensor_name] += num_examples

        names_to_tensors_aggregated = {
            name_key: (reduce(torch.add, names_to_dense_tensors[name_key]) / total_num_examples[name_key])
            for name_key in names_to_dense_tensors
        }

        return names_to_tensors_aggregated

    def unweighted_aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, Tensor]:
        """
        "results" consist of four parts: the exchanged (nonzero) parameter values,
        their coordinates within the tensor to which they belong,
        the shape of that tensor, and finally the name of that tensor.

        The first three items constitute the information that is needed to
        construct the tensor in the sparse COO format and convert
        it to a regular dense tensor.
        The tensor name is used to align tensors to ensure that averaging is performed only
        among tensors with the same name.

        This method performs the following steps:
            1. Align all tensors according to their names.
            2. For tensors that have the same name, construct the sparse COO tensors and convert them to dense tensors.
            3. Perform uniform averaging on the dense tensors across all clients.

        Note: this method performs uniform averaging.

        Args:
            results (List[Tuple[NDArrays, int]]): The weight results from each client's local training that need to be
                aggregated on the server-side and the number of training samples held on each client.
                The weight results consist of four parts, as detailed above.
                In this scheme, the clients pack the layer names into the results object along with the weight values
                to allow for alignment during aggregation.

        Returns:
            Dict[str, Tensor]: A dictionary mapping the name of the tensor that was aggregated to the aggregated
                weights.
        """
        names_to_dense_tensors: DefaultDict[str, List[Tensor]] = defaultdict(list)
        total_num_clients: DefaultDict[str, int] = defaultdict(int)

        for packed_parameters, _ in results:
            nonzero_parameter_values, additional_info = self.parameter_packer.unpack_parameters(packed_parameters)
            parameter_indices, tensor_shapes, tensor_names = additional_info

            # Sanity check.
            assert (
                len(nonzero_parameter_values) == len(parameter_indices) == len(tensor_shapes) == len(tensor_names)
                and len(tensor_names) > 0
            )
            for tensor_params, tensor_param_indices, tensor_shape, tensor_name in zip(
                nonzero_parameter_values, parameter_indices, tensor_shapes, tensor_names
            ):
                coo_tensor = torch.sparse_coo_tensor(
                    indices=torch.tensor(tensor_param_indices.T),
                    values=torch.tensor(tensor_params),
                    size=torch.Size(tensor_shape),
                )
                dense_tensor = coo_tensor.to_dense()
                names_to_dense_tensors[tensor_name].append(dense_tensor)
                total_num_clients[tensor_name] += 1

        names_to_tensors_aggregated = {
            name_key: (reduce(torch.add, names_to_dense_tensors[name_key]) / total_num_clients[name_key])
            for name_key in names_to_dense_tensors
        }

        return names_to_tensors_aggregated
