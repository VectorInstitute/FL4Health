from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    MetricsAggregationFn,
    NDArray,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.strategies.basic_fedavg import BasicFedAvg


class FedPCA(BasicFedAvg):
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
        svd_merging: bool = True,
    ) -> None:
        """
        Strategy responsible for performing federated Principal Component Analysis.
        More specifically, this strategy merges client-computed local principal components
        to obtain the principal components for all data.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
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
            svd_merging (bool): Indicates whether merging of client principal components is done by directly performing
                SVD or using a procedure based on QR decomposition. Defaults to True.
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
        # Since federated PCA does not use initial parameters, we fix it here.
        self.initial_parameters = Parameters(tensors=[], tensor_type="numpy.ndarray")
        self.svd_merging = svd_merging

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client parameters. In this case, merge all clients' local principal components.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. In this scheme, the clients pack the layer weights into
                the results object along with the weight values to allow for alignment during aggregation.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated parameters and the metrics dictionary.
                In this case, the parameters are the new singular vectors and their corresponding singular values.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        client_singular_values = []
        client_singular_vectors = []
        for _, fit_res in results:
            A = parameters_to_ndarrays(fit_res.parameters)
            singular_vectors, singular_values = A[0], A[1]
            client_singular_vectors.append(singular_vectors)
            client_singular_values.append(singular_values)

        if self.svd_merging:
            log(INFO, "Performing SVD-based merging.")
            merged_singular_vectors, merged_singular_values = self.merge_subspaces_svd(
                client_singular_vectors, client_singular_values
            )
        else:
            # use qr merging instead
            log(INFO, "Performing QR-based merging.")
            merged_singular_vectors, merged_singular_values = self.merge_subspaces_qr(
                client_singular_vectors, client_singular_values
            )
        parameters = ndarrays_to_parameters([merged_singular_vectors, merged_singular_values])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters, metrics_aggregated

    def merge_subspaces_svd(
        self, client_singular_vectors: NDArrays, client_singular_values: NDArrays
    ) -> Tuple[NDArray, NDArray]:
        """
        Produce the principal components for all the data distributed across clients by merging
        the principal components belonging to each local dataset.

        Each clients sends a matrix whose columns are its local principal components to the server.
        The corresponding singular values are also shared.

        The server arranges the local principal components into a block matrix, then performs SVD.

        More precisely, if U_i denotes the matrix of the principal components of client i, and S_i denotes
        the corresponding diagonal matrix of singular values, and there are n clients, then merging is done by
        performing SVD on the matrix

        B = [U_1 @ S_1 | U_2 @ S_2 | ... | U_n @ S_n],

        where the new left singular vectors are returned as the merging result.

        Notes:

        1. If U_i @ S_i is of size d by N_i, then B has size d by N, where N = N_1 + N_2 + ... + N_n.

        2. If

        U @ S @ V.T = B

        is the SVD of B, then it turns out that U = A @ U',
        where the columns of U' are the true principal components of the aggregated data,
        and A is some block unitary matrix.

        For the theoretical justification behind this procedure, see the paper
        "A Distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks".

        Args:
            client_singular_vectors (NDArrays): Local PCs.
            client_singular_values (NDArrays): Singular values corresponding to local PCs.

        Returns:
            Tuple[NDArray, NDArray]: merged PCs and corresponding singular values.

        Note:
            This method assumes that the *columns* of U_i's are the local principal components.
            Thus, after performing SVD on the matrix B (defined above), the merging result is the
            *left* singular vectors.

            This is in contrast with the client-side implementation of PCA
            (contained in class PcaModule), which assumes that the *rows* of the
            input data matrix are the data points.
            Hence, in PcaModule, the *right* singular vectors of the SVD of
            each client's data matrix are the principal components.

            (In a nutshell, the input data matrices in these two cases are "transposes" of each other.)
        """
        X = [U @ np.diag(S) for U, S in zip(client_singular_vectors, client_singular_values)]
        svd_input = np.concatenate(X, axis=1)
        new_singular_vectors, new_singular_values, _ = np.linalg.svd(svd_input, full_matrices=True)
        return new_singular_vectors, new_singular_values

    def merge_subspaces_qr(
        self, client_singular_vectors: NDArrays, client_singular_values: NDArrays
    ) -> Tuple[NDArray, NDArray]:
        """
        Produce the principal components (PCs) for all the data distributed across clients by merging the PCs
        belonging to each local dataset.

        Each clients sends a matrix whose columns are the local principal components to the server. The corresponding
        singular values are also shared.

        This implementation can be viewed as a more efficient approximation to
        the SVD-based merging in that it does not require performing SVD on a large matrix.

        Directly performing SVD does not take into account the following two observations, suggesting there are more
        efficient algorithms for merging:
            1. Each client's singular vectors are already orthonormal.
            2. The right singular vectors do not need to be computed since
            only the left singular vectors are returned as the merging result.

        In contrast, the algorithm here performs a QR decomposition on the large data matrix, which
        is more efficient than SVD, and SVD is only performed on a much smaller matrix.

        Similarly to the SVD-based merging, it returns an approximation of the true principal components
        of the aggregated data up to the multiplication of some block unitary matrix.

        For the theoretical justification behind this approach, see the paper
        "Subspace Tracking for Latent Semantic Analysis".

        Args:
            client_singular_vectors (NDArrays): Local PCs.
            client_singular_values (NDArrays): Singular values corresponding to local PCs.

        Returns:
            Tuple[NDArray, NDArray]: merged PCs and corresponding singular values.

        Note:
            Similar to merge_subspaces_svd, this method assumes that the *columns* of U_i's are
            the local principal components.
        """
        assert len(client_singular_values) >= 2
        if len(client_singular_values) == 2:
            U1, S1 = client_singular_vectors[0], np.diag(client_singular_values[0])
            U2, S2 = client_singular_vectors[1], np.diag(client_singular_values[1])
            return self.merge_two_subspaces_qr((U1, S1), (U2, S2))
        else:
            U, S = self.merge_subspaces_qr(client_singular_vectors[:-1], client_singular_values[:-1])
            U_last, S_last = client_singular_vectors[-1], client_singular_values[-1]
            return self.merge_two_subspaces_qr((U, np.diag(S)), (U_last, np.diag(S_last)))

    def merge_two_subspaces_qr(
        self, subspace1: Tuple[NDArray, NDArray], subspace2: Tuple[NDArray, NDArray]
    ) -> Tuple[NDArray, NDArray]:
        U1, S1 = subspace1
        U2, S2 = subspace2

        Z = U1.T @ U2
        Q, R = np.linalg.qr(U2 - U1 @ Z)

        d2 = S1.shape[1]
        d1 = R.shape[0]
        zeros = np.zeros(shape=(d1, d2))
        A = np.concatenate((S1, zeros), axis=0)
        B = np.concatenate(((Z @ S2), (R @ S2)), axis=0)
        svd_input = np.concatenate((A, B), axis=1)

        U3, S_final, _ = np.linalg.svd(svd_input, full_matrices=False)

        U_final = (np.concatenate((U1, Q), axis=1)) @ U3

        m, n = U1.shape[0], U1.shape[1] + U2.shape[1]
        rank = min(m, n)
        return U_final[:, :rank], S_final[:rank]
