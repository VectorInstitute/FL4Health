from collections.abc import Callable
from logging import INFO, WARNING

import numpy as np
from flwr.common import MetricsAggregationFn, NDArray, NDArrays, Parameters, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.functions import decode_and_pseudo_sort_results


MINIMUM_PCA_ClIENTS = 2
EVALUATE_FN_TYPE = Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None


class FedPCA(BasicFedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: EVALUATE_FN_TYPE = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
        svd_merging: bool = True,
    ) -> None:
        """
        Strategy responsible for performing federated Principal Component Analysis. More specifically, this strategy
        merges client-computed local principal components to obtain the principal components for all data.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_fit_clients (int, optional): Minimum number of clients used during fit. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of clients before starting FL. Defaults to 2.
            evaluate_fn (EVALUATE_FN_TYPE, optional): Optional function used for central server-side evaluation.
                Defaults to None.
            on_fit_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                client-side validation by providing a ``Config`` dictionary. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Parameters | None, optional): Initial global model parameters. Defaults to None.
            fit_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function. Defaults
                to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts. Defaults to
                True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
            svd_merging (bool, optional): Indicates whether merging of client principal components is done by directly
                performing SVD or using a procedure based on QR decomposition. Defaults to True.
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
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate client parameters. In this case, merge all clients' local principal components.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (list[tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. In this scheme, the clients pack the layer weights into
                the results object along with the weight values to allow for alignment during aggregation.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            (tuple[Parameters | None, dict[str, Scalar]]): The aggregated parameters and the metrics dictionary.
                In this case, the parameters are the new singular vectors and their corresponding singular values.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Sorting the results by elements and sample counts. This is primarily to reduce numerical fluctuations in
        # summing the numpy arrays during aggregation. This ensures that addition will occur in the same order,
        # reducing numerical fluctuation.
        decoded_and_sorted_results = [weights for _, weights, _ in decode_and_pseudo_sort_results(results)]

        client_singular_values = []
        client_singular_vectors = []
        for a in decoded_and_sorted_results:
            singular_vectors, singular_values = a[0], a[1]
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
    ) -> tuple[NDArray, NDArray]:
        """
        Produce the principal components for all the data distributed across clients by merging the principal
        components belonging to each local dataset.

        Each clients sends a matrix whose columns are its local principal components to the server. The corresponding
        singular values are also shared.

        The server arranges the local principal components into a block matrix, then performs SVD.

        More precisely, if ``U_i`` denotes the matrix of the principal components of client i, and ``S_i`` denotes
        the corresponding diagonal matrix of singular values, and there are n clients, then merging is done by
        performing SVD on the matrix

        ``B = [U_1 @ S_1 | U_2 @ S_2 | ... | U_n @ S_n],``

        where the new left singular vectors are returned as the merging result.

        Notes:
        1. If ``U_i @ S_i`` is of size ``d`` by ``N_i``, then ``B`` has size ``d`` by ``N``, where
           ``N = N_1 + N_2 + ... + N_n.``
        2. If ``U @ S @ V.T = B`` is the SVD of ``B``, then it turns out that ``U = A @ U'``, where the columns
           of ``U'`` are the true principal components of the aggregated data, and ``A`` is some block unitary matrix.

        For the theoretical justification behind this procedure, see the paper
        "A Distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks".

        **NOTE**: This method assumes that the *columns* of ``U_i``'s are the local principal components. Thus, after
        performing SVD on the matrix ``B`` (defined above), the merging result is the **left** singular vectors.

        This is in contrast with the client-side implementation of PCA (contained in class ``PcaModule``), which
        assumes that the **rows** of the input data matrix are the data points. Hence, in ``PcaModule``, the **right**
        singular vectors of the SVD of each client's data matrix are the principal components. (In a nutshell, the
        input data matrices in these two cases are "transposes" of each other.)

        Args:
            client_singular_vectors (NDArrays): Local PCs.
            client_singular_values (NDArrays): Singular values corresponding to local PCs.

        Returns:
            (tuple[NDArray, NDArray]): Merged PCs and corresponding singular values.
        """
        x = [u @ np.diag(s) for u, s in zip(client_singular_vectors, client_singular_values)]
        svd_input = np.concatenate(x, axis=1)
        new_singular_vectors, new_singular_values, _ = np.linalg.svd(svd_input, full_matrices=True)
        return new_singular_vectors, new_singular_values

    def merge_subspaces_qr(
        self, client_singular_vectors: NDArrays, client_singular_values: NDArrays
    ) -> tuple[NDArray, NDArray]:
        """
        Produce the principal components (PCs) for all the data distributed across clients by merging the PCs
        belonging to each local dataset.

        Each clients sends a matrix whose columns are the local principal components to the server. The corresponding
        singular values are also shared.

        This implementation can be viewed as a more efficient approximation to  the SVD-based merging in that it does
        not require performing SVD on a large matrix.

        Directly performing SVD does not take into account the following two observations, suggesting there are more
        efficient algorithms for merging:

        1. Each client's singular vectors are already orthonormal.
        2. The right singular vectors do not need to be computed since only the left singular vectors are returned as
           the merging result.

        In contrast, the algorithm here performs a QR decomposition on the large data matrix, which is more efficient
        than SVD, and SVD is only performed on a much smaller matrix.

        Similarly to the SVD-based merging, it returns an approximation of the true principal components
        of the aggregated data up to the multiplication of some block unitary matrix.

        For the theoretical justification behind this approach, see the paper "Subspace Tracking for Latent
        Semantic Analysis".

        **NOTE**: Similar to ``merge_subspaces_svd``, this method assumes that the **columns** of ``U_i``'s are the
        local principal components.

        Args:
            client_singular_vectors (NDArrays): Local PCs.
            client_singular_values (NDArrays): Singular values corresponding to local PCs.

        Returns:
            (tuple[NDArray, NDArray]): Merged PCs and corresponding singular values.
        """
        assert len(client_singular_values) >= MINIMUM_PCA_ClIENTS
        if len(client_singular_values) == MINIMUM_PCA_ClIENTS:
            u1, s1 = client_singular_vectors[0], np.diag(client_singular_values[0])
            u2, s2 = client_singular_vectors[1], np.diag(client_singular_values[1])
            return self.merge_two_subspaces_qr((u1, s1), (u2, s2))
        u, s = self.merge_subspaces_qr(client_singular_vectors[:-1], client_singular_values[:-1])
        u_last, s_last = client_singular_vectors[-1], client_singular_values[-1]
        return self.merge_two_subspaces_qr((u, np.diag(s)), (u_last, np.diag(s_last)))

    def merge_two_subspaces_qr(
        self, subspace1: tuple[NDArray, NDArray], subspace2: tuple[NDArray, NDArray]
    ) -> tuple[NDArray, NDArray]:
        u1, s1 = subspace1
        u2, s2 = subspace2

        z = u1.T @ u2
        q, r = np.linalg.qr(u2 - u1 @ z)

        d2 = s1.shape[1]
        d1 = r.shape[0]
        zeros = np.zeros(shape=(d1, d2))
        a = np.concatenate((s1, zeros), axis=0)
        b = np.concatenate(((z @ s2), (r @ s2)), axis=0)
        svd_input = np.concatenate((a, b), axis=1)

        u3, s_final, _ = np.linalg.svd(svd_input, full_matrices=False)

        u_final = (np.concatenate((u1, q), axis=1)) @ u3

        m, n = u1.shape[0], u1.shape[1] + u2.shape[1]
        rank = min(m, n)
        return u_final[:, :rank], s_final[:rank]
