import numpy as np

from fl4health.strategies.fedpca import FedPCA

data_dimension = 72
n1 = 300
n2 = 200
n3 = 256


def test_merging() -> None:
    np.random.seed(123)
    strategy = FedPCA()

    client1_data = np.random.rand(n1, data_dimension)
    client2_data = np.random.rand(n2, data_dimension)
    client3_data = np.random.rand(n3, data_dimension)

    X = np.concatenate((client1_data, client2_data, client3_data), axis=0)
    _, S, Vh = np.linalg.svd(X, full_matrices=True)

    _, S1, Vh1 = np.linalg.svd(client1_data, full_matrices=False)
    _, S2, Vh2 = np.linalg.svd(client2_data, full_matrices=False)
    _, S3, Vh3 = np.linalg.svd(client3_data, full_matrices=False)

    V1 = Vh1.T
    V2 = Vh2.T
    V3 = Vh3.T

    client_singular_vectors = [V1, V2, V3]
    client_singular_values = [S1, S2, S3]

    svd_merge_vectors, svd_merge_singular_values = strategy.merge_subspaces_svd(
        client_singular_vectors, client_singular_values
    )
    qr_merged_vectors, qr_merged_singular_values = strategy.merge_subspaces_qr(
        client_singular_vectors, client_singular_values
    )

    # svd_merge_vectors is only guaranteed to be the same as V up to the application
    # of a unitary transformation, thus we do not check whether it is close to V.
    assert np.allclose(svd_merge_singular_values, S)
    assert np.allclose(qr_merged_singular_values, S)
