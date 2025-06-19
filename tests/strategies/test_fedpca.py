import numpy as np

from fl4health.strategies.fedpca import FedPCA


data_dimension = 256
n1 = 300
n2 = 200
n3 = 256
n4 = 24


def test_merging() -> None:
    np.random.seed(123)
    strategy = FedPCA()

    client1_data = np.random.rand(n1, data_dimension)
    client2_data = np.random.rand(n2, data_dimension)
    client3_data = np.random.rand(n3, data_dimension)
    client4_data = np.random.rand(n4, data_dimension)

    x = np.concatenate((client1_data, client2_data, client3_data, client4_data), axis=0)
    _, s, vh = np.linalg.svd(x, full_matrices=True)

    _, s1, vh1 = np.linalg.svd(client1_data, full_matrices=False)
    _, s2, vh2 = np.linalg.svd(client2_data, full_matrices=False)
    _, s3, vh3 = np.linalg.svd(client3_data, full_matrices=False)
    _, s4, vh4 = np.linalg.svd(client4_data, full_matrices=False)

    v1 = vh1.T
    v2 = vh2.T
    v3 = vh3.T
    v4 = vh4.T

    client_singular_vectors = [v1, v2, v3, v4]
    client_singular_values = [s1, s2, s3, s4]

    svd_merged_vectors, svd_merged_singular_values = strategy.merge_subspaces_svd(
        client_singular_vectors, client_singular_values
    )
    qr_merged_vectors, qr_merged_singular_values = strategy.merge_subspaces_qr(
        client_singular_vectors, client_singular_values
    )

    # svd_merged_vectors is only guaranteed to be the same as V up to the application
    # of a blockwise unitary transformation, thus we do not check whether it is close to V.
    assert np.allclose(svd_merged_singular_values, s)
    assert np.allclose(qr_merged_singular_values, s)

    # Instead, we verify that the merging results are unitary matrices.
    assert np.allclose(svd_merged_vectors.T @ svd_merged_vectors, np.identity(svd_merged_vectors.shape[1]))
    assert np.allclose(qr_merged_vectors.T @ qr_merged_vectors, np.identity(qr_merged_vectors.shape[1]))
