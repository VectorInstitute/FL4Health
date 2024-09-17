import pytest
import torch
import torch.nn.functional as F

from fl4health.utils.data_generation import SyntheticIidFedProxDataset, SyntheticNonIidFedProxDataset


def test_covariance_matrix_construction() -> None:
    data_generator = SyntheticNonIidFedProxDataset(5, 1.0, 2.0, samples_per_client=20)
    covariance_matrix = data_generator.construct_covariance_matrix()

    # Correct size
    assert covariance_matrix.shape == (60, 60)

    # Assert off-diagonal elements are zero
    assert torch.allclose(covariance_matrix.sum(), covariance_matrix.diag().sum(), rtol=0.0, atol=0.00001)
    # Assert diagonal elements are correct
    diagonal = covariance_matrix.diag()
    assert diagonal[0] == 1.0
    assert pytest.approx(diagonal[1], abs=1e-5) == 0.435275
    assert pytest.approx(diagonal[29], abs=1e-5) == 0.016883
    assert pytest.approx(diagonal[35], abs=1e-5) == 0.0135655
    assert pytest.approx(diagonal[59], abs=1e-5) == 0.0073488


def test_get_input_output_tensors() -> None:
    torch.manual_seed(100)

    data_generator = SyntheticNonIidFedProxDataset(5, 1.0, 2.0, samples_per_client=5, input_dim=2, output_dim=3)
    sigma = data_generator.construct_covariance_matrix()
    X, Y = data_generator.get_input_output_tensors(mu=1.0, v=torch.ones((2)), sigma=sigma)

    # correct shapes
    assert X.shape == (5, 2)
    assert Y.shape == (5, 3)

    X_target = torch.Tensor(
        [[1.3607, 0.8114], [0.6062, 1.1603], [-0.3833, -0.5263], [0.6828, 0.4286], [2.7482, 0.8180]]
    )
    # Correct inputs
    assert torch.allclose(X, X_target, rtol=0.0, atol=1e-4)

    # Correct output calculations
    W = torch.Tensor([[0.0245, 1.4790], [-1.3652, 0.1953], [1.6587, 0.7414]])
    b = torch.Tensor([[0.7490], [1.4770], [0.4117]])
    Y_target_1 = F.one_hot(torch.argmax(F.softmax(torch.matmul(W, X[1, :]) + b.T, dim=1), dim=1), 3)
    Y_target_4 = F.one_hot(torch.argmax(F.softmax(torch.matmul(W, X[4, :]) + b.T, dim=1), dim=1), 3)
    assert torch.allclose(Y_target_1, Y[1, :], rtol=0.0, atol=1e-4)
    assert torch.allclose(Y_target_4, Y[4, :], rtol=0.0, atol=1e-4)

    # Unset seed
    torch.seed()


def test_generate_client_tensors() -> None:
    torch.manual_seed(100)

    data_generator = SyntheticNonIidFedProxDataset(5, 0.0, 0.0, temperature=2.0, samples_per_client=5000)
    client_tensors = data_generator.generate_client_tensors()

    assert len(client_tensors) == 5
    for input_tensors, output_tensors in client_tensors:
        assert input_tensors.shape == (5000, 60)
        assert output_tensors.shape == (5000, 10)

    client_0_label_counts = torch.sum(client_tensors[0][1], dim=0)
    client_2_label_counts = torch.sum(client_tensors[2][1], dim=0)

    assert torch.equal(client_0_label_counts, torch.Tensor([2, 7, 4939, 2, 29, 16, 1, 4, 0, 0]))
    assert torch.equal(client_2_label_counts, torch.Tensor([282, 0, 0, 5, 0, 4, 313, 0, 0, 4396]))

    data_generator = SyntheticNonIidFedProxDataset(5, 0.5, 0.5, temperature=2.0, samples_per_client=5000)
    client_tensors = data_generator.generate_client_tensors()

    assert len(client_tensors) == 5
    for input_tensors, output_tensors in client_tensors:
        assert input_tensors.shape == (5000, 60)
        assert output_tensors.shape == (5000, 10)

    client_1_label_counts = torch.sum(client_tensors[1][1], dim=0)
    client_4_label_counts = torch.sum(client_tensors[4][1], dim=0)

    assert torch.equal(client_1_label_counts, torch.Tensor([1, 5, 680, 103, 4, 3560, 559, 88, 0, 0]))
    assert torch.equal(client_4_label_counts, torch.Tensor([79, 228, 0, 1, 26, 6, 3344, 720, 1, 595]))

    torch.seed()


def test_dataset_generation() -> None:
    data_generator = SyntheticNonIidFedProxDataset(5, 0.5, 0.5, temperature=2.0, samples_per_client=50)
    datasets = data_generator.generate()
    assert len(datasets) == 5

    assert datasets[0].data.shape == (50, 60)

    targets = datasets[4].targets
    assert targets is not None
    assert targets.shape == (50, 10)


def test_iid_dataset_generation() -> None:
    data_generator = SyntheticIidFedProxDataset(5, temperature=2.0, samples_per_client=10)

    datasets = data_generator.generate()
    assert len(datasets) == 5

    assert datasets[0].data.shape == (10, 60)

    targets = datasets[4].targets
    assert targets is not None
    assert targets.shape == (10, 10)

    sigma = data_generator.construct_covariance_matrix()
    torch.manual_seed(100)
    input_reference, output_reference = data_generator.get_input_output_tensors(sigma)
    for _ in range(3):
        # resetting the seed for generation consistency. Each set of tensors use the same affine map and the input
        # should be generated in the same way.
        torch.manual_seed(100)
        input, output = data_generator.get_input_output_tensors(sigma)
        assert torch.allclose(input_reference, input, rtol=0.0, atol=1e-7)
        assert torch.allclose(output_reference, output, rtol=0.0, atol=1e-7)

    torch.seed()
