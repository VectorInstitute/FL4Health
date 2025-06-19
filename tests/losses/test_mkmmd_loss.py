import pytest
import torch
from torch.distributions import MultivariateNormal

from fl4health.losses.mkmmd_loss import MkMmdLoss


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X = torch.Tensor(
    [
        [1, 1, 1],
        [3, 4, 4],
        [4, 2, 1],
        [2, 1, 4],
        [1, 2, 1],
        [3, 4, 4],
        [4, 3, 3],
        [3, 3, 2],
        [4, 4, 4],
        [4, 2, 1],
        [1, 1, 1],
    ]
).to(DEVICE)
Y = torch.Tensor(
    [
        [4, 3, 4],
        [1, 2, 2],
        [3, 4, 1],
        [1, 4, 2],
        [4, 2, 4],
        [4, 1, 2],
        [2, 2, 1],
        [2, 3, 4],
        [3, 2, 1],
        [4, 1, 4],
        [2, 2, 2],
    ]
).to(DEVICE)

mkmmd_loss_linear = MkMmdLoss(DEVICE, gammas=torch.Tensor([2, 1, 1 / 2]), perform_linear_approximation=True)
mkmmd_loss_linear.betas = torch.Tensor([1.5, 2.0, -1.0]).reshape(-1, 1).to(DEVICE)
quads = mkmmd_loss_linear.construct_quadruples(X, Y)
inner_products_linear = mkmmd_loss_linear.compute_euclidean_inner_products_linear(quads)
all_h_us_linear = mkmmd_loss_linear.compute_all_h_u_from_inner_products_linear(inner_products_linear)
hat_d_per_kernel_linear = mkmmd_loss_linear.compute_hat_d_per_kernel(all_h_us_linear)
h_u_delta_w_i = mkmmd_loss_linear.form_h_u_delta_w_i(all_h_us_linear)
q_k_linear = mkmmd_loss_linear.compute_hat_q_k_linear(all_h_us_linear)

mkmmd_loss = MkMmdLoss(DEVICE, gammas=torch.Tensor([2, 1, 1 / 2]))
mkmmd_loss.betas = torch.Tensor([1.5, 2.0, -1.0]).reshape(-1, 1).to(DEVICE)
inner_products_all = mkmmd_loss.compute_euclidean_inner_products(X, Y)
all_h_us_all = mkmmd_loss.compute_all_h_u_from_inner_products(inner_products_all)
hat_d_per_kernel_all = mkmmd_loss.compute_hat_d_per_kernel(all_h_us_all)
kernel_samples_minus_expectation = mkmmd_loss.form_kernel_samples_minus_expectation(all_h_us_all, hat_d_per_kernel_all)
q_k = mkmmd_loss.compute_hat_q_k(all_h_us_all, hat_d_per_kernel_all)


def test_normalize_features() -> None:
    normalized_x = mkmmd_loss.normalize(X)
    assert normalized_x.shape == (11, 3)
    norm_of_rows = torch.linalg.norm(normalized_x, dim=1)
    for index in range(11):
        assert pytest.approx(norm_of_rows[index].item(), abs=0.0001) == 1.0


def test_construct_quadruples() -> None:
    # Should have grouped the X, Y features into quadruples (x_{2i-1}, x_{2i}, y_{2i-1}, y_{2i})
    # since the input number of samples is odd (11), we truncate and drop the last sample
    assert quads.shape == (5, 4, 3)
    quads_target = torch.Tensor(
        [
            [[1, 1, 1], [3, 4, 4], [4, 3, 4], [1, 2, 2]],
            [[4, 2, 1], [2, 1, 4], [3, 4, 1], [1, 4, 2]],
            [[1, 2, 1], [3, 4, 4], [4, 2, 4], [4, 1, 2]],
            [[4, 3, 3], [3, 3, 2], [2, 2, 1], [2, 3, 4]],
            [[4, 4, 4], [4, 2, 1], [3, 2, 1], [4, 1, 4]],
        ]
    ).to(DEVICE)
    assert torch.all(quads.eq(quads_target))


def test_inner_products_calculation_linear() -> None:
    # inner products should be of shape n_samples // 2 x 4 (inner product component)
    assert inner_products_linear.shape == (5, 4)
    inner_product_quad_1_0 = 2 * 2 + 1 * 1 + (-3) * (-3)
    inner_product_quad_1_1 = 2 * 2 + 0 * 0 + (-1) * (-1)
    inner_product_quad_1_2 = 3 * 3 + (-2) * (-2) + (-1) * (-1)
    inner_product_quad_1_3 = (-1) * (-1) + (-3) * (-3) + (3) * (3)
    assert inner_products_linear[1][0] == inner_product_quad_1_0
    assert inner_products_linear[1][1] == inner_product_quad_1_1
    assert inner_products_linear[1][2] == inner_product_quad_1_2
    assert inner_products_linear[1][3] == inner_product_quad_1_3

    inner_product_quad_4_0 = 0 * 0 + 2 * 2 + 3 * 3
    inner_product_quad_4_1 = (-1) * (-1) + (1) * (1) + (-3) * (-3)
    inner_product_quad_4_2 = 0 * 0 + 3 * 3 + 0 * 0
    inner_product_quad_4_3 = 1 * 1 + 0 * 0 + 0 * 0
    assert inner_products_linear[4][0] == inner_product_quad_4_0
    assert inner_products_linear[4][1] == inner_product_quad_4_1
    assert inner_products_linear[4][2] == inner_product_quad_4_2
    assert inner_products_linear[4][3] == inner_product_quad_4_3


def test_inner_product_calculations_all() -> None:
    assert inner_products_all.shape == (4, 11, 11)

    inner_product_x3_x3 = 0.0
    inner_product_x1_x2 = (3.0 - 4.0) * (3.0 - 4.0) + (4.0 - 2.0) * (4.0 - 2.0) + (4.0 - 1.0) * (4.0 - 1.0)
    # Index into XX'
    assert pytest.approx(inner_products_all[0, 3, 3].cpu(), abs=0.0001) == inner_product_x3_x3
    assert pytest.approx(inner_products_all[0, 1, 2].cpu(), abs=0.0001) == inner_product_x1_x2
    # Should be symmetric
    assert pytest.approx(inner_products_all[0, 2, 1].cpu(), abs=0.0001) == inner_product_x1_x2

    inner_product_x1_y1 = (3.0 - 1.0) * (3.0 - 1.0) + (4.0 - 2.0) * (4.0 - 2.0) + (4.0 - 2.0) * (4.0 - 2.0)
    inner_product_x1_y3 = (3.0 - 1.0) * (3.0 - 1.0) + (4.0 - 4.0) * (4.0 - 4.0) + (4.0 - 2.0) * (4.0 - 2.0)
    # Index into XY'
    assert pytest.approx(inner_products_all[2, 1, 1].cpu(), abs=0.0001) == inner_product_x1_y1
    assert pytest.approx(inner_products_all[2, 1, 3].cpu(), abs=0.0001) == inner_product_x1_y3
    # Should be transpose in X'Y
    assert pytest.approx(inner_products_all[3, 1, 1].cpu(), abs=0.0001) == inner_product_x1_y1
    assert pytest.approx(inner_products_all[3, 3, 1].cpu(), abs=0.0001) == inner_product_x1_y3

    inner_product_y2_y2 = 0.0
    inner_product_y5_y8 = (4.0 - 3.0) * (4.0 - 3.0) + (1.0 - 2.0) * (1.0 - 2.0) + (2.0 - 1.0) * (2.0 - 1.0)
    # Index into YY'
    assert pytest.approx(inner_products_all[1, 2, 2].cpu(), abs=0.0001) == inner_product_y2_y2
    assert pytest.approx(inner_products_all[1, 5, 8].cpu(), abs=0.0001) == inner_product_y5_y8
    # Should be symmetric
    assert pytest.approx(inner_products_all[1, 8, 5].cpu(), abs=0.0001) == inner_product_y5_y8

    inner_product_y1_x1 = (1.0 - 3.0) * (1.0 - 3.0) + (2.0 - 4.0) * (2.0 - 4.0) + (2.0 - 4.0) * (2.0 - 4.0)
    inner_product_y3_x3 = (1.0 - 2.0) * (1.0 - 2.0) + (4.0 - 1.0) * (4.0 - 1.0) + (2.0 - 4.0) * (2.0 - 4.0)
    # Index into X'Y
    assert pytest.approx(inner_products_all[3, 1, 1].cpu(), abs=0.0001) == inner_product_y1_x1
    assert pytest.approx(inner_products_all[3, 3, 3].cpu(), abs=0.0001) == inner_product_y3_x3
    # Should be transpose in XY', but we're looking at the diagonal
    assert pytest.approx(inner_products_all[2, 1, 1].cpu(), abs=0.0001) == inner_product_y1_x1
    assert pytest.approx(inner_products_all[2, 3, 3].cpu(), abs=0.0001) == inner_product_y3_x3


def test_compute_h_u_from_inner_products_linear() -> None:
    gamma_1, gamma_2 = torch.Tensor([2]).to(DEVICE), torch.Tensor([1]).to(DEVICE)
    h_u_gamma_1 = mkmmd_loss_linear.compute_h_u_from_inner_products_linear(inner_products_linear, gamma_1)
    h_u_gamma_2 = mkmmd_loss_linear.compute_h_u_from_inner_products_linear(inner_products_linear, gamma_2)

    # For each gamma there should be an h_u value per v_i quadruples
    assert h_u_gamma_1.shape == (1, 5)
    assert h_u_gamma_2.shape == (1, 5)

    # The fourth entry should coincide with computing h_u(v_4), so we compute h_u using the 4th inner product entry
    h_u_components_gamma_1_3_target = torch.exp((-1 * torch.Tensor([2, 10, 5, 3])).to(DEVICE) / gamma_1)
    # For the other gamma, the fourth entry should coincide with computing h_u(v_4), so we compute h_u using the
    # 4th inner product entry
    h_u_components_gamma_2_3_target = torch.exp((-1 * torch.Tensor([2, 10, 5, 3])).to(DEVICE) / gamma_2)
    h_u_components_gamma_2_0_target = torch.exp((-1 * torch.Tensor([22, 14, 2, 2])).to(DEVICE) / gamma_2)

    h_u_gamma_1_3_target = (
        h_u_components_gamma_1_3_target[0]
        + h_u_components_gamma_1_3_target[1]
        - h_u_components_gamma_1_3_target[2]
        - h_u_components_gamma_1_3_target[3]
    )
    h_u_gamma_2_3_target = (
        h_u_components_gamma_2_3_target[0]
        + h_u_components_gamma_2_3_target[1]
        - h_u_components_gamma_2_3_target[2]
        - h_u_components_gamma_2_3_target[3]
    )

    h_u_gamma_2_0_target = (
        h_u_components_gamma_2_0_target[0]
        + h_u_components_gamma_2_0_target[1]
        - h_u_components_gamma_2_0_target[2]
        - h_u_components_gamma_2_0_target[3]
    )

    assert torch.all(h_u_gamma_1_3_target.eq(h_u_gamma_1[0, 3]))
    assert torch.all(h_u_gamma_2_3_target.eq(h_u_gamma_2[0, 3]))
    assert torch.all(h_u_gamma_2_0_target.eq(h_u_gamma_2[0, 0]))


def test_compute_h_u_from_inner_products() -> None:
    gamma_1, gamma_2 = torch.Tensor([2]).to(DEVICE), torch.Tensor([1]).to(DEVICE)
    h_u_gamma_1 = mkmmd_loss.compute_h_u_from_inner_products(inner_products_all, gamma_1)
    h_u_gamma_2 = mkmmd_loss.compute_h_u_from_inner_products(inner_products_all, gamma_2)

    # For each gamma there should be an h_u value for all possible pairs (x_j, y_j) x (x_k, y_k)
    # or n_samples x n_samples
    assert h_u_gamma_1.shape == (1, 11, 11)
    assert h_u_gamma_2.shape == (1, 11, 11)

    # Looking at the j=3, k=5 entry for gamma_1, this should correspond to the pairs
    # (x_3, x_5), (y_3, y_5), (x_3, y_5), (y_3, x_5)
    h_u_components_gamma_1_3_5_target = torch.exp((-1 * torch.Tensor([10, 18, 8, 8])).to(DEVICE) / gamma_1)
    # Looking at the j=10, k=4 entry for gamma_1, this should correspond to the pairs
    # (x_10, x_4), (y_10, y_4), (x_10, y_4), (y_10, x_4)
    h_u_components_gamma_1_10_4_target = torch.exp((-1 * torch.Tensor([1, 8, 19, 2])).to(DEVICE) / gamma_1)

    # Looking at the j=0, k=1 entry for gamma_2, this should correspond to the pairs
    # (x_0, x_1), (y_0, y_1), (x_0, y_1), (y_0, x_1)
    h_u_components_gamma_2_0_1_target = torch.exp((-1 * torch.Tensor([20, 14, 2, 2])).to(DEVICE) / gamma_2)
    # Looking at the j=9, k=4 entry for gamma_2, this should correspond to the pairs
    # (x_9, x_4), (y_9, y_4), (x_9, y_4), (y_9, x_4)
    h_u_components_gamma_2_9_4_target = torch.exp((-1 * torch.Tensor([9, 1, 9, 19])).to(DEVICE) / gamma_2)

    h_u_gamma_1_3_5_target = (
        h_u_components_gamma_1_3_5_target[0]
        + h_u_components_gamma_1_3_5_target[1]
        - h_u_components_gamma_1_3_5_target[2]
        - h_u_components_gamma_1_3_5_target[3]
    )

    h_u_gamma_1_10_4_target = (
        h_u_components_gamma_1_10_4_target[0]
        + h_u_components_gamma_1_10_4_target[1]
        - h_u_components_gamma_1_10_4_target[2]
        - h_u_components_gamma_1_10_4_target[3]
    )

    h_u_gamma_2_0_1_target = (
        h_u_components_gamma_2_0_1_target[0]
        + h_u_components_gamma_2_0_1_target[1]
        - h_u_components_gamma_2_0_1_target[2]
        - h_u_components_gamma_2_0_1_target[3]
    )

    h_u_gamma_2_9_4_target = (
        h_u_components_gamma_2_9_4_target[0]
        + h_u_components_gamma_2_9_4_target[1]
        - h_u_components_gamma_2_9_4_target[2]
        - h_u_components_gamma_2_9_4_target[3]
    )

    assert torch.all(h_u_gamma_1_3_5_target.eq(h_u_gamma_1[0, 3, 5]))
    assert torch.all(h_u_gamma_1_10_4_target.eq(h_u_gamma_1[0, 10, 4]))
    assert torch.all(h_u_gamma_2_0_1_target.eq(h_u_gamma_2[0, 0, 1]))
    assert torch.all(h_u_gamma_2_9_4_target.eq(h_u_gamma_2[0, 9, 4]))


def test_compute_all_h_u_from_inner_products_linear() -> None:
    # shape should be num_kernels x number v_is
    assert all_h_us_linear.shape == (3, 5)
    h_u_gamma_1 = mkmmd_loss_linear.compute_h_u_from_inner_products_linear(
        inner_products_linear, torch.Tensor([2]).to(DEVICE)
    )
    h_u_gamma_2 = mkmmd_loss_linear.compute_h_u_from_inner_products_linear(
        inner_products_linear, torch.Tensor([1]).to(DEVICE)
    )
    h_u_gamma_3 = mkmmd_loss_linear.compute_h_u_from_inner_products_linear(
        inner_products_linear, torch.Tensor([0.5]).to(DEVICE)
    )
    assert torch.all(all_h_us_linear.eq(torch.cat([h_u_gamma_1, h_u_gamma_2, h_u_gamma_3], 0)))

    # test run all the way through from X, Y
    all_h_us_full = mkmmd_loss_linear.compute_all_h_u_linear(X, Y)
    assert torch.all(all_h_us_full.eq(all_h_us_linear))


def test_compute_all_h_u_from_inner_products() -> None:
    # shape should be num_kernels x num_samples x num_samples
    assert all_h_us_all.shape == (3, 11, 11)
    h_u_gamma_1 = mkmmd_loss.compute_h_u_from_inner_products(inner_products_all, torch.Tensor([2]).to(DEVICE))
    h_u_gamma_2 = mkmmd_loss.compute_h_u_from_inner_products(inner_products_all, torch.Tensor([1]).to(DEVICE))
    h_u_gamma_3 = mkmmd_loss.compute_h_u_from_inner_products(inner_products_all, torch.Tensor([0.5]).to(DEVICE))
    assert torch.all(all_h_us_all.eq(torch.cat([h_u_gamma_1, h_u_gamma_2, h_u_gamma_3])))

    # test run all the way through from X, Y
    all_h_us_full = mkmmd_loss.compute_all_h_u_all_samples(X, Y)
    assert torch.all(all_h_us_full.eq(all_h_us_all))


def test_compute_hat_d_per_kernel_linear() -> None:
    # shape should be num_kernels x 1
    assert hat_d_per_kernel_linear.shape == (3, 1)
    assert (
        pytest.approx(hat_d_per_kernel_linear[0, 0].item(), abs=0.00001)
        == ((1 / 5) * torch.sum(all_h_us_linear[0, :])).item()
    )
    assert (
        pytest.approx(hat_d_per_kernel_linear[1, 0].item(), abs=0.00001)
        == ((1 / 5) * torch.sum(all_h_us_linear[1, :])).item()
    )
    assert (
        pytest.approx(hat_d_per_kernel_linear[2, 0].item(), abs=0.00001)
        == ((1 / 5) * torch.sum(all_h_us_linear[2, :])).item()
    )


def test_compute_mkmmd_linear() -> None:
    betas = torch.Tensor([1.5, 2.0, -1.0]).reshape(-1, 1).to(DEVICE)
    mkmmd_target = (
        hat_d_per_kernel_linear[0, 0] * 1.5
        + hat_d_per_kernel_linear[1, 0] * 2.0
        + hat_d_per_kernel_linear[2, 0] * (-1.0)
    )
    assert pytest.approx(mkmmd_loss_linear.compute_mkmmd(X, Y, betas).cpu(), abs=0.0001) == mkmmd_target.cpu()


def test_compute_hat_d_per_kernel() -> None:
    # shape should be num_kernels x 1
    assert hat_d_per_kernel_all.shape == (3, 1)
    assert (
        pytest.approx(hat_d_per_kernel_all[0, 0].item(), abs=0.00001)
        == ((1 / 121) * torch.sum(all_h_us_all[0, :, :])).item()
    )
    assert (
        pytest.approx(hat_d_per_kernel_all[1, 0].item(), abs=0.00001)
        == ((1 / 121) * torch.sum(all_h_us_all[1, :, :])).item()
    )
    assert (
        pytest.approx(hat_d_per_kernel_all[2, 0].item(), abs=0.00001)
        == ((1 / 121) * torch.sum(all_h_us_all[2, :, :])).item()
    )


def test_compute_mkmmd() -> None:
    betas = torch.Tensor([1.5, 2.0, -1.0]).reshape(-1, 1).to(DEVICE)
    mkmmd_target = (
        hat_d_per_kernel_all[0, 0] * 1.5 + hat_d_per_kernel_all[1, 0] * 2.0 + hat_d_per_kernel_all[2, 0] * (-1.0)
    )
    assert pytest.approx(mkmmd_loss.compute_mkmmd(X, Y, betas).cpu(), abs=0.0001) == mkmmd_target.cpu()


def test_create_h_u_delta_w_i() -> None:
    # shape should be num num_kernels x v_is // 2 (noting that we drop the last v_i if uneven number)
    assert h_u_delta_w_i.shape == (3, 2)
    assert h_u_delta_w_i[0, 0] == all_h_us_linear[0, 0] - all_h_us_linear[0, 1]
    assert h_u_delta_w_i[0, 1] == all_h_us_linear[0, 2] - all_h_us_linear[0, 3]
    assert h_u_delta_w_i[1, 0] == all_h_us_linear[1, 0] - all_h_us_linear[1, 1]
    assert h_u_delta_w_i[1, 1] == all_h_us_linear[1, 2] - all_h_us_linear[1, 3]
    assert h_u_delta_w_i[2, 0] == all_h_us_linear[2, 0] - all_h_us_linear[2, 1]
    assert h_u_delta_w_i[2, 1] == all_h_us_linear[2, 2] - all_h_us_linear[2, 3]


def test_compute_hat_q_k_linear() -> None:
    # shape should be number of kernels x number of kernels
    assert q_k_linear.shape == (3, 3)
    assert q_k_linear[0, 0] == (0.5) * (
        h_u_delta_w_i[0, 0] * h_u_delta_w_i[0, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[0, 1]
    )
    assert q_k_linear[0, 1] == (0.5) * (
        h_u_delta_w_i[0, 0] * h_u_delta_w_i[1, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[1, 1]
    )
    assert q_k_linear[1, 0] == (0.5) * (
        h_u_delta_w_i[0, 0] * h_u_delta_w_i[1, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[1, 1]
    )
    assert q_k_linear[0, 2] == (0.5) * (
        h_u_delta_w_i[0, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[2, 1]
    )
    assert q_k_linear[2, 0] == (0.5) * (
        h_u_delta_w_i[0, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[2, 1]
    )
    assert q_k_linear[1, 1] == (0.5) * (
        h_u_delta_w_i[1, 0] * h_u_delta_w_i[1, 0] + h_u_delta_w_i[1, 1] * h_u_delta_w_i[1, 1]
    )
    assert q_k_linear[1, 2] == (0.5) * (
        h_u_delta_w_i[1, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[1, 1] * h_u_delta_w_i[2, 1]
    )
    assert q_k_linear[2, 1] == (0.5) * (
        h_u_delta_w_i[1, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[1, 1] * h_u_delta_w_i[2, 1]
    )
    assert q_k_linear[2, 2] == (0.5) * (
        h_u_delta_w_i[2, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[2, 1] * h_u_delta_w_i[2, 1]
    )


def test_form_kernel_samples_minus_expectation() -> None:
    assert kernel_samples_minus_expectation.shape == (3, 11, 11)

    kernel_samples_minus_expectation_0_3_5 = all_h_us_all[0, 3, 5] - hat_d_per_kernel_all[0, 0]
    kernel_samples_minus_expectation_1_3_5 = all_h_us_all[1, 3, 5] - hat_d_per_kernel_all[1, 0]
    kernel_samples_minus_expectation_2_0_1 = all_h_us_all[2, 0, 1] - hat_d_per_kernel_all[2, 0]

    assert (
        pytest.approx(kernel_samples_minus_expectation_0_3_5.cpu(), abs=0.00001)
        == kernel_samples_minus_expectation[0, 3, 5].cpu()
    )
    assert (
        pytest.approx(kernel_samples_minus_expectation_1_3_5.cpu(), abs=0.00001)
        == kernel_samples_minus_expectation[1, 3, 5].cpu()
    )
    assert (
        pytest.approx(kernel_samples_minus_expectation_2_0_1.cpu(), abs=0.00001)
        == kernel_samples_minus_expectation[2, 0, 1].cpu()
    )

    kernel_samples_minus_expectation_0_7_2 = all_h_us_all[0, 7, 2] - hat_d_per_kernel_all[0, 0]
    kernel_samples_minus_expectation_1_9_1 = all_h_us_all[1, 9, 1] - hat_d_per_kernel_all[1, 0]
    kernel_samples_minus_expectation_2_1_1 = all_h_us_all[2, 1, 1] - hat_d_per_kernel_all[2, 0]

    assert (
        pytest.approx(kernel_samples_minus_expectation_0_7_2.cpu(), abs=0.00001)
        == kernel_samples_minus_expectation[0, 7, 2].cpu()
    )
    assert (
        pytest.approx(kernel_samples_minus_expectation_1_9_1.cpu(), abs=0.00001)
        == kernel_samples_minus_expectation[1, 9, 1].cpu()
    )
    assert (
        pytest.approx(kernel_samples_minus_expectation_2_1_1.cpu(), abs=0.00001)
        == kernel_samples_minus_expectation[2, 1, 1].cpu()
    )


def unrolled_summation(kernel_1_index: int, kernel_2_index: int) -> torch.Tensor:
    sum = torch.Tensor([0.0]).to(DEVICE)
    for i in range(11):
        for j in range(11):
            sum += (
                kernel_samples_minus_expectation[kernel_1_index][i][j]
                * kernel_samples_minus_expectation[kernel_2_index][i][j]
            )
    return sum


def test_compute_q_k() -> None:
    assert q_k.shape == (3, 3)

    q_k_0_0 = (1 / (121 - 1)) * unrolled_summation(0, 0)
    q_k_1_0 = (1 / (121 - 1)) * unrolled_summation(1, 0)
    q_k_0_1 = (1 / (121 - 1)) * unrolled_summation(0, 1)
    q_k_1_2 = (1 / (121 - 1)) * unrolled_summation(1, 2)
    q_k_2_1 = (1 / (121 - 1)) * unrolled_summation(2, 1)
    # Should be symmetric
    assert pytest.approx(q_k_1_0.cpu(), abs=0.00001) == q_k_0_1.cpu()
    assert pytest.approx(q_k_1_2.cpu(), abs=0.00001) == q_k_2_1.cpu()

    assert pytest.approx(q_k[0, 0].cpu(), abs=0.00001) == q_k_0_0[0].cpu()
    assert pytest.approx(q_k[1, 0].cpu(), abs=0.00001) == q_k_1_0[0].cpu()
    assert pytest.approx(q_k[0, 1].cpu(), abs=0.00001) == q_k_0_1[0].cpu()
    assert pytest.approx(q_k[1, 2].cpu(), abs=0.00001) == q_k_1_2[0].cpu()
    assert pytest.approx(q_k[2, 1].cpu(), abs=0.00001) == q_k_2_1[0].cpu()


def test_beta_with_largest_hat_d() -> None:
    test_hat_d = torch.Tensor([[-1.2, 1.3, 2.1, 0.45, -6.5, -0.33]]).to(DEVICE).t()
    test_hat_q_k = torch.Tensor(
        [
            [-1, 1.5, 2.5, 3, -5.5, 0],
            [3, 4, 2.2, 5.6, 3.1, -1.2],
            [-1.5, 1.2, 3.4, 2.1, 0.5, -0.1],
            [1.5, 2.1, 3.2, 4.5, 2.1, 1.2],
            [4, 3.2, 3.5, -2, 1.2, 3.4],
            [1.2, 3.4, 2.1, 1.2, 3.4, 2.1],
        ]
    ).to(DEVICE)
    assert test_hat_d.shape == (6, 1)
    one_hot_beta_max = mkmmd_loss.beta_with_extreme_kernel_base_values(
        test_hat_d, test_hat_q_k, minimize_type_two_error=True
    )
    beta_target_max = torch.Tensor([[1, 0, 0, 0, 0, 0]]).to(DEVICE).t()
    assert one_hot_beta_max.shape == (6, 1)
    assert torch.all(one_hot_beta_max.eq(beta_target_max))

    one_hot_beta_min = mkmmd_loss.beta_with_extreme_kernel_base_values(
        test_hat_d, test_hat_q_k, minimize_type_two_error=False
    )
    beta_target_min = torch.Tensor([[0, 0, 0, 0, 1, 0]]).to(DEVICE).t()
    assert one_hot_beta_min.shape == (6, 1)
    assert torch.all(one_hot_beta_min.eq(beta_target_min))


def test_optimize_betas_for_x_y() -> None:
    # The simple test cases above result in a set of hat_ds that are all negative for the linear estimate. In this
    # case, we perform the selection of a single kernel as recommended in Gretton
    degenerate_betas = mkmmd_loss_linear.optimize_betas(X, Y, lambda_m=0.0001)
    beta_target = torch.Tensor([[1, 0, 0]]).to(DEVICE).t()
    assert torch.all(degenerate_betas.eq(beta_target))

    non_degenerate_beta = mkmmd_loss.optimize_betas(X, Y, lambda_m=0.0001)
    beta_target = torch.Tensor([[0, 0, 1.0]]).to(DEVICE).t()
    assert torch.allclose(non_degenerate_beta, beta_target, rtol=0, atol=1e-7)


def test_gamma_defaults() -> None:
    default_mkmmd = MkMmdLoss(DEVICE)
    neg_gamma_powers = [-3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0]
    pos_gamma_powers = [0.25, 0.5, 0.75, 1]
    gamma_powers = neg_gamma_powers + pos_gamma_powers
    default_gammas = torch.Tensor([2**power for power in gamma_powers]).to(DEVICE)
    assert torch.all(default_mkmmd.gammas.eq(default_gammas))
    assert torch.all(torch.where(default_mkmmd.betas >= 0.0, True, False))
    assert pytest.approx(torch.sum(default_mkmmd.betas).item(), abs=0.0001) == 1.0


def test_get_best_vertex_for_objective_function() -> None:
    lambda_m = 0.0001
    regularized_q_k = 2 * q_k_linear + lambda_m * torch.eye(3).to(DEVICE)
    best_vertex = mkmmd_loss_linear.get_best_vertex_for_objective_function(hat_d_per_kernel_linear, regularized_q_k)
    assert best_vertex.shape == (3, 1)

    assert pytest.approx(best_vertex[0, 0].item(), abs=0.0001) == -4.1689
    assert pytest.approx(best_vertex[1, 0].item(), abs=0.0001) == 0.0
    assert pytest.approx(best_vertex[2, 0].item(), abs=0.0001) == 0.0
    assert pytest.approx(torch.mm(hat_d_per_kernel_linear.t(), best_vertex).cpu(), abs=0.0001) == 1.0


def test_optimizer_betas_in_non_degenerate_case_linear() -> None:
    lambda_m = 0.0001
    torch.manual_seed(42)
    default_mkmmd = MkMmdLoss(DEVICE, perform_linear_approximation=True)

    # First sample is from a zero mean gaussian with unit covariance (dimension 5)
    p = MultivariateNormal(torch.zeros(5), torch.eye(5))
    x_local = p.sample(
        torch.Size(
            [
                100,
            ]
        )
    )

    # Second sample is a mixture of two gaussian with zero mean except the first has mean 1.0 in the first coordinate
    # and the second has mean 1.0 in the second coordinate
    q_1 = MultivariateNormal(torch.tensor([1.0, 0, 0, 0, 0]), torch.eye(5))
    q_2 = MultivariateNormal(torch.tensor([0, 1.0, 0, 0, 0]), torch.eye(5))
    y_local = (
        q_1.sample(
            torch.Size(
                [
                    100,
                ]
            )
        )
        + q_2.sample(
            torch.Size(
                [
                    100,
                ]
            )
        )
    ) / 2.0
    x_local = x_local.to(DEVICE)
    y_local = y_local.to(DEVICE)
    all_h_u_per_vi_local = default_mkmmd.compute_all_h_u_linear(x_local, y_local)
    hat_d_per_kernel_local = default_mkmmd.compute_hat_d_per_kernel(all_h_u_per_vi_local)
    mkmmd_before_opt = default_mkmmd(x_local, y_local)
    target_mkmmd = torch.mm(default_mkmmd.betas.t(), hat_d_per_kernel_local)[0][0]
    assert pytest.approx(mkmmd_before_opt.item(), abs=0.000001) == target_mkmmd.cpu()

    hat_q_k = default_mkmmd.compute_hat_q_k_linear(all_h_u_per_vi_local)
    regularized_hat_q_k = 2 * hat_q_k + lambda_m * torch.eye(19).to(DEVICE)
    raw_betas = default_mkmmd.form_and_solve_qp(hat_d_per_kernel_local, regularized_hat_q_k)
    raw_betas = torch.clamp(raw_betas, min=0)
    assert pytest.approx(torch.mm(raw_betas.t(), hat_d_per_kernel_local).cpu(), abs=0.0001) == 1.0000

    betas_local = default_mkmmd.optimize_betas(x_local, y_local, lambda_m)
    assert torch.all(betas_local.eq((1 / torch.sum(raw_betas)) * raw_betas))

    default_mkmmd = MkMmdLoss(DEVICE, minimize_type_two_error=False, perform_linear_approximation=True)
    betas_local = default_mkmmd.optimize_betas(x_local, y_local, lambda_m)
    one_hot_betas = torch.zeros_like(betas_local)

    one_hot_betas[0, 0] = 1
    assert torch.all(betas_local.eq(one_hot_betas))


def test_optimizer_betas_in_non_degenerate_case() -> None:
    lambda_m = 0.0001
    torch.manual_seed(42)
    default_mkmmd = MkMmdLoss(DEVICE)

    # First sample is from a zero mean gaussian with unit covariance (dimension 5)
    p = MultivariateNormal(torch.zeros(5), torch.eye(5))
    x_local = p.sample(
        torch.Size(
            [
                100,
            ]
        )
    )

    # Second sample is a mixture of two gaussian with zero mean except the first has mean 1.0 in the first coordinate
    # and the second has mean 1.0 in the second coordinate
    q_1 = MultivariateNormal(torch.tensor([1.0, 0, 0, 0, 0]), torch.eye(5))
    q_2 = MultivariateNormal(torch.tensor([0, 1.0, 0, 0, 0]), torch.eye(5))
    y_local = (
        q_1.sample(
            torch.Size(
                [
                    100,
                ]
            )
        )
        + q_2.sample(
            torch.Size(
                [
                    100,
                ]
            )
        )
    ) / 2.0
    x_local = x_local.to(DEVICE)
    y_local = y_local.to(DEVICE)

    all_h_u_per_sample = default_mkmmd.compute_all_h_u_all_samples(x_local, y_local)
    hat_d_per_kernel_local = default_mkmmd.compute_hat_d_per_kernel(all_h_u_per_sample)
    mkmmd_before_opt = default_mkmmd(x_local, y_local)
    target_mkmmd = torch.mm(default_mkmmd.betas.t(), hat_d_per_kernel_local)[0][0]
    assert pytest.approx(mkmmd_before_opt.item(), abs=0.000001) == target_mkmmd.cpu()

    hat_q_k = default_mkmmd.compute_hat_q_k(all_h_u_per_sample, hat_d_per_kernel_local)
    regularized_hat_q_k = 2 * hat_q_k + lambda_m * torch.eye(19).to(DEVICE)
    raw_betas = default_mkmmd.form_and_solve_qp(hat_d_per_kernel_local, regularized_hat_q_k)
    raw_betas = torch.clamp(raw_betas, min=0)
    assert pytest.approx(torch.mm(raw_betas.t(), hat_d_per_kernel_local).cpu(), abs=0.0001) == 1.0000

    betas_local = default_mkmmd.optimize_betas(x_local, y_local, lambda_m)
    assert torch.all(betas_local.eq((1 / torch.sum(raw_betas)) * raw_betas))

    default_mkmmd.betas = betas_local
    mkmmd_after_opt = default_mkmmd(x_local, y_local)
    assert mkmmd_after_opt.item() > mkmmd_before_opt.item()

    default_mkmmd = MkMmdLoss(DEVICE, minimize_type_two_error=False)
    betas_local = default_mkmmd.optimize_betas(x_local, y_local, lambda_m)
    one_hot_betas = torch.zeros_like(betas_local)

    one_hot_betas[1, 0] = 1
    assert torch.allclose(one_hot_betas, betas_local, rtol=0.0, atol=1e-6)
