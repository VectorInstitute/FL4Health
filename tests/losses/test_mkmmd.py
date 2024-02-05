import pytest
import torch
from torch.distributions import MultivariateNormal

from fl4health.losses.mkmmd_loss import MkMmdLoss

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
)
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
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mkmmd_loss = MkMmdLoss(DEVICE, gammas=torch.Tensor([2, 1, 1 / 2]), betas=torch.Tensor([1.5, 2.0, -1.0]).reshape(-1, 1))
quads = mkmmd_loss.construct_quadruples(X, Y)
inner_products = mkmmd_loss.compute_euclidean_inner_products(quads)
all_h_us = mkmmd_loss.compute_all_h_u_from_inner_products(inner_products)
hat_d_per_kernel = mkmmd_loss.compute_hat_d_per_kernel(all_h_us)
h_u_delta_w_i = mkmmd_loss.create_h_u_delta_w_i(all_h_us)
Q_k = mkmmd_loss.compute_hat_Q_k(all_h_us)


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
    )
    assert torch.all(quads.eq(quads_target))


def test_inner_products_calculation() -> None:
    # inner products should be of shape n_samples // 2 x 4 (inner product component)
    assert inner_products.shape == (5, 4)
    inner_product_quad_1_0 = 2 * 2 + 1 * 1 + (-3) * (-3)
    inner_product_quad_1_1 = 2 * 2 + 0 * 0 + (-1) * (-1)
    inner_product_quad_1_2 = 3 * 3 + (-2) * (-2) + (-1) * (-1)
    inner_product_quad_1_3 = (-1) * (-1) + (-3) * (-3) + (3) * (3)
    assert inner_products[1][0] == inner_product_quad_1_0
    assert inner_products[1][1] == inner_product_quad_1_1
    assert inner_products[1][2] == inner_product_quad_1_2
    assert inner_products[1][3] == inner_product_quad_1_3

    inner_product_quad_4_0 = 0 * 0 + 2 * 2 + 3 * 3
    inner_product_quad_4_1 = (-1) * (-1) + (1) * (1) + (-3) * (-3)
    inner_product_quad_4_2 = 0 * 0 + 3 * 3 + 0 * 0
    inner_product_quad_4_3 = 1 * 1 + 0 * 0 + 0 * 0
    assert inner_products[4][0] == inner_product_quad_4_0
    assert inner_products[4][1] == inner_product_quad_4_1
    assert inner_products[4][2] == inner_product_quad_4_2
    assert inner_products[4][3] == inner_product_quad_4_3


def test_compute_h_u_from_inner_products() -> None:
    gamma_1, gamma_2 = torch.Tensor([2]).to(DEVICE), torch.Tensor([1]).to(DEVICE)
    h_u_gamma_1 = mkmmd_loss.compute_h_u_from_inner_products(inner_products, gamma_1)
    h_u_gamma_2 = mkmmd_loss.compute_h_u_from_inner_products(inner_products, gamma_2)

    # For each gamma there should be an h_u value per v_i quadruples
    assert h_u_gamma_1.shape == (1, 5)
    assert h_u_gamma_2.shape == (1, 5)

    # The fourth entry should coincide with computing h_u(v_4), so we compute h_u using the 4th inner product entry
    h_u_components_gamma_1_3_target = torch.exp((-1 * torch.Tensor([2, 10, 5, 3])) / (2 * torch.pow(gamma_1, 2)))
    # For the other gamma, the fourth entry should coincide with computing h_u(v_4), so we compute h_u using the
    # 4th inner product entry
    h_u_components_gamma_2_3_target = torch.exp((-1 * torch.Tensor([2, 10, 5, 3])) / (2 * torch.pow(gamma_2, 2)))
    h_u_components_gamma_2_0_target = torch.exp((-1 * torch.Tensor([22, 14, 2, 2])) / (2 * torch.pow(gamma_2, 2)))

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


def test_compute_all_h_u_from_inner_products() -> None:
    # shape should be num_kernels x number v_is
    assert all_h_us.shape == (3, 5)
    h_u_gamma_1 = mkmmd_loss.compute_h_u_from_inner_products(inner_products, torch.Tensor([2]).to(DEVICE))
    h_u_gamma_2 = mkmmd_loss.compute_h_u_from_inner_products(inner_products, torch.Tensor([1]).to(DEVICE))
    h_u_gamma_3 = mkmmd_loss.compute_h_u_from_inner_products(inner_products, torch.Tensor([0.5]).to(DEVICE))
    assert torch.all(all_h_us.eq(torch.cat([h_u_gamma_1, h_u_gamma_2, h_u_gamma_3], 0)))

    # test run all the way through from X, Y
    all_h_us_full = mkmmd_loss.compute_all_h_u_per_v_i(X, Y)
    assert torch.all(all_h_us_full.eq(all_h_us))


def test_compute_hat_d_per_kernel() -> None:
    # shapre should be num_kernels x 1
    hat_d_per_kernel.shape == (3, 1)
    assert hat_d_per_kernel[0, 0] == (1 / 5) * (torch.sum(all_h_us[0, :]))
    assert hat_d_per_kernel[1, 0] == (1 / 5) * (torch.sum(all_h_us[1, :]))
    assert hat_d_per_kernel[2, 0] == (1 / 5) * (torch.sum(all_h_us[2, :]))


def test_compute_mkmmd() -> None:
    betas = torch.Tensor([1.5, 2.0, -1.0]).reshape(-1, 1).to(DEVICE)
    mkmmd_target = hat_d_per_kernel[0, 0] * 1.5 + hat_d_per_kernel[1, 0] * 2.0 + hat_d_per_kernel[2, 0] * (-1.0)
    assert mkmmd_loss.compute_mkmmd(X, Y, betas) == mkmmd_target


def test_create_h_u_delta_w_i() -> None:
    # shape should be num num_kernels x v_is // 2 (noting that we drop the last v_i if uneven number)
    assert h_u_delta_w_i.shape == (3, 2)
    assert h_u_delta_w_i[0, 0] == all_h_us[0, 0] - all_h_us[0, 1]
    assert h_u_delta_w_i[0, 1] == all_h_us[0, 2] - all_h_us[0, 3]
    assert h_u_delta_w_i[1, 0] == all_h_us[1, 0] - all_h_us[1, 1]
    assert h_u_delta_w_i[1, 1] == all_h_us[1, 2] - all_h_us[1, 3]
    assert h_u_delta_w_i[2, 0] == all_h_us[2, 0] - all_h_us[2, 1]
    assert h_u_delta_w_i[2, 1] == all_h_us[2, 2] - all_h_us[2, 3]


def test_compute_hat_Q_k() -> None:
    # shape should be number of kernels x number of kernels
    assert Q_k.shape == (3, 3)
    assert Q_k[0, 0] == (0.5) * (h_u_delta_w_i[0, 0] * h_u_delta_w_i[0, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[0, 1])
    assert Q_k[0, 1] == (0.5) * (h_u_delta_w_i[0, 0] * h_u_delta_w_i[1, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[1, 1])
    assert Q_k[1, 0] == (0.5) * (h_u_delta_w_i[0, 0] * h_u_delta_w_i[1, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[1, 1])
    assert Q_k[0, 2] == (0.5) * (h_u_delta_w_i[0, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[2, 1])
    assert Q_k[2, 0] == (0.5) * (h_u_delta_w_i[0, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[0, 1] * h_u_delta_w_i[2, 1])
    assert Q_k[1, 1] == (0.5) * (h_u_delta_w_i[1, 0] * h_u_delta_w_i[1, 0] + h_u_delta_w_i[1, 1] * h_u_delta_w_i[1, 1])
    assert Q_k[1, 2] == (0.5) * (h_u_delta_w_i[1, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[1, 1] * h_u_delta_w_i[2, 1])
    assert Q_k[2, 1] == (0.5) * (h_u_delta_w_i[1, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[1, 1] * h_u_delta_w_i[2, 1])
    assert Q_k[2, 2] == (0.5) * (h_u_delta_w_i[2, 0] * h_u_delta_w_i[2, 0] + h_u_delta_w_i[2, 1] * h_u_delta_w_i[2, 1])


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


def test_optimize_betas_degenerate_case() -> None:
    # The simple test cases above result in a set of hat_ds that are all negative. In this case, we perform the
    # selection of a single kernel as recommended in Gretton
    degenerate_betas = mkmmd_loss.optimize_betas(X, Y, lambda_m=0.0001)
    beta_target = torch.Tensor([[1, 0, 0]]).to(DEVICE).t()
    assert torch.all(degenerate_betas.eq(beta_target))


def test_gamma_defaults() -> None:
    default_mkmmd = MkMmdLoss(DEVICE)
    neg_gamma_powers = [-3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0]
    pos_gamma_powers = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]
    gamma_powers = neg_gamma_powers + pos_gamma_powers
    default_gammas = torch.Tensor([2**power for power in gamma_powers])
    assert torch.all(default_mkmmd.gammas.eq(default_gammas))
    assert torch.all(torch.where(default_mkmmd.betas >= 0.0, True, False))
    assert pytest.approx(torch.sum(default_mkmmd.betas).item(), abs=0.0001) == 1.0


def test_get_best_vertex_for_objective_function() -> None:
    lambda_m = 0.0001
    regularized_Q_k = 2 * Q_k + lambda_m * torch.eye(3).to(DEVICE)
    best_vertex = mkmmd_loss.get_best_vertex_for_objective_function(hat_d_per_kernel, regularized_Q_k)
    assert best_vertex.shape == (3, 1)
    # check this
    assert pytest.approx(best_vertex[0, 0].item(), abs=0.0001) == 1.0
    assert pytest.approx(best_vertex[1, 0].item(), abs=0.0001) == 0.0
    assert pytest.approx(best_vertex[2, 0].item(), abs=0.0001) == 0.0
    assert pytest.approx(torch.mm(hat_d_per_kernel.t(), best_vertex), abs=0.0001) == 1.0


def test_optimizer_betas_in_non_degenerate_case() -> None:
    lambda_m = 0.0001
    torch.manual_seed(42)
    default_mkmmd = MkMmdLoss(DEVICE)
    # First sample is from a zero mean gaussian with unit covariance (dimension 5)
    p = MultivariateNormal(torch.zeros(5), torch.eye(5))
    X = p.sample(
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
    Y = (
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
    all_h_u_per_vi_local = default_mkmmd.compute_all_h_u_per_v_i(X, Y)
    hat_d_per_kernel_local = default_mkmmd.compute_hat_d_per_kernel(all_h_u_per_vi_local)
    mkmmd_before_opt = default_mkmmd(X, Y)
    assert pytest.approx(mkmmd_before_opt.item(), abs=0.0001) == 0.02652

    hat_Q_k = default_mkmmd.compute_hat_Q_k(all_h_u_per_vi_local)
    regularized_hat_Q_k = (2 * hat_Q_k + lambda_m * torch.eye(29)).to(DEVICE)
    unnormalized_betas = default_mkmmd.form_and_solve_qp(hat_d_per_kernel_local, regularized_hat_Q_k)
    assert pytest.approx(torch.mm(unnormalized_betas.t(), hat_d_per_kernel_local), abs=0.0001) == 1.0000

    betas_local = default_mkmmd.optimize_betas(X, Y, lambda_m)
    assert torch.all(betas_local.eq((1 / torch.sum(unnormalized_betas)) * unnormalized_betas))

    default_mkmmd = MkMmdLoss(DEVICE, minimize_type_two_error=False)
    betas_local = default_mkmmd.optimize_betas(X, Y, lambda_m)
    one_hot_betas = torch.zeros_like(betas_local)
    one_hot_betas[0, 0] = 1.0
    assert torch.all(betas_local.eq(one_hot_betas))
