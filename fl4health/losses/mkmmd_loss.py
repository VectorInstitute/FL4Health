# multi-kernel maximum mean discrepancy
# cao bin, HKUST, China, binjacobcao@gmail.com
# free to charge for academic communication

from logging import INFO
from typing import Optional

import torch
from flwr.common.logger import log
from qpth.qp import QPFunction, QPSolvers


class MkMmdLoss(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        gammas: Optional[torch.Tensor] = None,
        betas: Optional[torch.Tensor] = None,
        minimize_type_two_error: bool = True,
    ) -> None:
        """
        Compute the multi-kernel maximum mean discrepancy (MK-MMD) between the source and target domains. Also allows
        for optimization of the coefficients, beta

        Args:
            device (torch.device): Device onto which tensors should be moved
            gammas (Optional[torch.Tensor], optional): These are known as the length-scales of the RBF functions used
                to compute the Mk-MMD distances. The length of this list defines the number of kernels used in the
                norm measurement. If none, a default of 29 kernels is used. Defaults to None.
            betas (Optional[torch.Tensor], optional): These are the linear coefficients used on the basis of kernels
                to compute the Mk-MMD measure. If not provided, a unit-length, random default is constructed. These
                can be optimized using the functions of this class. Defaults to None.
            minimize_type_two_error (Optional[bool], optional): Whether we're aiming to minimize the type II error in
                optimizing the betas or maximize it. The first coincides with trying to minimize feature distance. The
                second coincides with trying to maximize their feature distance. Defaults to True.
        """
        super().__init__()
        self.device = device
        if gammas is None:
            # Note arange is not inclusive, so this ends up being [-3.5, 3.5] in steps of 0.5
            default_gamma_powers = torch.arange(-3.5, 3.75, 0.25, device=device)
            self.gammas = torch.pow(2.0, default_gamma_powers)
        else:
            self.gammas = gammas
        self.kernel_num = len(self.gammas)

        if betas is None:
            rand_coefficients = torch.rand((self.kernel_num, 1)).to(self.device)
            # normalize the coefficients to sum to 1
            self.betas = (1 / torch.sum(rand_coefficients)) * rand_coefficients
        else:
            assert betas.shape == (self.kernel_num, 1)
            self.betas = betas.to(self.device)

        self.minimize_type_two_error = minimize_type_two_error

    def construct_quadruples(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        In this function, we assume that X, Y: n_samples, n_features are the same size. We construct the quadruples
        v_i = [x_{2i-1}, x_{2i}, y_{2i-1}, y_{2i}] forming a matrix of dimension n_samples/2, 4, n_features
        Note that if n_samples is not divisible by 2, we leave off the modulus
        """
        n_samples, n_features = X.shape
        # truncate if not divisible by 2
        if n_samples % 2 == 1:
            X = X[:-1, :]
            Y = Y[:-1, :]
        v_i = torch.cat((X.reshape(n_samples // 2, 2, n_features), Y.reshape(n_samples // 2, 2, n_features)), dim=1)
        return v_i

    def compute_euclidean_inner_products(self, v_i_quadruples: torch.Tensor) -> torch.Tensor:
        # We want to compute the RBF kernel values. To do this, we need to compute ||x - y||^2 for the relevant pairs
        # x and y. That is the inner product. Note that ||x - y||^2 = <x - y, x - y> = (x-y)^T(x-y)
        # For the quadruples of the form (x,  x', y, y') we need distances for (x, x'), (y, y'), (x, y'), (x, y')
        # NOTE: The inner product for a vector x^Tx is just the square and sum of the components of x
        x_x_prime = torch.sum((v_i_quadruples[:, 0, :] - v_i_quadruples[:, 1, :]) ** 2, dim=1, keepdim=True)
        y_y_prime = torch.sum((v_i_quadruples[:, 2, :] - v_i_quadruples[:, 3, :]) ** 2, dim=1, keepdim=True)
        x_y_prime = torch.sum((v_i_quadruples[:, 0, :] - v_i_quadruples[:, 3, :]) ** 2, dim=1, keepdim=True)
        x_prime_y = torch.sum((v_i_quadruples[:, 1, :] - v_i_quadruples[:, 2, :]) ** 2, dim=1, keepdim=True)
        # each inner product is a tensor of dimension len(v_i_quadruples), we return a tensor of shape
        # len(v_i_quadruples) x 4
        return torch.cat([x_x_prime, y_y_prime, x_y_prime, x_prime_y], dim=1)

    def compute_h_u_from_inner_products(
        self, inner_product_quadruples: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        # Gamma should be of shape torch.Tensor([gamma])
        assert gamma.shape == (1,)
        # inner_product_quadruples has shape number of v_i_quadruples x 4
        # h_u_componets should have the same shape
        h_u_components = torch.exp((-1 * inner_product_quadruples) / (2 * torch.pow(gamma, 2)))
        # Each column of h_u_components should now be u(x_{2i-1}, x_{2i}), u(y_{2i-1}, y_{2i}), u(x_{2i-1}, y_{2i}),
        # and u(x_{2i}, y_{2i-1}), where u is the kernel_index^th kernel
        # So we compute:
        #   h_u[x_{2i-1}, x_{2i},y_{2i-1}, y_{2i}] =  u(x_{2i-1}, x_{2i}) + u(y_{2i-1}, y_{2i})
        # - u(x_{2i-1}, y_{2i}) - u(x_{2i}, y_{2i-1})
        h_u = h_u_components[:, 0] + h_u_components[:, 1] - h_u_components[:, 2] - h_u_components[:, 3]
        # this results in a matrix of shape 1 x number of v_i_quadruples
        return h_u.unsqueeze(0)

    def compute_all_h_u_from_inner_products(self, inner_product_quadruples: torch.Tensor) -> torch.Tensor:
        k_list = [
            self.compute_h_u_from_inner_products(inner_product_quadruples, gamma.reshape(1)) for gamma in (self.gammas)
        ]
        # Matrix should be of shape number of kernels x number of quadruples, since we compute the kernel value on all
        # quadrules for every kernel
        return torch.cat(k_list)

    def compute_all_h_u_per_v_i(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # v_i = [x_{2i-1}, x_{2i}, y_{2i-1}, y_{2i}]
        v_is = self.construct_quadruples(X, Y)
        # For the quadruples of the form (x,  x', y, y') we need distances for (x, x'), (y, y'), (x, y'), (x, y')
        inner_product_quadruples = self.compute_euclidean_inner_products(v_is)
        # all_h_u has shape number of kernels x number of quadruples
        all_h_u = self.compute_all_h_u_from_inner_products(inner_product_quadruples)
        return all_h_u

    def compute_hat_d_per_kernel(self, all_h_u_per_v_i: torch.Tensor) -> torch.Tensor:
        # all_h_u_per_vi has shape number kernels x number of v_is. Each row is h_u(v_i) for kernel u with the ith
        # column being that kernel evaluated on v_i.
        # output shape is number of kernels x 1
        return torch.mean(all_h_u_per_v_i, dim=1, keepdim=True)

    def compute_mkmmd(self, X: torch.Tensor, Y: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # In this function, we assume that X, Y: n_samples, n_features are the same size and that beta is a tensor of
        # shape number of kernels x 1
        all_h_u_per_vi = self.compute_all_h_u_per_v_i(X, Y)
        hat_d_per_kernel = self.compute_hat_d_per_kernel(all_h_u_per_vi)
        # Take the dot product between the indivudal kernel hat_d values to scale by the basis coefficients
        return torch.mm(beta.t(), hat_d_per_kernel).squeeze(0)

    def create_h_u_delta_w_i(self, all_h_u_per_v_i: torch.Tensor) -> torch.Tensor:
        # all_h_u_per_v_i has dimension number kernels x number of v_i quadruples
        n_kernels, n_v_i_s = all_h_u_per_v_i.shape
        # if the number of v_is is not divisible by two just drop the final ones
        if n_v_i_s % 2 == 1:
            all_h_u_per_v_i = all_h_u_per_v_i[:, :-1]
        # pairing the h_u(v_i) values as h_u(v_{2i-1}), h_u(v_{2i}), think w_is from the paper
        h_u_v_i_pairs = all_h_u_per_v_i.reshape(n_kernels, n_v_i_s // 2, 2)
        return h_u_v_i_pairs[:, :, 0] - h_u_v_i_pairs[:, :, 1]

    def compute_hat_Q_k(self, all_h_u_per_v_i: torch.Tensor) -> torch.Tensor:
        # all_h_u_per_v_i has dimension number kernels x number of v_i quadruples
        h_u_delta_w_i = self.create_h_u_delta_w_i(all_h_u_per_v_i)

        Q_k_matrix: torch.Tensor = torch.zeros((self.kernel_num, self.kernel_num)).to(self.device)
        len_w_is = h_u_delta_w_i.shape[1]
        for i in range(len_w_is):
            # For each basis function we're adding in the value of h_{j, \Delta}(w_i)*h_{k, \Delta}(w_i) from the
            # construction above to the proper entry in Q. Note that Q is symmetric. So we can construct the symmetric
            # entries at the same time.
            for j in range(self.kernel_num):
                for k in range(j + 1):
                    Q_k_matrix[j][k] += h_u_delta_w_i[j][i] * h_u_delta_w_i[k][i]
                    if j != k:
                        Q_k_matrix[k][j] += h_u_delta_w_i[j][i] * h_u_delta_w_i[k][i]
        # Q_k_matrix has shape number of kernels x number of kernels
        return Q_k_matrix / len_w_is

    def beta_with_extreme_kernel_base_values(
        self, hat_d_per_kernel: torch.Tensor, hat_Q_k: torch.Tensor, minimize_type_two_error: bool = True
    ) -> torch.Tensor:
        kernel_base_values = torch.tensor(
            [hat_d_per_kernel[i] / hat_Q_k[i][i] for i in range(len(hat_d_per_kernel))]
        ).to(self.device)
        if minimize_type_two_error:
            log(
                INFO,
                "Rather than optimizing, we select a single kernel with largest hat_d_k/hat_Q_k_lambda",
            )
            largest_kernel_index = torch.argmax(kernel_base_values)
        else:
            log(
                INFO,
                "Rather than optimizing, we select a single kernel with smallest hat_d_k/hat_Q_k_lambda",
            )
            largest_kernel_index = torch.argmin(kernel_base_values)
        beta_one_hot = torch.zeros_like(hat_d_per_kernel)
        beta_one_hot[largest_kernel_index] = 1.0
        return beta_one_hot

    def compute_vertices(self, hat_d_per_kernel: torch.Tensor) -> torch.Tensor:
        return 1.0 / hat_d_per_kernel

    def get_best_vertex_for_objective_function(
        self, hat_d_per_kernel: torch.Tensor, hat_Q_k: torch.Tensor
    ) -> torch.Tensor:
        # vertices have shape num kernels x 1
        vertices_weights = self.compute_vertices(hat_d_per_kernel)
        maximum_value = -torch.inf
        best_index = 0
        best_vertex = torch.zeros_like(hat_d_per_kernel).to(self.device)
        for i in range(self.kernel_num):
            vertices = torch.zeros_like(hat_d_per_kernel).to(self.device)
            vertices[i, 0] = vertices_weights[i, 0]
            objective_value = torch.mm(torch.mm(vertices.t(), hat_Q_k), vertices).item()
            if objective_value > maximum_value:
                maximum_value = objective_value
                best_index = i
        best_vertex[best_index, 0] = vertices_weights[best_index, 0]
        return best_vertex

    def form_and_solve_qp(self, hat_d_per_kernel: torch.Tensor, regularized_Q_k: torch.Tensor) -> torch.Tensor:
        # p = \vec{0} of shape (number of kernels, 1). It is only used for the QP setup
        p = torch.zeros(self.kernel_num).to(self.device)

        # We want each beta >= 0, the QP defines the constraint as G\beta <= h. So h is a vector of zeros (see below)
        # and we want each -1*beta <=0 to guarantee that beta>=0. So G is an identify matrix scaled by -1
        G = -1 * torch.eye(self.kernel_num).to(self.device)

        # This is the RHS of the inequality constraint on the beta coefficients. We want them to be >= 0
        h = torch.zeros(self.kernel_num).to(self.device)

        # We want the beta^T * hat_d_per_kernel = 1. Translated into the form of the quadratic program below,
        # this means A is essentially n_p_k
        b = torch.tensor([1.0]).to(self.device)

        # Solve a batch of QPs.

        #   This function solves a batch of QPs, each optimizing over
        #   `nz` variables and having `nineq` inequality constraints
        #   and `neq` equality constraints.
        #   The optimization problem for each instance in the batch
        #   (dropping indexing from the notation) is of the form

        #     \hat z =   argmin_z 1/2 z^T Q z + p^T z
        #             subject to Gz <= h
        #                         Az  = b
        # Dimensions are as follows:
        #   Q is (kernel_num, kernel_num)
        #   p is (kernel_num, 1)
        #   G is (kernel_num, kernel_num)
        #   h is (kernel_num, 1)
        #   A is (1, kernel_num)
        #   b is a scalar

        # QPFunction returns betas in the shape 1 x num_kernels to we take the transpose for consistency
        return QPFunction(verbose=False, solver=QPSolvers.CVXPY, check_Q_spd=False)(
            regularized_Q_k, p, G, h, hat_d_per_kernel.t(), b
        ).t()

    def optimize_betas(self, X: torch.Tensor, Y: torch.Tensor, lambda_m: float = 1e-5) -> torch.Tensor:
        # In this function, we assume that X, Y: n_samples, n_features
        all_h_u_per_v_i = self.compute_all_h_u_per_v_i(X, Y)

        # shape of hat_d_per_kernel is number of kernels x 1
        hat_d_per_kernel = self.compute_hat_d_per_kernel(all_h_u_per_v_i)
        # check to see that at least one of them is positive. If none of them are positive, then select a single kernel
        # with largest hat_d, similar to the suggestion of Gretton et al. in "Optimal Kernel Choice for Large-Scale
        # Two-Sample Tests", 2012

        # shape of hat_Q_k is number of kernels x number of kernels
        hat_Q_k = self.compute_hat_Q_k(all_h_u_per_v_i)
        # Eigen shift hat_Q_k and scale by 2 as the QP setup scales by 1/2
        regularized_Q_k = 2 * hat_Q_k + lambda_m * torch.eye(self.kernel_num).to(self.device)

        if not torch.any(hat_d_per_kernel > 0):
            log(INFO, f"None of the estimates for hat_d are positive: {hat_d_per_kernel.squeeze()}.")
            return self.beta_with_extreme_kernel_base_values(
                hat_d_per_kernel, regularized_Q_k, minimize_type_two_error=True
            )

        if self.minimize_type_two_error:
            unnormalized_betas = self.form_and_solve_qp(hat_d_per_kernel, regularized_Q_k)
        else:
            # If we're trying to maximize the type II error, then we are trying to maximize a convex function over a
            # convex polygon of beta values. So the maximum is found at one of the vertices
            unnormalized_betas = self.get_best_vertex_for_objective_function(hat_d_per_kernel, regularized_Q_k)
        optimized_betas = (1.0 / torch.sum(unnormalized_betas)) * unnormalized_betas
        return optimized_betas

    def forward(self, Xs: torch.Tensor, Xt: torch.Tensor) -> torch.Tensor:
        """Compute the multi-kernel maximum mean discrepancy (MK-MMD) between the source and target domains.

        Args:
            Xs (torch.Tensor): Source domain data, shape (n_samples, n_features)
            Xt (torch.Tensor): Target domain data, shape (n_samples, n_features)

        Returns:
            torch.Tensor: MK-MMD value
        """

        return self.compute_mkmmd(Xs, Xt, self.betas)
