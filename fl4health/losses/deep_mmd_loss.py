from typing import Optional, Tuple

import numpy as np
import torch


class ModelLatentF(torch.nn.Module):
    def __init__(self, x_in_dim: int, hidden_dim: int, x_out_dim: int):
        """
        Deep network for learning the deep kernel over features.

        Args:
            x_in_dim (int): The input dimension of the deep network.
            hidden_dim (int): The hidden dimension of the deep network.
            x_out_dim (int): The output dimension of the deep network.
        """
        super().__init__()
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in_dim, hidden_dim, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, x_out_dim, bias=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the deep network.

        Args:
            input (torch.Tensor): The input tensor to the deep network.

        Returns:
            torch.Tensor: The output tensor of the deep network.
        """
        feature_latent_map = self.latent(input)
        return feature_latent_map


class DeepMmdLoss(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_size: int,
        hidden_size: int = 10,
        output_size: int = 50,
        lr: float = 0.001,
        training: bool = True,
        is_unbiased: bool = True,
        gaussian_degree: int = 1,
        optimization_steps: int = 5,
    ) -> None:
        """
        Compute the Deep MMD (Maximum Mean Discrepancy) loss, as proposed in the paper Learning Deep Kernels for
        Non-Parametric Two-Sample Tests. This loss function uses a kernel-based approach to assess whether two
        samples are drawn from the same distribution. By minimizing this loss, we can learn a deep kernel that
        reduces the MMD distance between two distributions, ensuring that the input feature representations are
        aligned. This implementation is inspired by the original code from the paper:
        https://github.com/fengliu90/DK-for-TST.

        Args:
            device (torch.device): Device onto which tensors should be moved
            input_size (int): The length of the input feature representations of the deep network as the deep
                kernel used to compute the MMD loss.
            hidden_size (int, optional): The hidden size of the deep network as the deep kernel used to compute
                the MMD loss. Defaults to 10.
            output_size (int, optional): The output size of the deep network as the deep kernel used to compute
                the MMD loss. Defaults to 50.
            lr (float, optional): Learning rate for training the Deep Kernel. Defaults to 0.001.
            training (bool, optional): Whether the Deep Kernel is in training mode. Defaults to True.
            is_unbiased (bool, optional): Whether to use the unbiased estimator for the MMD loss. Defaults to True.
            gaussian_degree (int, optional): The degree of the generalized Gaussian kernel. Defaults to 1.
            optimization_steps (int, optional): The number of optimization steps to train the Deep Kernel in each
                forward pass. Defaults to 5.
        """

        super().__init__()
        self.device = device
        self.lr = lr
        self.training = training
        self.is_unbiased = is_unbiased
        self.gaussian_degree = gaussian_degree  # generalized Gaussian (if L>1)
        self.optimization_steps = optimization_steps

        # Initialize the model
        self.featurizer = ModelLatentF(input_size, hidden_size, output_size).to(self.device)

        # Initialize parameters
        self.epsilon_opt: torch.Tensor = torch.log(torch.from_numpy(np.random.rand(1) * 10 ** (-10)).to(self.device))
        self.epsilon_opt.requires_grad = self.training
        self.sigma_q_opt: torch.Tensor = torch.sqrt(torch.tensor(2 * 32 * 32, dtype=torch.float).to(self.device))
        self.sigma_q_opt.requires_grad = self.training
        self.sigma_phi_opt: torch.Tensor = torch.sqrt(torch.tensor(0.005, dtype=torch.float).to(self.device))
        self.sigma_phi_opt.requires_grad = self.training

        # Initialize optimizers
        self.optimizer_F = torch.optim.AdamW(
            list(self.featurizer.parameters()) + [self.epsilon_opt] + [self.sigma_q_opt] + [self.sigma_phi_opt],
            lr=self.lr,
        )

    def pairwise_distiance_squared(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the paired distance between x and y.

        Args:
            X (torch.Tensor): The input tensor X.
            Y (torch.Tensor): The input tensor Y.

        Returns:
            torch.Tensor: The paired distance between X and Y.
        """
        x_norm = (X**2).sum(1).view(-1, 1)
        y_norm = (Y**2).sum(1).view(1, -1)
        paired_distance = x_norm + y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
        paired_distance[paired_distance < 0] = 0
        return paired_distance

    def h1_mean_var_gram(
        self,
        k_x: torch.Tensor,
        k_y: torch.Tensor,
        k_xy: torch.Tensor,
        is_var_computed: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute value of MMD and std of MMD using kernel matrix.

        Args:
            k_x (torch.Tensor): The kernel matrix of x.
            k_y (torch.Tensor): The kernel matrix of y.
            k_xy (torch.Tensor): The kernel matrix of x and y.
            is_var_computed (bool): Whether to compute the variance of the MMD.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The value of MMD and the variance of MMD
                if required to compute.
        """
        nx = k_x.shape[0]
        ny = k_y.shape[0]

        if self.is_unbiased:
            # compute the unbiased MMD estimator (\hat{\text{MMD}}_u^2) defined in Eq. (2) of the paper
            xx = torch.div((torch.sum(k_x) - torch.sum(torch.diag(k_x))), (nx * (nx - 1)))
            yy = torch.div((torch.sum(k_y) - torch.sum(torch.diag(k_y))), (ny * (ny - 1)))
            xy = torch.div((torch.sum(k_xy) - torch.sum(torch.diag(k_xy))), (nx * (ny - 1)))

        else:
            # compute the biased MMD estimator (\hat{\text{MMD}}_b^2) defined below Equation (2) of the paper
            xx = torch.div((torch.sum(k_x)), (nx * nx))
            yy = torch.div((torch.sum(k_y)), (ny * ny))
            xy = torch.div((torch.sum(k_xy)), (nx * ny))

        mmd2 = xx - 2 * xy + yy
        if not is_var_computed:
            return mmd2, None
        h_ij = k_x + k_y - k_xy - k_xy.transpose(0, 1)

        # Compute the variance estimate of MMD defined in Equation (5) of the paper
        v1 = (4.0 / ny**3) * (torch.dot(h_ij.sum(1), h_ij.sum(1)))
        v2 = (4.0 / nx**4) * (h_ij.sum() ** 2)
        variance_estimate = v1 - v2 + (10 ** (-8))
        return mmd2, variance_estimate

    def MMDu(
        self,
        features: torch.Tensor,
        len_s: int,
        features_org: torch.Tensor,
        sigma_q: torch.Tensor,
        sigma_phi: torch.Tensor,
        epsilon: torch.Tensor,
        is_smooth: bool = True,
        is_var_computed: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute value of deep-kernel MMD and std of deep-kernel MMD using merged data.

        Args:
            features (torch.Tensor): The output features of the deep network.
            len_s (int): The length of the sample.
            features_org (torch.Tensor): The original input features of the deep network.
            sigma_q (torch.Tensor): The sigma_q parameter.
            sigma_phi (torch.Tensor): The sigma_phi parameter.
            epsilon (torch.Tensor): The epsilon parameter.
            is_smooth (bool, optional): Whether to use the smooth version of the MMD.
                Defaults to True.
            is_var_computed (bool, optional): Whether to compute the variance of the MMD.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The value of MMD and the variance of MMD
                if required to compute.
        """
        x = features[0:len_s, :]  # fetch the sample 1 (features of deep networks)
        y = features[len_s:, :]  # fetch the sample 2 (features of deep networks)
        distance_xx = self.pairwise_distiance_squared(x, x)
        distance_yy = self.pairwise_distiance_squared(y, y)
        distance_xy = self.pairwise_distiance_squared(x, y)

        if is_smooth:
            x_original = features_org[0:len_s, :]  # fetch the original sample 1
            y_original = features_org[len_s:, :]  # fetch the original sample 2
            distance_xx_original = self.pairwise_distiance_squared(x_original, x_original)
            distance_yy_original = self.pairwise_distiance_squared(y_original, y_original)
            distance_xy_original = self.pairwise_distiance_squared(x_original, y_original)

            kernel_x = (1 - epsilon) * torch.exp(
                -((distance_xx / sigma_phi) ** self.gaussian_degree) - distance_xx_original / sigma_q
            ) + epsilon * torch.exp(-distance_xx_original / sigma_q)
            kernel_y = (1 - epsilon) * torch.exp(
                -((distance_yy / sigma_phi) ** self.gaussian_degree) - distance_yy_original / sigma_q
            ) + epsilon * torch.exp(-distance_yy_original / sigma_q)
            kernel_xy = (1 - epsilon) * torch.exp(
                -((distance_xy / sigma_phi) ** self.gaussian_degree) - distance_xy_original / sigma_q
            ) + epsilon * torch.exp(-distance_xy_original / sigma_q)

        else:
            kernel_x = torch.exp(-distance_xx / sigma_phi)
            kernel_y = torch.exp(-distance_yy / sigma_phi)
            kernel_xy = torch.exp(-distance_xy / sigma_phi)

        # kernel_x reprsents k_w(x_i, x_j), kernel_y represents k_w(y_i, y_j), kernel_xy represents
        # k_w(x_i, y_j) for all i, j in the sample X and sample Y defined in Equation (1) of the paper
        return self.h1_mean_var_gram(kernel_x, kernel_y, kernel_xy, is_var_computed)

    def train_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Train the Deep MMD kernel.

        Args:
            X (torch.Tensor): The input tensor X.
            Y (torch.Tensor): The input tensor Y.
        """

        self.featurizer.train()
        self.sigma_q_opt.requires_grad = True
        self.sigma_phi_opt.requires_grad = True
        self.epsilon_opt.requires_grad = True

        features = torch.cat([X, Y], 0)

        # ------------------------------
        #  Train deep network for MMD-D
        # ------------------------------
        # Zero gradients
        self.optimizer_F.zero_grad()
        # Compute output of deep network
        model_output = self.featurizer(features)
        # Compute epsilon, sigma_q and sigma_phi in \kappa_w(x, y) in Equation (1) of the paper
        epsilon = torch.exp(self.epsilon_opt) / (1 + torch.exp(self.epsilon_opt))
        sigma_q = self.sigma_q_opt**2
        sigma_phi = self.sigma_phi_opt**2
        # Compute Deep MMD value and variance estimates
        mmd_value_estimate, mmd_var_estimate = self.MMDu(
            features=model_output,
            len_s=X.shape[0],
            features_org=features.view(features.shape[0], -1),
            sigma_q=sigma_q,
            sigma_phi=sigma_phi,
            epsilon=epsilon,
            is_var_computed=True,
        )
        if mmd_var_estimate is None:
            raise AssertionError("Error: Variance of MMD is not computed. Please set is_var_computed=True.")
        mmd_std_estimate = torch.sqrt(mmd_var_estimate)
        # Forming \hat{J}_{\lambda} defined in Equation (4) of the paper (STAT_u)
        stat_u = torch.div(-1 * mmd_value_estimate, mmd_std_estimate)
        # Compute gradient
        stat_u.backward()
        # Update weights using gradient descent
        self.optimizer_F.step()

    def compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Deep MMD Loss.

        Args:
            X (torch.Tensor): The input tensor X.
            Y (torch.Tensor): The input tensor Y.

        Returns:
            torch.Tensor: The value of Deep MMD Loss.
        """

        self.featurizer.eval()
        self.sigma_q_opt.requires_grad = False
        self.sigma_phi_opt.requires_grad = False
        self.epsilon_opt.requires_grad = False

        features = torch.cat([X, Y], 0)

        # Compute output of deep network
        model_output = self.featurizer(features)
        # Compute epsilon, sigma_q and sigma_phi in \kappa_w(x, y) in Equation (1) of the paper
        epsilon = torch.exp(self.epsilon_opt) / (1 + torch.exp(self.epsilon_opt))
        sigma_q = self.sigma_q_opt**2
        sigma_phi = self.sigma_phi_opt**2
        # Compute Deep MMD value estimates
        mmd_value_estimate, _ = self.MMDu(
            features=model_output,
            len_s=X.shape[0],
            features_org=features.view(features.shape[0], -1),
            sigma_q=sigma_q,
            sigma_phi=sigma_phi,
            epsilon=epsilon,
            is_var_computed=False,
        )

        return mmd_value_estimate

    def forward(self, Xs: torch.Tensor, Xt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Deep MMD Loss where it first trains the deep kernel for number of optimization
        steps and then computes the MMD loss.

        Args:
            Xs (torch.Tensor): The source input tensor.
            Xt (torch.Tensor): The target input tensor.

        Returns:
            torch.Tensor: The value of Deep MMD Loss.
        """

        if self.training:
            for _ in range(self.optimization_steps):
                self.train_kernel(Xs.clone().detach(), Xt.clone().detach())

        return self.compute_kernel(Xs, Xt)
