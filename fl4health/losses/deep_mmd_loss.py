from typing import Optional, Tuple

import numpy as np
import torch


class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""

    def __init__(self, x_in_dim: int, hidden_dim: int, x_out_dim: int):
        """Init latent features."""
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
        """Forward the LeNet."""
        feature_latent_map = self.latent(input)
        return feature_latent_map


class DeepMmdLoss(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_size: int,
        lr: float = 0.001,
        hidden_size: int = 10,
        output_size: int = 50,
        training: bool = True,
        is_unbiased: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.lr = lr
        self.training = training

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
            list(self.featurizer.parameters()) + [self.epsilon_opt] + [self._q_opt] + [self.sigma_phi_opt], lr=self.lr
        )

        self.is_unbiased = is_unbiased
        self.L = 1  # generalized Gaussian (if L>1)

    def pairwise_distiance_squared(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """compute the paired distance between x and y."""
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
        use_1sample_U: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """compute value of MMD and std of MMD using kernel matrix."""
        nx = k_x.shape[0]
        ny = k_y.shape[0]

        if self.is_unbiased:
            xx = torch.div((torch.sum(k_x) - torch.sum(torch.diag(k_x))), (nx * (nx - 1)))
            yy = torch.div((torch.sum(k_y) - torch.sum(torch.diag(k_y))), (ny * (ny - 1)))
            # one-sample U-statistic.
            if use_1sample_U:
                xy = torch.div((torch.sum(k_xy) - torch.sum(torch.diag(k_xy))), (nx * (ny - 1)))
            else:
                xy = torch.div(torch.sum(k_xy), (nx * ny))
        else:
            xx = torch.div((torch.sum(k_x)), (nx * nx))
            yy = torch.div((torch.sum(k_y)), (ny * ny))
            xy = torch.div((torch.sum(k_xy)), (nx * ny))

        mmd2 = xx - 2 * xy + yy
        if not is_var_computed:
            return mmd2, None
        h_ij = k_x + k_y - k_xy - k_xy.transpose(0, 1)

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
        use_1sample_U: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
        x = features[0:len_s, :]  # fetch the sample 1 (features of deep networks)
        y = features[len_s:, :]  # fetch the sample 2 (features of deep networks)
        d_xx = self.pairwise_distiance_squared(x, x)
        d_yy = self.pairwise_distiance_squared(y, y)
        d_xy = self.pairwise_distiance_squared(x, y)
        if is_smooth:
            x_original = features_org[0:len_s, :]  # fetch the original sample 1
            y_original = features_org[len_s:, :]  # fetch the original sample 2
            d_xx_original = self.pairwise_distiance_squared(x_original, x_original)
            d_yy_original = self.pairwise_distiance_squared(y_original, y_original)
            d_xy_original = self.pairwise_distiance_squared(x_original, y_original)
            k_x = (1 - epsilon) * torch.exp(
                -((d_xx / sigma_phi) ** self.L) - d_xx_original / sigma_q
            ) + epsilon * torch.exp(-d_xx_original / sigma_q)
            k_y = (1 - epsilon) * torch.exp(
                -((d_yy / sigma_phi) ** self.L) - d_yy_original / sigma_q
            ) + epsilon * torch.exp(-d_yy_original / sigma_q)
            k_xy = (1 - epsilon) * torch.exp(
                -((d_xy / sigma_phi) ** self.L) - d_xy_original / sigma_q
            ) + epsilon * torch.exp(-d_xy_original / sigma_q)
        else:
            k_x = torch.exp(-d_xx / sigma_phi)
            k_y = torch.exp(-d_yy / sigma_phi)
            k_xy = torch.exp(-d_xy / sigma_phi)

        return self.h1_mean_var_gram(k_x, k_y, k_xy, is_var_computed, use_1sample_U)

    def train_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Train the Deep MMD kernel."""

        self.featurizer.train()
        self.sigma_q_opt.requires_grad = True
        self.sigma_phi_opt.requires_grad = True
        self.epsilon_opt.requires_grad = True

        features = torch.cat([X, Y], 0)

        # ------------------------------
        #  Train deep network for MMD-D
        # ------------------------------
        # Initialize optimizer
        self.optimizer_F.zero_grad()
        # Compute output of deep network
        model_output = self.featurizer(features)
        # Compute epsilon, sigma_q and sigma_phi
        ep = torch.exp(self.epsilon_opt) / (1 + torch.exp(self.epsilon_opt))
        sigma_q = self.sigma_q_opt**2
        sigma_phi = self.sigma_phi_opt**2
        # Compute Compute J (STAT_u)
        mmd_value_estimate, mmd_var_estimate = self.MMDu(
            features=model_output,
            len_s=X.shape[0],
            features_org=features.view(features.shape[0], -1),
            sigma_q=sigma_q,
            sigma_phi=sigma_phi,
            epsilon=ep,
            is_var_computed=True,
        )
        if mmd_var_estimate is None:
            raise AssertionError("Error: Variance of MMD is not computed. Please set is_var_computed=True.")
        mmd_std_estimate = torch.sqrt(mmd_var_estimate)
        stat_u = torch.div(-1 * mmd_value_estimate, mmd_std_estimate)
        # Compute gradient
        stat_u.backward()
        # Update weights using gradient descent
        self.optimizer_F.step()

    def compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the Deep MMD Loss."""

        self.featurizer.eval()
        self.sigma_q_opt.requires_grad = False
        self.sigma_phi_opt.requires_grad = False
        self.epsilon_opt.requires_grad = False

        features = torch.cat([X, Y], 0)

        # Compute output of deep network
        model_output = self.featurizer(features)
        # Compute epsilon, sigma_q and sigma_phi
        ep = torch.exp(self.epsilon_opt) / (1 + torch.exp(self.epsilon_opt))
        sigma_q = self.sigma_opt**2
        sigma_phi = self.sigma_phi_opt**2
        # Compute Compute J (STAT_u)
        mmd_value_estimate, _ = self.MMDu(
            features=model_output,
            len_s=X.shape[0],
            features_org=features.view(features.shape[0], -1),
            sigma_q=sigma_q,
            sigma_phi=sigma_phi,
            epsilon=ep,
            is_var_computed=False,
        )

        return mmd_value_estimate

    def forward(self, Xs: torch.Tensor, Xt: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.train_kernel(Xs.clone().detach(), Xt.clone().detach())

        return self.compute_kernel(Xs, Xt)
