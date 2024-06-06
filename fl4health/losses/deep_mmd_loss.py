from typing import Optional, Tuple

import numpy as np
import torch


class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""

    def __init__(self, x_in: int, H: int, x_out: int):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant


class DeepMmdLoss(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_size: int,
        lr: float = 0.001,
        hidden_size: int = 10,
        output_size: int = 50,
        layer_name: Optional[str] = None,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.lr = lr
        self.layer_name = layer_name
        self.training = training

        # Initialize the model
        self.featurizer = ModelLatentF(input_size, hidden_size, output_size).to(self.device)

        # Initialize parameters
        self.epsilonOPT: torch.Tensor = torch.log(torch.from_numpy(np.random.rand(1) * 10 ** (-10)).to(self.device))
        self.epsilonOPT.requires_grad = self.training
        self.sigmaOPT: torch.Tensor = torch.sqrt(torch.tensor(2 * 32 * 32, dtype=torch.float).to(self.device))
        self.sigmaOPT.requires_grad = self.training
        self.sigma0OPT: torch.Tensor = torch.sqrt(torch.tensor(0.005, dtype=torch.float).to(self.device))
        self.sigma0OPT.requires_grad = self.training

        # Initialize optimizers
        self.optimizer_F = torch.optim.Adam(
            list(self.featurizer.parameters()) + [self.epsilonOPT] + [self.sigmaOPT] + [self.sigma0OPT], lr=self.lr
        )

    def Pdist2(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """compute the paired distance between x and y."""
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)
        Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        Pdist[Pdist < 0] = 0
        return Pdist

    def h1_mean_var_gram(
        self, Kx: torch.Tensor, Ky: torch.Tensor, Kxy: torch.Tensor, is_var_computed: bool, use_1sample_U: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """compute value of MMD and std of MMD using kernel matrix."""
        Kxxy = torch.cat((Kx, Kxy), 1)
        Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
        Kxyxy = torch.cat((Kxxy, Kyxy), 0)
        nx = Kx.shape[0]
        ny = Ky.shape[0]
        is_unbiased = True
        if is_unbiased:
            xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
            yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
            # one-sample U-statistic.
            if use_1sample_U:
                xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
            else:
                xy = torch.div(torch.sum(Kxy), (nx * ny))
            mmd2 = xx - 2 * xy + yy
        else:
            xx = torch.div((torch.sum(Kx)), (nx * nx))
            yy = torch.div((torch.sum(Ky)), (ny * ny))
            # one-sample U-statistic.
            if use_1sample_U:
                xy = torch.div((torch.sum(Kxy)), (nx * ny))
            else:
                xy = torch.div(torch.sum(Kxy), (nx * ny))
            mmd2 = xx - 2 * xy + yy
        if not is_var_computed:
            return mmd2, None, Kxyxy
        hh = Kx + Ky - Kxy - Kxy.transpose(0, 1)
        V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
        V2 = (hh).sum() / (nx) / nx
        varEst = 4 * (V1 - V2**2)
        return mmd2, varEst, Kxyxy

    def MMDu(
        self,
        Fea: torch.Tensor,
        len_s: int,
        Fea_org: torch.Tensor,
        sigma: torch.Tensor,
        sigma0: torch.Tensor,
        epsilon: torch.Tensor,
        is_smooth: bool = True,
        is_var_computed: bool = True,
        use_1sample_U: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
        X = Fea[0:len_s, :]  # fetch the sample 1 (features of deep networks)
        Y = Fea[len_s:, :]  # fetch the sample 2 (features of deep networks)
        X_org = Fea_org[0:len_s, :]  # fetch the original sample 1
        Y_org = Fea_org[len_s:, :]  # fetch the original sample 2
        L = 1  # generalized Gaussian (if L>1)
        Dxx = self.Pdist2(X, X)
        Dyy = self.Pdist2(Y, Y)
        Dxy = self.Pdist2(X, Y)
        Dxx_org = self.Pdist2(X_org, X_org)
        Dyy_org = self.Pdist2(Y_org, Y_org)
        Dxy_org = self.Pdist2(X_org, Y_org)
        if is_smooth:
            Kx = (1 - epsilon) * torch.exp(-((Dxx / sigma0) ** L) - Dxx_org / sigma) + epsilon * torch.exp(
                -Dxx_org / sigma
            )
            Ky = (1 - epsilon) * torch.exp(-((Dyy / sigma0) ** L) - Dyy_org / sigma) + epsilon * torch.exp(
                -Dyy_org / sigma
            )
            Kxy = (1 - epsilon) * torch.exp(-((Dxy / sigma0) ** L) - Dxy_org / sigma) + epsilon * torch.exp(
                -Dxy_org / sigma
            )
        else:
            Kx = torch.exp(-Dxx / sigma0)
            Ky = torch.exp(-Dyy / sigma0)
            Kxy = torch.exp(-Dxy / sigma0)

        return self.h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

    def train_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Train the kernel."""

        self.featurizer.train()
        self.sigmaOPT.requires_grad = True
        self.sigma0OPT.requires_grad = True
        self.epsilonOPT.requires_grad = True

        features = torch.cat([X, Y], 0)

        # ------------------------------
        #  Train deep network for MMD-D
        # ------------------------------
        # Initialize optimizer
        self.optimizer_F.zero_grad()
        # Compute output of deep network
        model_output = self.featurizer(features)
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
        sigma = self.sigmaOPT**2
        sigma0_u = self.sigma0OPT**2
        # Compute Compute J (STAT_u)
        mmd_value_temp, mmd_var_temp, _ = self.MMDu(
            Fea=model_output,
            len_s=X.shape[0],
            Fea_org=features.view(features.shape[0], -1),
            sigma=sigma,
            sigma0=sigma0_u,
            epsilon=ep,
        )
        if mmd_var_temp is None:
            raise AssertionError("Error: Variance of MMD is not computed. Please set is_var_computed=True.")
        mmd_std_temp = torch.sqrt(mmd_var_temp + 10 ** (-8))
        STAT_u = torch.div(-1 * mmd_value_temp, mmd_std_temp)
        # Compute gradient
        STAT_u.backward()
        # Update weights using gradient descent
        self.optimizer_F.step()
        return

    def compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Train the kernel."""

        self.featurizer.eval()
        self.sigmaOPT.requires_grad = False
        self.sigma0OPT.requires_grad = False
        self.epsilonOPT.requires_grad = False

        features = torch.cat([X, Y], 0)

        # Compute output of deep network
        model_output = self.featurizer(features)
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
        sigma = self.sigmaOPT**2
        sigma0_u = self.sigma0OPT**2
        # Compute Compute J (STAT_u)
        mmd_value_temp, _, _ = self.MMDu(
            Fea=model_output,
            len_s=X.shape[0],
            Fea_org=features.view(features.shape[0], -1),
            sigma=sigma,
            sigma0=sigma0_u,
            epsilon=ep,
        )

        return mmd_value_temp

    def forward(self, Xs: torch.Tensor, Xt: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.train_kernel(Xs.clone().detach(), Xt.clone().detach())

        return self.compute_kernel(Xs, Xt)
