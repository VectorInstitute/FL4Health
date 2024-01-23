# multi-kernel maximum mean discrepancy
# cao bin, HKUST, China, binjacobcao@gmail.com
# free to charge for academic communication

from typing import List, Optional

import torch
from qpth.qp import QPFunction


class MKMMDLoss(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        gamma_list: List[float] = [
            2,
            1,
            1 / 2,
            1 / 4,
            1 / 8,
        ],
        min_distance: Optional[bool] = False,
    ) -> None:
        """Compute the multi-kernel maximum mean discrepancy (MK-MMD) between the source and target domains.
        Args:

            gamma_list (List[float]): list of length scales for rbf kernels
        """
        super().__init__()

        self.kernel_num = len(gamma_list)
        self.gamma_list = torch.tensor(gamma_list)
        self.device = device
        if min_distance:
            self.sign = -1
        else:
            self.sign = 1

    def rbf_multi_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Compute the Radial Basis Function (RBF) kernel between two sets of data points.

        Args:
            X1 (torch tensor): shape (n_samples1, n_features)
            X2 (torch tensor): shape (n_samples2, n_features)
            gamma (torch tensor): the kernel parameters

        Returns:
            K (torch.tensor): shape (n_samples1, n_samples2), the RBF kernel matrix
        """

        # Calculate the pairwise squared Euclidean distances
        dist_sq = torch.sum(X1**2, dim=1, keepdim=True) + torch.sum(X2**2, dim=1) - 2 * torch.mm(X1, X2.t())

        # Compute the RBF kernel matrix
        k_list = []
        for gamma in self.gamma_list:
            k_list.append((torch.exp((-1 * dist_sq) / (2 * torch.pow(gamma, 2)))).unsqueeze(0))

        return torch.cat(k_list)

    def compute_mmd(
        self,
        XX: torch.Tensor,
        YY: torch.Tensor,
        YX: torch.Tensor,
        XY: torch.Tensor,
    ) -> torch.Tensor:

        return XX.mean(dim=(1, 2)) + YY.mean(dim=(1, 2)) - XY.mean(dim=(1, 2)) - YX.mean(dim=(1, 2))

    def estimate_η_k(
        self,
        XX: torch.Tensor,
        YY: torch.Tensor,
        YX: torch.Tensor,
        XY: torch.Tensor,
    ) -> torch.Tensor:

        η_k_vector: torch.Tensor = torch.zeros((1, self.kernel_num)).to(self.device)
        batch_num = len(XX[0])

        for i in range(0, batch_num, 2):

            η_k_vector += XX[:, i, i + 1] + YY[:, i, i + 1] - XY[:, i, i + 1] - YX[:, i, i + 1]

        return 2 * η_k_vector / batch_num

    def estimate_qk_vector(
        self,
        XX: torch.Tensor,
        YY: torch.Tensor,
        YX: torch.Tensor,
        XY: torch.Tensor,
    ) -> torch.Tensor:

        Q_k_vector: torch.Tensor = torch.zeros((self.kernel_num, self.kernel_num)).to(self.device)
        batch_num = len(XX[0])

        for i in range(0, batch_num, 4):

            h_d = (XX[:, i, i + 1] + YY[:, i, i + 1] - XY[:, i, i + 1] - YX[:, i, i + 1]) - (
                XX[:, i + 2, i + 3] + YY[:, i + 2, i + 3] - XY[:, i + 2, i + 3] - YX[:, i + 2, i + 3]
            )

            for i in range(self.kernel_num):
                for j in range(i + 1):
                    Q_k_vector[i][j] += h_d[i] * h_d[j]
                    Q_k_vector[j][i] += h_d[i] * h_d[j]
        return 4 * Q_k_vector / batch_num

    def forward(self, Xs: torch.Tensor, Xt: torch.Tensor) -> torch.Tensor:
        """Compute the multi-kernel maximum mean discrepancy (MK-MMD) between the source and target domains.

        Args:
            Xs (torch.Tensor): Source domain data, shape (n_samples, n_features)
            Xt (torch.Tensor): Target domain data, shape (n_samples, n_features)

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: MK-MMD, weights of kernels (beta)
        """

        # cal weights for each rbf kernel
        XX = self.rbf_multi_kernel(Xs, Xs)
        YY = self.rbf_multi_kernel(Xt, Xt)
        YX = self.rbf_multi_kernel(Xt, Xs)
        XY = self.rbf_multi_kernel(Xs, Xt)

        # vector η_k, Eq.(2)
        η_p_k = self.estimate_η_k(XX, YY, YX, XY)

        # Eq.(7)
        Q_p_k = self.estimate_qk_vector(XX, YY, YX, XY)

        # solve the standard quadratic programming problem
        # see : https://github.com/Bin-Cao/KMMTransferRegressor/blob/main/KMMTR/KMM.py
        Q = 2 * Q_p_k + 1e-5 * torch.eye(self.kernel_num).to(self.device)  # λm = 1e-5

        # p = -sign*η_k ， maximum η_k * beta in QB
        p = -1 * η_p_k.reshape(-1, 1).float().squeeze(1)

        # sign*beta >= 0
        G = -1 * self.sign * torch.eye(self.kernel_num).to(self.device)

        h = torch.zeros((self.kernel_num, 1)).to(self.device).squeeze(1)

        # the summation of the beta is sign*1, Eq.(3), let's D = 1
        A = torch.ones((1, self.kernel_num)).to(self.device)

        b = self.sign * torch.tensor(1.0).to(self.device)

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
        #   b is ()
        beta = self.sign * QPFunction(verbose=False)(Q, p, G, h, A, b)

        # two rows above section 2.2 Empirical estimate of the MMD, asymptotic distribution, and test
        η_k = self.compute_mmd(XX, YY, YX, XY)

        MK_MMD = torch.matmul(η_k.float(), beta.squeeze().detach())

        return MK_MMD
