import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.utils
from torch.distributions.multivariate_normal import MultivariateNormal

from fl4health.utils.dataset import TensorDataset


class SyntheticFedProxDataset(ABC):
    def __init__(
        self,
        num_clients: int,
        temperature: float = 1.0,
        input_dim: int = 60,
        output_dim: int = 10,
        samples_per_client: int = 1000,
    ) -> None:
        """
        Abstract base class to support synthetic dataset generation in the style of the original FedProx paper.

        Paper link: https://arxiv.org/abs/1812.06127
        Reference code: https://github.com/litian96/FedProx/tree/master/data/synthetic_1_1

        NOTE: In the implementations here, all clients receive the same number of samples. In the original FedProx
        setup, they are sampled using a power law.

        Args:
            num_clients (int): Number of datasets (one per client) to generate
            temperature (float, optional): temperature used for the softmax mapping to labels. Defaults to 1.0.
            input_dim (int, optional): dimension of the input features for the synthetic dataset. Default is as in the
                FedProx paper. Defaults to 60.
            output_dim (int, optional): dimension of the output labels for the synthetic dataset. These are one-hot
                encoding labels. Default is as in the FedProx paper. Defaults to 10.
            samples_per_client (int, optional): Number of samples to generate in each client's dataset.
                Defaults to 1000.
        """
        self.num_clients = num_clients
        self.temperature = temperature
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.samples_per_client = samples_per_client

        # Sigma in the FedProx paper
        self.input_covariance = self.construct_covariance_matrix()

    def construct_covariance_matrix(self) -> torch.Tensor:
        """
        This function generations the covariance matrix used in generating input features. It is fixed across all
        datasets. It is a diagonal matrix with diagonal entries x_{j, j} = j^{-1.2}, where j starts at 1 in this
        notation. The matrix is of dimension input_dim x input_dim

        Returns:
            torch.Tensor: Covariance matrix for generation of input features.
        """
        sigma_diagonal = torch.zeros(self.input_dim)
        for i in range(self.input_dim):
            # indexing in the original implementation starts at 1, so i+1
            sigma_diagonal[i] = math.pow((i + 1), -1.2)
        return torch.diag(sigma_diagonal)

    def map_inputs_to_outputs(self, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        This function maps features x to a label y as done in the original paper. The first stage is the affine
        transformation hat{y} = (1/T)*(Wx + b). Then y = softmax(hat{y}). Sampling from the distribution, we then
        one hot encode the resulting label sample.

        NOTE: This procedure differs slightly from that of the original paper, which simply took a one hot on the
        softmax distribution. The current strategy allows for a bit more label stochasticity.

        Args:
            x (torch.Tensor): The input features to be mapped to output labels. Shape is (dataset size, input_dim)
            W (torch.Tensor): The linear transformation matrix. Shape is (output_dim, input_dim)
            b (torch.Tensor): The bias in the linear transformation. Shape is (output_dim, 1)

        Returns:
            torch.Tensor: The labels associated with each of the inputs. The shape is (dataset size, output_dim)
        """
        raw_y = (torch.matmul(x, W.T) + b.T.repeat(self.samples_per_client, 1)) / self.temperature
        distributions = F.softmax(raw_y, dim=1)
        samples = torch.multinomial(distributions, 1)
        return F.one_hot(samples, num_classes=self.output_dim).squeeze()

    def generate(self) -> List[TensorDataset]:
        """
        Based on the class parameters, generate a list of synthetic TensorDatasets, one for each client.

        Returns:
            List[TensorDataset]: Synthetic datasets for each client.
        """
        client_tensors = self.generate_client_tensors()
        assert (
            len(client_tensors) == self.num_clients
        ), "The tensors returned by generate_client_tensors should have the same length as self.num_clients"
        client_datasets = [TensorDataset(X, Y) for X, Y in client_tensors]
        return client_datasets

    @abstractmethod
    def generate_client_tensors(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Method to be implemented determining how to generate the tensors in the subclasses. Each of the subclasses
        uses the affine mapping, but the parameters for how that affine mapping is setup are different and determined
        in this function.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: input and output tensors for each of the clients.
        """
        pass


class SyntheticNonIidFedProxDataset(SyntheticFedProxDataset):
    def __init__(
        self,
        num_clients: int,
        alpha: float,
        beta: float,
        temperature: float = 1.0,
        input_dim: int = 60,
        output_dim: int = 10,
        samples_per_client: int = 1000,
    ) -> None:
        """
        NON-IID Synthetic dataset generator modeled after the implementation in the original FedProx paper. See Section
        5.1 in the paper link below for additional details. The non-IID generation code is modeled after the code
        housed in the github link below as well.

        Paper link: https://arxiv.org/abs/1812.06127
        Reference code: https://github.com/litian96/FedProx/tree/master/data/synthetic_1_1

        NOTE: This generator ends up with fairly skewed labels in generation. That is, many of the clients will not
        have representations of all the labels. This has been verified as also occurring in the reference code above
        and is not a bug.

        The larger alpha and beta are, the more heterogeneous the clients data is. The larger alpha is, the more
        "different" the affine transformations are from one another. The larger beta is, the larger the variance in the
        centers of the input features.

        Args:
            num_clients (int): Number of datasets (one per client) to generate
            alpha (float): This is the standard deviation for the mean (u_k), drawn from a centered normal
                distribution, which is used to generate the elements of the affine transformation components W, b.
            beta (float): This is the standard deviation for each element of the multidimensional mean (v_k),
                drawn from a centered normal distribution, which is used to generate the elements of the input features
                for x ~ N(B_k, Sigma)
            temperature (float, optional): temperature used for the softmax mapping to labels. Defaults to 1.0.
            input_dim (int, optional): dimension of the input features for the synthetic dataset. Default is as in the
                FedProx paper. Defaults to 60.
            output_dim (int, optional): dimension of the output labels for the synthetic dataset. These are one-hot
                encoding labels. Default is as in the FedProx paper. Defaults to 10.
            samples_per_client (int, optional): Number of samples to generate in each client's dataset.
                Defaults to 1000.
        """
        super().__init__(
            num_clients=num_clients,
            temperature=temperature,
            input_dim=input_dim,
            output_dim=output_dim,
            samples_per_client=samples_per_client,
        )
        self.alpha = alpha
        self.beta = beta

    def get_input_output_tensors(
        self, mu: float, v: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function takes values for the center of elements in the affine transformation elements (mu), the centers
        feature each of the input feature dimensions (v), and the covariance of those features (sigma) and produces
        the input, output tensor pairs with the appropriate dimensions

        Args:
            mu (float): The mean value from which each element of W and b are to be drawn ~ mathcal{N}(mu, 1)
            v (torch.Tensor): This is assumed to be a 1D tensor of size self.input_dim and represents the mean for the
                multivariate normal from which to draw the input x
            sigma (torch.Tensor): This is assumed to be a 2D tensor of shape (input_dim, input_dim) and represents the
                covariance matrix Sigma of the multivariate normal from which to draw the input x. It should be a
                diagonal matrix as well.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: X and Y for the clients synthetic dataset. Shape of X is
                n_samples x input dimension. Shape of Y is n_samples x output_dim and is one-hot encoded
        """

        multivariate_normal = MultivariateNormal(loc=v, covariance_matrix=sigma)
        # size of x should be samples_per_client x input_dim
        x = multivariate_normal.sample(torch.Size((self.samples_per_client,)))

        W = torch.normal(mu, torch.ones((self.output_dim, self.input_dim)))
        b = torch.normal(mu, torch.ones(self.output_dim, 1))

        return x, self.map_inputs_to_outputs(x, W, b)

    def generate_client_tensors(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        For the Non-IID synthetic generator, this function uses the values of alpha and beta to sample the parameters
        that will be used to generate the synthetic datasets on each client. For each client, beta is used to sample
        a mean value from which to generate the input features, alpha is used to sample a mean for the transformation
        components of W and b. Note that sampling occurs for EACH client independently. The larger alpha and beta
        the larger the variance in these values, implying higher probability of heterogeneity.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: Set of input and output tensors for each client.
        """
        tensors_per_client: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.num_clients):
            B = torch.normal(0.0, self.beta, (1,))
            # v_k in the FedProx paper
            input_means = torch.normal(B, torch.ones(self.input_dim))

            # u_k in the FedProx paper
            affine_transform_means = torch.normal(0, self.alpha, (1,)).item()
            client_X, client_Y = self.get_input_output_tensors(
                affine_transform_means, input_means, self.input_covariance
            )
            tensors_per_client.append((client_X, client_Y))
        return tensors_per_client


class SyntheticIidFedProxDataset(SyntheticFedProxDataset):
    def __init__(
        self,
        num_clients: int,
        temperature: float = 1.0,
        input_dim: int = 60,
        output_dim: int = 10,
        samples_per_client: int = 1000,
    ) -> None:
        """
        IID Synthetic dataset generator modeled after the implementation in the original FedProx paper. See Appendix
        C.1 in the paper link below for additional details. The IID generation code is based strictly on the
        description in the appendix for IID dataset generation.

        Paper link: https://arxiv.org/abs/1812.06127

        NOTE: This generator ends up with fairly skewed labels in generation. That is, many of the clients will not
        have representations of all the labels. This has been verified as also occurring in the reference code above
        and is not a bug.

        Args:
            num_clients (int): Number of datasets (one per client) to generate
            temperature (float, optional): temperature used for the softmax mapping to labels. Defaults to 1.0.
            input_dim (int, optional): dimension of the input features for the synthetic dataset. Default is as in the
                FedProx paper. Defaults to 60.
            output_dim (int, optional): dimension of the output labels for the synthetic dataset. These are one-hot
                encoding labels. Default is as in the FedProx paper. Defaults to 10.
            samples_per_client (int, optional): Number of samples to generate in each client's dataset.
                Defaults to 1000.
        """
        super().__init__(
            num_clients=num_clients,
            temperature=temperature,
            input_dim=input_dim,
            output_dim=output_dim,
            samples_per_client=samples_per_client,
        )

        # As described in the original paper, the affine transformation is SHARED by all clients and the elements
        # of W and b are sampled from standard normal distributions.
        self.W = torch.normal(0, torch.ones((self.output_dim, self.input_dim)))
        self.b = torch.normal(0, torch.ones(self.output_dim, 1))
        # Similarly, all input features across clients are all sampled from a centered multidimensional normal
        # distribution with diagonal covariance matrix sigma (see base class for definition).
        self.input_multivariate_normal = MultivariateNormal(
            loc=torch.zeros(self.input_dim), covariance_matrix=self.input_covariance
        )

    def get_input_output_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        As described in the original FedProx paper (Appendix C.1), the features are all sampled from a centered
        multidimensional normal distribution with diagonal covariance matrix shared across clients.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: X and Y for the clients synthetic dataset. Shape of X is
                n_samples x input dimension. Shape of Y is n_samples x output_dim and is one-hot encoded
        """
        # size of x should be samples_per_client x input_dim
        x = self.input_multivariate_normal.sample(torch.Size((self.samples_per_client,)))
        assert x.shape == (self.samples_per_client, self.input_dim)

        return x, self.map_inputs_to_outputs(x, self.W, self.b)

    def generate_client_tensors(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        For IID generation, this function is simple, as we need not sample any parameters per client for use in
        generation, as these are all shared across clients.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: Set of input and output tensors for each client.
        """
        tensors_per_client: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.num_clients):
            client_X, client_Y = self.get_input_output_tensors()
            tensors_per_client.append((client_X, client_Y))
        return tensors_per_client
