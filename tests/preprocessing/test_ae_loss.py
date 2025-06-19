import torch
from sklearn.metrics import mean_squared_error

from fl4health.preprocessing.autoencoders.loss import VaeLoss


def kl_divergence_normal(
    q_params: tuple[torch.Tensor, torch.Tensor], p_params: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    mu_q, logvar_q = q_params
    mu_p, logvar_p = p_params
    # Create Normal distributions using the parameters
    q_distribution = torch.distributions.Normal(mu_q, logvar_q.exp().sqrt())
    p_distribution = torch.distributions.Normal(mu_p, logvar_p.exp().sqrt())

    # Calculate the KL divergence
    return torch.distributions.kl.kl_divergence(q_distribution, p_distribution).sum()


def test_analytical_kl_divergence_loss() -> None:
    base_loss = torch.nn.MSELoss(reduction="sum")
    vae_loss = VaeLoss(latent_dim=10, base_loss=base_loss)
    mu = torch.randn(10)
    logvar = torch.randn(10)

    mu_p, logvar_p = torch.zeros_like(mu), torch.zeros_like(logvar)

    # Calculate the KL divergence using the input tensors
    # expected_kl_loss = kl_div(torch.log_softmax(q_sample, dim=-1), p_sample, reduction='sum')
    kl_divergence_value = kl_divergence_normal((mu, logvar), (mu_p, logvar_p))

    # Calculate the KL divergence using the VaeLoss method
    calculated_kl_loss = vae_loss.standard_normal_kl_divergence_loss(mu, logvar)

    # Check if the calculated KL divergence is close to the expected value
    tolerance = 1e-5
    assert torch.abs(calculated_kl_loss - kl_divergence_value).item() < tolerance


def test_unpack_model_output() -> None:
    base_loss = torch.nn.MSELoss(reduction="sum")
    vae_loss = VaeLoss(latent_dim=5, base_loss=base_loss)

    # Create dummy input 3D tensor with dummy mu and logvar
    batch_size = 5
    data_dim1 = 3
    data_dim2 = 10
    data_dim3 = 10
    preds = torch.randn(batch_size, data_dim1, data_dim2, data_dim3)
    mu = torch.randn(batch_size, vae_loss.latent_dim)
    logvar = torch.randn(batch_size, vae_loss.latent_dim)

    # Packing variational autoencoder outputs as it is done in fl4health.model_bases.autoencoder_base
    flattened_output = preds.view(preds.shape[0], -1)
    # Flattened_output should be 2D.
    assert flattened_output.dim() == 2
    packed_tensor = torch.cat((logvar, mu, flattened_output), dim=1)

    # Test the unpack_model_output function
    unpacked_preds, unpacked_mu, unpacked_logvar = vae_loss.unpack_model_output(packed_tensor)

    # Check if the shapes of unpacked tensors are correct
    assert unpacked_preds.shape == (batch_size, data_dim1 * data_dim2 * data_dim3)
    assert unpacked_mu.shape == unpacked_logvar.shape == (batch_size, vae_loss.latent_dim)


def test_custom_base_loss() -> None:
    vae_loss = VaeLoss(latent_dim=5, base_loss=torch.nn.MSELoss())
    preds1 = torch.tensor([[1, -2, 3], [4, -5, 6], [7, -8, 9]]).float()
    preds2 = torch.tensor([[10, -12, 2], [-3, 15, -6], [-4, 8, -3]]).float()
    target = torch.tensor([[10, -12, 2], [-3, 15, -6], [-4, 8, -3]]).float()

    assert int(vae_loss.base_loss(preds2, target)) == 0
    tolerance = 1e-5
    assert (
        torch.abs(
            vae_loss.base_loss(preds1, target) - torch.tensor(mean_squared_error(preds1.numpy(), target.numpy()))
        ).item()
        < tolerance
    )
