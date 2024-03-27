from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from tests.test_utils.custom_client_proxy import CustomClientProxy


def test_sample() -> None:
    test_n_total_clients = 20
    test_n_sampled_clients = 2
    client_manager = FixedSamplingClientManager()

    for i in range(test_n_total_clients):
        client_manager.register(CustomClientProxy(str(i)))

    test_sample_1 = client_manager.sample(test_n_sampled_clients)
    test_sample_2 = client_manager.sample(test_n_sampled_clients)

    assert test_sample_1 == test_sample_2

    client_manager.reset_sample()
    test_sample_3 = client_manager.sample(test_n_sampled_clients)

    # TODO how to make sure this is not going to fail often by selecting the same sample by accident?

    assert test_sample_3 != test_sample_1
