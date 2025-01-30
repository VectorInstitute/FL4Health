from flwr.common.typing import Config, GetPropertiesIns
from flwr.server.client_proxy import ClientProxy

from fl4health.servers.polling import poll_client, poll_clients
from tests.test_utils.custom_client_proxy import CustomClientProxy


def test_poll_client() -> None:
    client_proxy = CustomClientProxy(cid="c0", num_samples=10)
    config: Config = {"test": 0}

    ins = GetPropertiesIns(config=config)
    _, res = poll_client(client_proxy, ins)

    assert res.properties["num_samples"] == 10


def test_poll_clients() -> None:
    client_ids = [f"c{i}" for i in range(10)]
    sample_counts = [11 for _ in range(10)]
    clients = [CustomClientProxy(cid, count) for cid, count in zip(client_ids, sample_counts)]
    config: Config = {"test": 0}
    ins = GetPropertiesIns(config=config)
    clients_instructions: list[tuple[ClientProxy, GetPropertiesIns]] = [(client, ins) for client in clients]

    results, _ = poll_clients(client_instructions=clients_instructions, max_workers=None, timeout=None)

    property_results = [result[1] for result in results]

    for res in property_results:
        assert res.properties["num_samples"] == 11
