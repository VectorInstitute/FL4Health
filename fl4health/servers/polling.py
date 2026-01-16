import concurrent.futures

from flwr.common.typing import Code, GetPropertiesIns, GetPropertiesRes
from flwr.server.client_proxy import ClientProxy


PollResultsAndFailures = tuple[
    list[tuple[ClientProxy, GetPropertiesRes]],
    list[tuple[ClientProxy, GetPropertiesRes] | BaseException],
]


def _handle_finished_future_after_poll(
    future: concurrent.futures.Future,
    results: list[tuple[ClientProxy, GetPropertiesRes]],
    failures: list[tuple[ClientProxy, GetPropertiesRes] | BaseException],
) -> None:
    """
    Convert finished future into either a result or a failure for polling.

    Args:
        future (concurrent.futures.Future): The future returned by a client executing polling. It is either added
            to results if there are no exceptions or failures if there are any.
        results (list[tuple[ClientProxy, GetPropertiesRes]]): Set of good results from clients that have accumulated.
        failures (list[tuple[ClientProxy, GetPropertiesRes] | BaseException]): The set of failing results that
            have accumulated for the polling.
    """
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, GetPropertiesRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def poll_client(client: ClientProxy, ins: GetPropertiesIns) -> tuple[ClientProxy, GetPropertiesRes]:
    """
    Get Properties of client. This is run for each client to extract the properties from the target client.

    Args:
        client (ClientProxy): Client proxy representing one of the clients managed by the server.
        ins (GetPropertiesIns): ins provides any configurations required to help the client retrieve the correct
            properties.

    Returns:
        (tuple[ClientProxy, GetPropertiesRes]): Returns the resulting properties from the client response.
    """
    property_res: GetPropertiesRes = client.get_properties(ins=ins, timeout=None, group_id=None)
    return client, property_res


def poll_clients(
    client_instructions: list[tuple[ClientProxy, GetPropertiesIns]],
    max_workers: int | None,
    timeout: float | None,
) -> PollResultsAndFailures:
    """
    Poll clients concurrently on all selected clients.

    Args:
        client_instructions (list[tuple[ClientProxy, GetPropertiesIns]]): This is the set of instructions for the
            polling to be passed to each client. Each client is represented by a single ``ClientProxy`` in the list.
        max_workers (int | None): This is the maximum number of concurrent workers to be used by the server to
            poll the clients. This should be set if pooling an extremely large number, if none a maximum of 32 workers
            are used.
        timeout (float | None): How long the executor should wait to receive a response before moving on.

    Returns:
        (PollResultsAndFailures): Object holding the results and failures associate with the concurrent polling.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(poll_client, client_proxy, property_ins)
            for client_proxy, property_ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, GetPropertiesRes]] = []
    failures: list[tuple[ClientProxy, GetPropertiesRes] | BaseException] = []
    for future in finished_fs:
        _handle_finished_future_after_poll(future=future, results=results, failures=failures)

    return results, failures
