from fl4health.clients.adaptive_drift_constraint_client import AdaptiveDriftConstraintClient


class FedProxClient(AdaptiveDriftConstraintClient):
    """
    This client implements the FedProx algorithm from Federated Optimization in Heterogeneous Networks. The idea is
    fairly straightforward. The local loss for each client is augmented with a norm on the difference between the
    local client weights during training (w) and the initial globally shared weights (w^t).

    NOTE: The initial value for mu (the drift penalty weight) is set on the server side and passed to each client
    through parameter exchange. It is stored as the more generally named drift_penalty_weight.
    """

    def update_before_train(self, current_server_round: int) -> None:
        # Saving the initial weights and detaching them so that we don't compute gradients with respect to the
        # tensors. These are used to form the FedProx loss.
        self.drift_penalty_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.model.parameters()
        ]

        return super().update_before_train(current_server_round)
