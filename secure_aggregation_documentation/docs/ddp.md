# Distributed Differential Privacy 

### Secure Aggregation Client 
```py
# Defined in fl4health/clients/secure_aggregation_client.py
class SecureAggregationClient
```

1. Supports the secure aggregation (SecAgg) protocol without dropout (dropout case is 90% implemented).
2. Uses Opacus to support instance level DP. See `setup_opacus()` and `fit()` methods.
3. Save metrics to json file during FL.
4. Supports distributed discrete Gaussian (DDG) mechanism.
5. Support various post processing such as discretization, fast Welsh-Hadamard transform, rounding which have been accelerated.  See `process_model_post_training()` and `secure_and_privatize()`.
5. Support mini-client (currently saves model for each miniclient, this is inefficient and needs to be improved to be scalable).

Please note the computation of l1, l2, and $\ell_\infty$ errors still needs to be updated after the mini-client approach is implemented. Make sure this is correctly implemented before use. Secure Aggregation Server code should also be adjusted for calculating errors in the mini-client case. 

```py
    def secure_and_privatize(self) -> NDArrays:
        vector = self.process_model_post_training(mini_client_id=1)
        for id in range(2, 1+self.num_mini_clients):
            vector += self.process_model_post_training(mini_client_id=id)
            # DEBUG
            log(INFO, f'Arithmetic modulus = {self.crypto.arithmetic_modulus}')
            vector %= self.crypto.arithmetic_modulus

        mask = torch.tensor(self.crypto.get_pair_mask_sum(vector_dim=self.padded_model_dim, allow_dropout=False))
        # self.echo('mask', mask)
        # vector *= 0
        vector += mask
        vector %= self.crypto.arithmetic_modulus

        vector_np = vector.cpu().numpy()

        # NOTE this delta is not the model delta for mini-clients
        # find actual average delta or max delta here and return from method
        delta = vector_np

        return [vector_np, delta, vectorize_model(self.model).cpu().numpy()]
```

### Secure Aggregation Server 
```py
# Defined in fl4health/server/secure_aggregation_server.py
class SecureAggregationServer
```

1. The `fit()` function governs FL.
2. The `secure_aggregation()` method encapsulates server SecAgg and DDP post processing.
3. Server saves metrics to JSON.


### Secure Aggregation Strategy

Aggregation on the server is first processed here onced received from clients.

```py
# fl4health/strategies/secure_aggregation_strategy.py
class SecureAggregationStrategy
```

### Secure Aggregation Exchanger

Back in the days, post processing was done in the exchanger. Some legacy code is still there, might be of interest for future work. 

```py
# fl4health/parameter_exchange/secure_aggregation_exchanger.py
class SecureAggregationExchanger
```