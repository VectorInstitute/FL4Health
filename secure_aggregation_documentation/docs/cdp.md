# Central Differential Privacy 

### Central DP Client 

```py
# fl4health/clients/central_dp_client.py
class CentralDPClient
```

1. Supports instance level DP
2. Client side clipping (later server side noising)
3. We hard coded several privacy parameters for now. This need to be automatically passed in the future.

```py
def setup_opacus(self) -> None:
    privacy_engine = PrivacyEngine()

    # NOTE hard coded in for now
    self.noise_multiplier = 1e-16
    self.clipping_bound = 1e16

    self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
        module=self.model,
        optimizer=self.optimizer,
        data_loader=self.train_loader,
        noise_multiplier=self.noise_multiplier,
        max_grad_norm=self.clipping_bound,
        clipping="flat",
        poisson_sampling=False # keep this False unless specifically requested to be True
    )
```

```py
# fl4health/server/central_dp_server.py
class CentralDPServer
```

1. Metrics are recorded in `fit()` in the block ` with open(self.metrics_path, 'r') as file:`
2. Server adds central noise (continuous noise) to client clipped updates.