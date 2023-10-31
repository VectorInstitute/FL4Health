# Modify Client Parameter

In the `SecureAggregationClient` we overwrite the `fit()` method in order to expose the parameters.
We can access the updated model parameters (i.e. after training an epoch or one step of gradient descent) via `get_parameters()` which in turn obtains parameters through the `parameter_exchanger` object on the client side. The function of the parameter exchanger is to update the local model with the global model after federated averaging, and to seriaize local models to be sent to the server. These parameter exchanging operations are called pull and push, respectively.

`SecureAggregationClient` inherits from FL4Health `BasicClient`, which in turn sets the parameter exchanger like this

```py title="fl4health/clients/basic_client.py" linenums="1"
class BasicClient(...):
    self.parameter_exchanger = self.get_parameter_exchanger(config)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
    return FullParameterExchanger()
```

By default we are getting the FullParameterExchanger, and for us to customize `SecureAggregationClient` to add mask and noise, we must create a new parameter exchanger, extending 
`FullParameterExchanger`, that supports parameter post (training) processing.

