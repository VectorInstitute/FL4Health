# How Flower Works

Flower's `Server` class has a `fit()` method that coordinates $N$ rounds of federated learning.
Each round is executed by the `fit_round()` method, which works in a three step process 

1. set up configuration to call clients via `Strategy.config_fit()`

2. broadcasts configurations to clients for local training and receive response via `fit_clients`

3. construct global model from client updates via `Strategy.aggregate_fit()` 

```py title="server.py" linenums="1"
class Server:
    def fit():
        # roughly speaking this is how the server coordinates training
        for i in range(num_rounds):
            self.parameters = self.fit_round()

```

The counter tracking the federated round is called `server_round` and is a parameter to key methods of the `Strategy` class such as `Strategy.configure_fit()` which is assigns each client 
what updates will be sent to them, and `Strategy.aggregate_fit()` constructs new global model parameters.

The server-to-client messages is stored as a list variable 

```
client_instructions = Strategy.configure_fit()
```

which each list item is a tuple `(client_proxy, ins)`. The `Server.fit_clients()` method calls the client side function `Client.fit()` for each client identified by `client_proxy`, passing to each client the packet of instruction data `ins`.

The instructor data `ins` passed to the client by the server has type `FitIns`, defined as

```py title="flwr/common/typing.py" linenums="1"
@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]
```

where `FitIns.parameters` refers to the model parameters and has type

```py title="flwr/common/typing.py" linenums="1"
@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str
```

and `FitIns.config` is a dictionary that can be customized to pass other parameters to the client besides the model parameters.

Analogously server-to-client communication through `FitIns` the client-to-server communication 
is packaged with the data type `FitRes` defined as

```py title="flwr/common/typing.py" linenums="1"
@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: Dict[str, Scalar]
```

The limitation is that the `FitRes` response is only returned by client side `Client.fit()` method. It turns out that each client side function is associated with unique `*Ins` and `*Res` types (see `flwr/common/typing.py`). For example what is useful for client-server communication is the `Client.get_property()` method, whose input and reponse type are 

```py title="flwr/common/typing.py" linenums="1"
@dataclass
class GetPropertiesIns:
    """Properties request for a client."""

    config: Config


@dataclass
class GetPropertiesRes:
    """Properties response from a client."""

    status: Status
    properties: Properties
```

Here the types `Config` and `Properties` are both `Dict[str, Scalar]` with `Scalar = Union[bool, bytes, float, int, str]`. The `Status` code is defined 

```py title="flwr/common/typing.py" linenums="1"
@dataclass
class Status:
    """Client status."""

    code: Code
    message: str
    
class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4
```