# Secure Aggregation 

The main SecAgg cryptographic procedures such as key exchange and agreement, is handled by 
`ClientCryptoKit` and `ServerCryptoKit` found in 
```
fl4health/security/secure_aggregation.py
```
This file contains a few tests that remains to be converted into actual unit tests.

# Secure Aggregation (old)

Implementation based on [Practical Secure Aggregation for Privacy-Preserving Machine Learning (SecAgg)](https://dl.acm.org/doi/10.1145/3133956.3133982){:target="_blank"}.
Below is a matching of the detailed protocol on Page 1181 and its implementation in `FL4Health`.

## Setup

The `SecureAggregationServer` receives

1. Integer $t\geq 2$ Shamir threshold ([Section 3.1](https://dl.acm.org/doi/pdf/10.1145/3133956.3133982){:target="_blank"} page 1176)
2. Integer $R_U > 1$ such that each model parameter lies in the interval $[0, R_U-1]$ assuming model parameters have been discretized into integers ([Section 7.1](https://dl.acm.org/doi/pdf/10.1145/3133956.3133982){:target="_blank"} page 1185)

These are set in `config.yaml`, and can be updated with `SecureAggregationServer` methods `set_shamir_threshold()` and `set_model_integer_range()` which offer type checking and logging.

All other parameters in the `Setup` phase of SecAgg described in the paper are implicitly determined in our implementation:

| Parameter   | Name                  | How it is determined               |
| ----------- | -----------------     | ---------------------------------- |
| $m$         | model dimension       | from model                         |
| $n$         | client count          | from client manager                |
| $R$         | modulus of arithmetic | $R = n(R_U-1) + 1$                 |

As clients may dropout, the modulus $R$ is updated at the begining of each federated round.

## Round 0 (Advertise Keys)

### Server
The server begins by communicating to each client (Alice) the number of online peers `number_of_bobs` they have on this round of
SecAgg.

The client also receives the modulus of arithmetic `arithmetic_modulus` which depends on `number_of_bobs` and has been calculated from the **Setup** stage before.

Currently we do not adjust the Shamir reconstruction threshold based on dropout severeness. The client integers we assign as ID to each client also persist across all FL rounds. Future work may explore adjusting these parameters in an adptive fashion before the `advertise_keys()` method of `SecureAggregationServer`, and these updated versions will be automatically communicated to clients without needing to adjust the `advertise_keys()`.

For each client, the server receives two public keys: one for encryption, the other for masking. The server then registers these with the `ServerCryptoKit` to be broadcasted to clients in the next stage.

Server records all dropouts during this stage and verifies the number of online peers remains above Shamir threshold.

### Client

The `SecureAggregationClient` handles server SecAgg instructions via the `get_properties()` method, which is a name standarized in `fl4health/server/polling.py`
as **the** method on the client side which the server calls for arbitrary communication. If need arises in the future for a client to initiate arbitrary communication with the server, the `polling.py` may be extended accordingly.

For each SecAgg stage (Setup, AdvertiseKey, ShareKeys, MaskedInputCollection, Unmasking) the
`get_properties()` method matches with a unique identifier for the stage under the `Event` enum defined in the `secure_aggregation.py` module, leading to execution of the corresponding stage of SecAgg.

## Round 1 (Share Keys)

### Server
Broadcasts public keys for self-masking and pairwise-masking to all online clients (online according to the `ClientManager`). Upon receiving Shamir shares in step 5 of the Client section below, the server forwards these shares to clients.

!!! Note

    Server verifies if there are too many dropouts each time it communicates with the clients.

### Client
1. Upon receiving the public key broadcast, each client performs Diffie-Hellman key agreement with each of their peer clients, using  `ClientCryptoKit`.
2. Each online client generates self-mask seed.
3. Each client computes Shamir secret shares of their self-mask seed as well as their pair-mask secret.
4. Each secret share is assigned to a receiver client and is encrypted with the shared key generated during Diffie-Hellman key agreement above.
5. Clients return these encrypted shares to the server.


## Round 2 (Masked Input Collection)

### Client
The `ClientCryptoKit` method `get_duo_mask()` computes the sum of pair-masks and self-mask. Use `get_self_mask()` to get self-mask and
`get_pair_mask_sum()` to get pair-mask.

!!! Note
    We do not return the masked input vector from `get_properties()`, but rather from client `fit()` function. This is because we need to post process model parameters and inject privacy noise.

When client calls `fit()` it receives the global model, and preceeds with local training. Then 
the model is post processed with `process_model_post_training()`. Depending on the context, post
processing may entail privacy treatments (dp-SGD clipping or privacy noise injection) or security 
encryption (masking, quantization, transformation). 



## Round 3 (Consistency Check)

!!! warning

    We do not implement the public key infrastructure, see red portion of Page 1181. It is designed to provide security in the active-adversary model, and has stronger assumptions than we work with.

    Consequently in the current implementation, once a client
    drops out, they are lost for the remaining federated rounds: attempting to reconnect a dropped client may open a back door
    for impersonation attacks, currently there is no implemented mechanism to reauthenticate dropped clients.

## Round 4 (Unmasking)
