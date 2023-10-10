# Privacy Mechanisms

!!! info ""

    A **privacy mechanism** is a randomized algorithm which takes as input a user's data, and through injecting noise to the data, the algorithm produces a privitized version of the users data as ouput, which can subsequently be shared in the clear for tasks such as federated learning.
    
The following discrete privacy mechanisms have been implemented to privitize clients' discrete vectors for the cryptographic primitive called secure aggregation (SecAgg) and its variations.

1. Discrete Gaussian mechanism
2. Poisson binomial mechanism
3. Skellam mechanism
4. Binomial mechanism

## 1. Discrete Gaussian Mechanism

This is an additive mechanism that works on discrete user data $x$ defined by $\mapsto x + Z$ where $Z\sim \mathcal{N}_{\mathbb{Z}}(\mu, \sigma ^2)$ is a random variable that takes values in $\mathbb{Z}$ with the following probability mass function

$$\mathbb{P}[Z = n] = \frac{e^{ - \frac{(n-\mu)^2}{2\sigma ^2}}}{\sum_{m\in\mathbb{Z}} e^{ - \frac{(m-\mu)^2}{2\sigma ^ 2}}} \  $$

for each $n \in \mathbb{Z}$. We used **rejection sampling** for drawing samples from the discrete Gaussian distribution. The mechanism is exposed through the following function 

``` py title="dpDiscreteGaussian.py" linenums="1"
def DiscreteGaussianMechanism(query_vector: List[int], 
                                variance: float) -> List[int]:

```

## 2. Poisson Binomial Mechanism


!!! info inline end "Note"

    This is the only non-additive privacy mechanism implemented. It gives an unbiased estimator of the mean. 

In the scalar version of this mechanism, a real valued input $x$ is assoicated with a 
binomial random variable $$Z\sim \text{Binom}(m, p(x))$$ where $m$ is the modulus for moular arithmetic used in SecAgg and the success probability $p(x)$ encodes user data $x \in \mathbb{R}$.

The estimator of the mean  
$$\hat{\mu} = c_1\sum_i{Z}_i + c_2$$
turns out to be unbiased. Parameters $c_1$, $c_2$ depend on various parameters of the algorithm, and the summation is taken over the clients.

``` py title="dpPoissonBinomial.py" linenums="1"
# server side function for Kashin-frame generation 
def Kashin_alternative(dim: int) -> List[List[int]]:

# server side function for aggregation
def PoissonBinomialMechanism_Server(inverse_matrix: List[List[int]],
                                    radius: float,
                                    modulus: int,
                                    theta: float,
                                    *client_vectors: List[int]) -> List[int]:

# client side function 
def PoissonBinomialMechanism_Client(query_vector: List[float], 
                             radius: float, 
                             tight_frame: List[float], 
                             theta: float, 
                             K: float, 
                             modulus: int) -> List[int]:

```

##  3. Skellam Mechanism

Additive mechanism $x\mapsto x + Z$ where the Skellam random variable $Z = P_1 - P_2$ is a
difference of two independent Poisson random variables.

The entry point to the mechanism is 

``` py title="dpSkellam.py" linenums="1"
def SkellamMechanism(query_vector: list[int], 
                     skellam_variance: float) -> list[int]:
```

## 4. Binomial Mechanism 

Additive mechanism $x\mapsto x + (Z - Np)\cdot s$ where $Z\sim \text{Binom}(N, p)$ and $s=\frac{1}{j}$ for some $j\in\mathbb{N}$ is the quantization scale.

``` py title="dpBinomial.py" linenums="1"
def BinomialMechanism(query_vector: List[int], 
                      N: int, p: float, j: int) -> List[int]:
```

## TODO

- [ ] Include references to this documentation.
- [x] Debugging 
- [x] Create documentation  
- [x] Do assertions on input range and possible zero division errors.




