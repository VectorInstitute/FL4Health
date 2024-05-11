# Privacy 

### DP Mechanisms 

The experiments in this branch mostly uses the discrete Gaussian mechanism. You can find an accelerated DGauss sampler in 
```py
# fl4health/privacy_mechanisms/discrete_gaussian_mechanism.py
def generate_discrete_gaussian_vector()
```
You will also find here some of the post processing transforms like fast-WHT here.

Some utilities are found in `fl4health/privacy_mechanisms/slow_discrete_gaussian_mechanism.py` as well as `fl4health/security/processing.py` These remain to be migrated to a separate utility and all the files which import this refactored.

See also the `gaussian_mechanism()` in `fl4health/privacy_mechanisms/gaussian_mechanism.py` which is used in central DP experiments.

On the much older branch called `dp-mechanisms` you can find a few more mechanism

1. Skellman mechanism
2. Poisson bionomial mechanism
3. Binomial mechanism 
4. Discrete Gaussian mechanism (slow)

together with their documentation. 

### Privacy Accountant

- For DDGauss

```py
# fl4health/privacy/distributed_discrete_gaussian_accountant.py
class DDGaussAccountant
```
This allows for subsampling amplification. But may run into math errors when the unamplified epsilon is too large.


- For subsampled Gaussian this is taken from TensorFlow code base 
```
fl4health/privacy/subsampled_gaussian_accountant.py
```
