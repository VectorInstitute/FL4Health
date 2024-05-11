# Taken from https://github.com/google-research/federated/tree/master/distributed_dp
# Calculates the gamma (granularity) in DDGauss paper

import math 
import numpy as np
import tensorflow_privacy as tfp

RDP_ORDERS = tuple(range(2, 129)) + (256,)
DIV_EPSILON = 1e-22
from scipy import optimize

from scipy import special

def _compute_delta(orders, rdp, eps):
  """Compute delta given a list of RDP values and target epsilon.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.

  Returns:
    Pair of (delta, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if eps < 0:
    raise ValueError("Value of privacy loss bound epsilon must be >=0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   delta = min( np.exp((rdp_vec - eps) * (orders_vec - 1)) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4):
  logdeltas = []  # work in log space to avoid overflows
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")
    # For small alpha, we are better of with bound via KL divergence:
    # delta <= sqrt(1-exp(-KL)).
    # Take a min of the two bounds.
    logdelta = 0.5 * math.log1p(-math.exp(-r))
    if a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value for alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      rdp_bound = (a - 1) * (r - eps + math.log1p(-1 / a)) - math.log(a)
      logdelta = min(logdelta, rdp_bound)

    logdeltas.append(logdelta)

  idx_opt = np.argmin(logdeltas)
  return min(math.exp(logdeltas[idx_opt]), 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.

  Returns:
    Pair of (eps, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      eps = 0  # No need to try further computation if we have eps = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value of alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      eps = np.inf
    eps_vec.append(eps)

  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Computes delta (or eps) for given eps (or delta) from RDP values.

  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not `None`, the epsilon for which we compute the
      corresponding delta.
    target_delta: If not `None`, the delta for which we compute the
      corresponding epsilon. Exactly one of `target_eps` and `target_delta` must
      be `None`.

  Returns:
    A tuple of epsilon, delta, and the optimal order.

  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order

def compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders):
  """Compute RDP of the Sampled (Distributed) Discrete Gaussian Mechanism.

  See Proposition 14 / Eq. 17 (Page 16) of the main paper.

  Args:
    q: The sampling rate.
    l1_scale: The l1 scale of the discrete Gaussian mechanism (i.e.,
      l1_sensitivity/stddev). For distributed version, stddev is the noise
      stddev after summing all the noise shares.
    l2_scale: The l2 scale of the discrete Gaussian mechanism (i.e.,
      l2_sensitivity/stddev). For distributed version, stddev is the noise
      stddev after summing all the noise shares.
    tau: The inflation parameter due to adding multiple discrete Gaussians. Set
      to zero when analyzing the the discrete Gaussian mechanism. For the
      distributed discrete Gaussian mechanisn, see Theorem 1.
    dim: The dimension of the vector query.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders, must all be greater than 1. If
      provided orders are not integers, they are rounded down to the nearest
      integer.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  orders = [int(order) for order in orders]

  def eps(order):
    assert order > 1, 'alpha must be greater than 1.'
    # Proposition 14 of https://arxiv.org/pdf/2102.06387.pdf.
    term_1 = (order / 2.0) * l2_scale**2 + tau * dim
    term_2 = (order / 2.0) * (l2_scale**2 + 2 * l1_scale * tau + tau**2 * dim)
    term_3 = (order / 2.0) * (l2_scale + np.sqrt(dim) * tau)**2
    return min(term_1, term_2, term_3)

  if q == 1:
    rdp = np.array([eps(order) for order in orders])
  else:
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps), eps(order))
        for order in orders
    ])

  return rdp * steps


def log_comb(n, k):
  gammaln = special.gammaln
  return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def _compute_rdp_subsampled(alpha, gamma, eps, upper_bound=True):
  """Computes RDP with subsampling.

  Reference: http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf.

  Args:
    alpha: The RDP order.
    gamma: The subsampling probability.
    eps: The RDP function taking alpha as input.
    upper_bound: A bool indicating whether to use Theorem 5 of the referenced
      paper above (if set to True) or Theorem 6 (if set to False).

  Returns:
    The RDP with subsampling.
  """
  if isinstance(alpha, float):
    assert alpha.is_integer()
    alpha = int(alpha)
  assert alpha > 1
  assert 0 < gamma <= 1

  if upper_bound:
    a = [0, eps(2)]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1),
         special.comb(alpha, 2) * (gamma**2) * (1 - gamma)**(alpha - 2)]

    for l in range(3, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(3)

  else:
    a = [0]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1)]

    for l in range(2, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(1)

  return special.logsumexp(a=a, b=b) / (alpha - 1)


def compute_rdp_dgaussian_simplified(q, l2_scale, tau, dim, steps, orders):
  """Compute RDP of the Sampled (Distributed) Discrete Gaussian Mechanism."""
  orders = [int(order) for order in orders]

  def eps(order):
    """See Proposition 14 / Eq. 17 (Page 16) of the main paper."""
    assert order >= 1, 'alpha must be greater than or equal to 1.'
    term_1 = order * (l2_scale**2) / 2.0 + tau * dim
    term_2 = (order / 2.0) * (l2_scale + math.sqrt(dim) * tau)**2
    return min(term_1, term_2)

  if q == 1:
    rdp = np.array([eps(order) for order in orders])
  else:
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps), eps(order))
        for order in orders
    ])

  return rdp * steps

def rounded_l1_norm_bound(l2_norm_bound, dim):
  # In general we have L1 <= sqrt(d) * L2. In the scaled and rounded domain
  # where coordinates are integers we also have L1 <= L2^2.
  return l2_norm_bound * min(np.sqrt(dim), l2_norm_bound)

def rounded_l2_norm_bound(l2_norm_bound, beta, dim):
  """Computes the L2 norm bound after stochastic rounding to integers.

  Note that this function is *agnostic* to the actual vector whose coordinates
  are to be rounded, and it does *not* consider the effect of scaling (i.e.
  we assume the input norm is already scaled before rounding).

  See Theorem 1 of https://arxiv.org/pdf/2102.06387.pdf.

  Args:
    l2_norm_bound: The L2 norm (bound) of the vector whose coordinates are to be
      stochastically rounded to the integer grid.
    beta: A float constant in [0, 1). See the initializer docstring of the
      aggregator.
    dim: The dimension of the vector to be rounded.

  Returns:
    The inflated L2 norm bound after stochastic rounding (conditionally
    according to beta).
  """
  assert int(dim) == dim and dim > 0, f'Invalid dimension: {dim}'
  assert 0 <= beta < 1, 'beta must be in the range [0, 1)'
  assert l2_norm_bound > 0, 'Input l2_norm_bound should be positive.'

  bound_1 = l2_norm_bound + np.sqrt(dim)
  if beta == 0:
    return bound_1

  squared_bound_2 = np.square(l2_norm_bound) + 0.25 * dim
  squared_bound_2 += (
      np.sqrt(2.0 * np.log(1.0 / beta)) * (l2_norm_bound + 0.5 * np.sqrt(dim)))
  bound_2 = np.sqrt(squared_bound_2)
  return min(bound_1, bound_2)


def ddgauss_epsilon(gamma,
                    local_stddev,
                    num_clients,
                    l2_sens,
                    beta,
                    dim,
                    q,
                    steps,
                    delta,
                    l1_sens=None,
                    rounding=True,
                    orders=RDP_ORDERS):
  """Computes epsilon of (distributed) discrete Gaussian via RDP."""
  scale = 1.0 / (gamma + DIV_EPSILON)
  l1_sens = l1_sens or (l2_sens * np.sqrt(dim))
  if rounding:
    l2_sens = rounded_l2_norm_bound(l2_sens * scale, beta, dim) / scale
    l1_sens = rounded_l1_norm_bound(l2_sens * scale, dim) / scale

  tau = 0
  for k in range(1, num_clients):
    tau += math.exp(-2 * (math.pi * local_stddev * scale)**2 * (k / (k + 1)))
  tau *= 10

  l1_scale = l1_sens / (np.sqrt(num_clients) * local_stddev)
  l2_scale = l2_sens / (np.sqrt(num_clients) * local_stddev)
  rdp = compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders)
  eps, _, order = get_privacy_spent(orders, rdp, target_delta=delta)
  return eps, order

def ddgauss_local_stddev(q,
                         epsilon,
                         l2_clip_norm,
                         gamma,
                         beta,
                         steps,
                         num_clients,
                         dim,
                         delta,
                         orders=RDP_ORDERS):
  """Selects the local stddev for the distributed discrete Gaussian."""

  def stddev_opt_fn(stddev):
    stddev += DIV_EPSILON
    cur_epsilon, _ = ddgauss_epsilon(
        gamma,
        stddev,
        num_clients,
        l2_clip_norm,
        beta,
        dim,
        q,
        steps,
        delta,
        orders=orders)
    return (epsilon - cur_epsilon)**2

  stddev_result = optimize.minimize_scalar(stddev_opt_fn)
  if stddev_result.success:
    return stddev_result.x
  else:
    return -1

def ddgauss_params(q,
                   epsilon,
                   l2_clip_norm,
                   bits,
                   num_clients,
                   dim,
                   delta,
                   beta,
                   steps,
                   k=4,
                   rho=1,
                   sqrtn_norm_growth=False,
                   orders=RDP_ORDERS):
  """Selects gamma and local noise standard deviation from the given parameters.

  Args:
    q: The sampling factor.
    epsilon: The target overall epsilon.
    l2_clip_norm: The l2 clipping norm for the client vectors.
    bits: The number of bits per coordinate for the aggregated noised vector.
    num_clients: The number of clients per step.
    dim: The dimension of the vector query.
    delta: The target delta.
    beta: The constant in [0, 1) controlling conditional randomized rounding.
      See Proposition 22 of the paper.
    steps: The total number of steps.
    k: The number of standard deviations of the signal to bound (see Thm. 34 /
      Eq. 61 of the paper).
    rho: The flatness parameter of the random rotation (see Lemma 29).
    sqrtn_norm_growth: A bool indicating whether the norm of the sum of the
      vectors grow at a rate of `sqrt(n)` (i.e. norm(sum_i x_i) <= sqrt(n) * c).
      If `False`, we use the upper bound `norm(sum_i x_i) <= n * c`. See also
      Eq. 61 of the paper.
    orders: The RDP orders.

  Returns:
    The selected gamma and the local noise standard deviation.
  """
  n_factor = num_clients**(1 if sqrtn_norm_growth else 2)

  def stddev(x):
    return ddgauss_local_stddev(q, epsilon, l2_clip_norm, x, beta, steps,
                                num_clients, dim, delta, orders)

  def mod_min(x):
    return k * math.sqrt(rho / dim * l2_clip_norm**2 * n_factor +
                         (x**2 / 4.0 + stddev(x)**2) * num_clients)

  def gamma_opt_fn(x):
    return (math.pow(2, bits) - 2 * mod_min(x) / (x + DIV_EPSILON))**2

  gamma_result = optimize.minimize_scalar(gamma_opt_fn)
  if not gamma_result.success:
    raise ValueError('Cannot compute gamma.')

  gamma = gamma_result.x
  # Select the local_stddev that gave the best gamma.
  local_stddev = ddgauss_local_stddev(q, epsilon, l2_clip_norm, gamma, beta,
                                      steps, num_clients, dim, delta, orders)
  return gamma, local_stddev


# gamma, local_stddev = ddgauss_params(
#     q=1,
#     epsilon=1,
#     l2_clip_norm=1,
#     bits=16,
#     num_clients=4,
#     dim=2**10,
#     delta=1/10**6,
#     beta=0.5,
#     steps=100,
#     k=4)

if __name__ == '__main__':
  # # EMNIST
  # print('EMNIST')
  # for eps in [3, 6, 10]:
  #   for k in [1,2,3, 4]:
  #     print('start')

  #     gamma, local_stddev = ddgauss_params(
  #     q=100/3400,
  #     epsilon=eps,
  #     l2_clip_norm=0.03,
  #     bits=16,
  #     num_clients=3400,
  #     dim=2**20,
  #     delta=1/3400,
  #     beta=0.607,
  #     steps=1500,
  #     k=k)
  #     print(eps, '(k, gamma, local_stddev): ', (k, gamma, local_stddev))

  #     print('end')
  # # SO-TP
  # print('SO-TP')
  # for eps in [5, 10, 15, 20]:
  #   for k in [1,2,3, 4]:
  #     gamma, local_stddev = ddgauss_params(
  #     q=60/342477,
  #     epsilon=eps,
  #     l2_clip_norm=2,
  #     bits=16,
  #     num_clients=342477,
  #     dim=2**23,
  #     delta=1/342477,
  #     beta=0.607,
  #     steps=1500,
  #     k=k)
  #     print(eps, '(k, gamma, local_stddev): ', (k, gamma, local_stddev))

  # SP-NWP
  # print('SP-NWP')
  # for eps in [5, 10, 15]:
  #   for k in [1,2,3, 4]:
  #     gamma, local_stddev = ddgauss_params(
  #     q=100/342477,
  #     epsilon=eps,
  #     l2_clip_norm=0.3,
  #     bits=16,
  #     num_clients=342477,
  #     dim=2**22,
  #     delta=1/10**6,
  #     beta=0.607,
  #     steps=1600,
  #     k=k)
  #     print(eps, '(k, gamma, local_stddev): ', (k, gamma, local_stddev))

  # FLamby
  print('FLamby')
  gamma, local_stddev = ddgauss_params(
      q=64/12413,
      epsilon=10,
      l2_clip_norm=2,
      bits=16,
      num_clients=4,
      dim=1_000_000,
      delta=1/10**6,
      beta=0.607,
      steps=100,
      k=4)
  print((gamma, local_stddev))
