# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generally useful small functions."""
import functools
import operator
from typing import Union, Tuple, Optional, Literal, Any, Sequence
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from synjax._src import constants

# pylint: disable=g-long-lambda

Array = jax.Array
Shape = Tuple[int, ...]
EPS = constants.EPS
INF = constants.INF

############################################################################
####  Missing math utils.
############################################################################


def log_comb(n: Array, k: Array) -> Array:
  """Computes a logarithm of combination (n, k)."""
  gammaln = jax.scipy.special.gammaln
  return gammaln(n + 1)-gammaln(k + 1)-gammaln(n+1-k)


def log_catalan(n: Array) -> Array:
  """Computes the log of nth number in Catalan series."""
  return log_comb(2*n, n) - jnp.log1p(n)


def log_delannoy(m: Array, n: Array, *, max_input_value: int) -> Array:
  """Computes a logarithm of a Delannoy number."""
  m = jnp.asarray(m)
  n = jnp.asarray(n)
  k = jnp.arange(max_input_value)
  res = log_comb(m[..., None], k) + log_comb(n[..., None], k) + k * jnp.log(2)
  mask = jnp.arange(max_input_value) <= jnp.minimum(m, n)[..., None]
  return jax.nn.logsumexp(jnp.where(mask, res, -jnp.inf), axis=-1)


############################################################################
####  Functions with gradients that are more numerically stable.
############################################################################


@jax.custom_gradient
def safe_log(x: Array) -> Array:
  x = jnp.asarray(x, dtype=jnp.float32)
  res = jnp.where(x < EPS, -INF, jnp.log(jnp.maximum(x, EPS)))
  # pytype: disable=bad-return-type
  return res, lambda g: g/jnp.maximum(x, EPS)
  # pytype: enable=bad-return-type


# pylint: disable=redefined-builtin
def safe_clip(x: Array, min: float, max: float) -> Array:
  return straight_through_replace(x, jnp.clip(x, min, max))


def safe_log_softmax(
    log_potentials: Array, axis: int|tuple[int, ...] = -1) -> Array:
  return jax.nn.log_softmax(safe_clip(log_potentials, -INF, INF), axis=axis)


InversionMethod = Literal["solve", "qr"]


def inv(x: Array, *, inv_method: Optional[InversionMethod] = None,
        matmul_precision: Optional[jax.lax.Precision] = None,
        test_invertability: bool = False):
  """Matrix inverse with controlable precision and inversion algorithm.

  The default behaviour of this function is the same as jnp.linalg.inv.
  For the definitions of forward and reverse mode autodiff of matrix inverse
  used in this implementation see Giles (2008) Section 2.2.3.
  References:
    Giles, 2008: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf#page=4
  Args:
    x: Squared matrix (possibly batches) that needs to be inverted.
    inv_method: Choise of the algorithm for computing matrix inverse.
    matmul_precision: Precision of matrix multiplication in the backward pass.
    test_invertability: Test if matrix is invertable before inverting it. If it
                        is not invertable, return matrix of zeros.
  Returns:
    Matrix inverse.
  """
  @jax.custom_jvp
  def inv_fn(x):
    if inv_method is None or inv_method == "solve":
      inverse = jnp.linalg.solve(
          x, jnp.broadcast_to(jnp.eye(x.shape[-1]), x.shape))
    elif inv_method == "qr":
      q, r = jnp.linalg.qr(x)
      r_inv = jax.scipy.linalg.solve_triangular(
          r, jnp.broadcast_to(jnp.eye(x.shape[-1]), x.shape), lower=False)
      q_tr = jnp.swapaxes(q, -1, -2)
      inverse = jnp.matmul(r_inv, q_tr, precision=matmul_precision)
    else:
      raise NotImplementedError
    if test_invertability:
      mask = jnp.isfinite(jnp.linalg.slogdet(x)[1])[..., None, None]
      return jnp.where(mask, jnp.nan_to_num(inverse), 0.)
    else:
      return inverse
  @inv_fn.defjvp
  def inv_fn_jvp(primals, tangents):  # pylint: disable=unused-variable
    # Giles (2008) Section 2.2.3 for forward mode autodiff of matrix inverse.
    x = inv_fn(primals[0])
    a = jnp.matmul(x, tangents[0], precision=matmul_precision)
    a = jnp.matmul(a, x, precision=matmul_precision)
    return x, -a
  return inv_fn(x)


def safe_slogdet(
    x: Array, *, logdet_method: Optional[Literal["lu", "qr"]] = None,
    inv_method: Optional[InversionMethod] = None,
    matmul_precision: Optional[Union[str, jax.lax.Precision]] = None,
    test_invertability: bool = True):
  """Signed log determinant with more stable gradients."""
  @jax.custom_jvp
  def slogdet_fn(y):
    sign, log_abs_det = jnp.linalg.slogdet(y, method=logdet_method)
    return sign, jnp.clip(jnp.nan_to_num(log_abs_det, neginf=-INF, posinf=INF),
                          -INF, INF)
  @slogdet_fn.defjvp
  def slogdet_fn_jvp(primals, tangents):  # pylint: disable=unused-variable
    y, = primals
    primals_out = slogdet_fn(y)
    inverse = inv(y, inv_method=inv_method, matmul_precision=matmul_precision,
                  test_invertability=test_invertability)
    tangents_out = (jnp.zeros_like(primals_out[0]),
                    jnp.einsum("...ji,...ij->...", tangents[0], inverse))
    return primals_out, tangents_out
  return slogdet_fn(x)


############################################################################
####  Improved jnp.roll for TPU.
############################################################################


def is_tpu_machine() -> bool:
  return any(dev.platform == "tpu" for dev in jax.devices())


def roll(x: Array, shift: Union[int, Array], axis: int = -1) -> Array:
  if isinstance(shift, Array) and is_tpu_machine():
    # This happens during vmap on TPU where it's better to roll with matmul.
    return _tpu_roll(x, shift, axis)
  else:
    return jnp.roll(x, shift, axis)


def _tpu_roll(x: Array, shift: Union[int, Array], axis: int = -1) -> Array:
  """Significantly faster version of jnp.roll implemented with matmul on TPU."""
  d = x.shape[axis]
  permutation = (jnp.arange(d)-shift)%d
  return _tpu_take(x, permutation, axis)


def _tpu_take(x: Array, indices: Array, axis: int = -1) -> Array:
  if indices.ndim != 1:
    raise ValueError("This function supports only 1-dim indices.")
  dtype = x.dtype
  axis %= x.ndim  # Converts possibly negative axis into a positive one.
  permutation_matrix = jax.nn.one_hot(indices, num_classes=x.shape[axis],
                                      axis=-1)
  x_pre = jnp.moveaxis(x, axis, -2)
  x_post = jnp.matmul(permutation_matrix, x_pre)
  return jnp.moveaxis(x_post, -2, axis).astype(dtype)


############################################################################
####  One-hot vector utils
############################################################################


def max_one_hot(x: Array, axis: Union[int, Tuple[int, ...]] = -1, *,
                straight_through: bool = False) -> Array:
  y = (x == x.max(axis=axis, keepdims=True)).astype(jnp.float32)
  if straight_through:
    y = straight_through_replace(x, y)
  return y


def reinmax_sample(
    log_potential: Array, *, key: Array, temperature: float = 1.0,
    axis: Union[int, Tuple[int, ...]] = -1) -> Array:
  """ReinMax -- discrete sampling that has 2nd order approximation of gradients.

  ReinMax was published in Liu et al (2023) as an alternative to
  Straight-Through and Streaight-Through-Gumbel-Softmax sampling.
  Forward pass will provide a regular one-hot sample, but the backward pass will
  have gradients that are having a higher order approximation than the regular
  Straight-Through estimator.

  Args:
    log_potential: Log-potentials to sample from.
    key: Random key.
    temperature: Temperature for smoothing the gradients. Temperature has the
      opposite effect compared to ST-Gumbel. In ReinMax increasing temperature
      makes gradient estimates more smooth.
    axis: Axis over which to sample.
  Returns:
    Sampled one-hot vector that will have 2nd order approximate gradients.

  References:
    Liu et al 2023
    Bridging Discrete and Backpropagation: Straight-Through and Beyond
    https://openreview.net/pdf?id=mayAyPrhJI
  """
  sample = sample_one_hot(log_potential, key=key, relaxation=None, axis=axis)
  softmax = functools.partial(jax.nn.softmax, axis=axis)
  p0 = softmax(log_potential)
  p0t = softmax(jnp.log(p0)/temperature)
  p1 = softmax(straight_through_replace(log_potential, jnp.log((sample+p0t)/2)))
  return straight_through_replace(2*p1 - p0/2, sample)


def gapped_straight_through(
    logits: jax.Array, *, key: jax.Array, axis: int|tuple[int, ...] = -1,
    temperature: float = 1.0, gap: float = 1., hard_sample: bool = True,
    ) -> jax.Array:
  """Gapped straight-through estimator.

  Similar to ST-Gumbel-Softmax but with a deterministic noise given the sampled
  one-hot vector. This should decrease the variance in the estimator.

  Args:
    logits: Logits of the distribution to sample from.
    key: Random key.
    axis: Axis over which to sample.
    temperature: Temperature for smoothing the gradients.
    gap: Gap to add to the logits when computing the deterministic perturbation.
    hard_sample: If True (default) returns the hard one-hot sample, if False
      returns the perturbed logits.
  Returns:
    Sampled one-hot vector that will be differentiable via Gapped ST estimator.

  Refernces:
    Fan et al (2022) -- https://proceedings.mlr.press/v162/fan22a/fan22a.pdf
  """
  logits = jax.nn.log_softmax(logits, axis=axis)
  sample = sample_one_hot(logits, key=key, relaxation=None, axis=axis)
  max_logits = jnp.max(logits, axis=axis, keepdims=True)
  # Adding m1 perturbation moves sampled logit up to be the highest score.
  m1 = (max_logits - logits)*sample
  # Subtracting m2 moves non-sampled ones lower than the sampled by a margin.
  m2 = jax.nn.relu(logits - max_logits + gap) * (1-sample)
  perturbation = jax.lax.stop_gradient(m1 - m2)
  soft_sample = tempered_softmax(
      logits + perturbation, temperature=temperature, axis=axis)
  if hard_sample:
    return straight_through_replace(soft_sample, sample)
  else:
    return soft_sample


def zgr(
    logits: jax.Array, *, key: jax.Array, axis: int|tuple[int, ...] = -1
    ) -> jax.Array:
  """Differentiable ZGR sampling from Shekhovtsov (2023).

  Args:
    logits: Logits of the distribution to sample from.
    key: Random key.
    axis: Axis over which to sample.
  Returns:
    Sampled one-hot vector that will be differentiable via ZGR estimator.

  It is equivalent to 1/2 of combination of straigh-trough estimator and
  DARN estimator (Gregor et al, 2014). It is also equivalent to Gumbel-Rao
  estimator (Paulus et al, 2021) when temperature approaches 0.

  References:
    Shekhovtsov (2023) -- https://openreview.net/pdf?id=9GjM8UzCYN
    Shekhovtsov implementation https://github.com/shekhovt/ZGR/blob/main/zgr.py
    Gregor et al (2014) https://proceedings.mlr.press/v32/gregor14.pdf
    Paulus et al (2021) -- https://openreview.net/pdf?id=Mk6PZtgAgfq
  """
  logp = jax.nn.log_softmax(logits, axis=axis)  # (..., d)
  p = jnp.exp(logp)  # (..., d)
  x = sample_one_hot(logp, axis=axis, key=key, relaxation=None)  # (..., d)
  logpx = (logp*x).sum(axis, keepdims=True)  # (... , 1) -- log probability of x
  dx_re = jax.lax.stop_gradient(x - p) * logpx  # (..., d)
  dx_st = p  # (..., d)
  dx = (dx_st + dx_re) / 2
  return x + (dx - jax.lax.stop_gradient(dx))


def zgr_binary(logits: jax.Array, *, key: jax.Array) -> jax.Array:
  """Differentiable sampling from Shekhovtsov (2023) for binary random vars.

  Args:
    logits: Logits of the distribution to sample from.
    key: Random key.
  Returns:
    Sampled binary values that will be differentiable via ZGR-binary estimator.

  ZGR specialized for binary random variables. In that special case this
  function is equivalent to 1/2 DARN estimator (Gregor et al, 2014) and to
  Gumbel-Rao estimator (Paulus et al, 2021) when temperature approaches 0.

  References:
    Shekhovtsov (2023) -- https://openreview.net/pdf?id=9GjM8UzCYN
    Shekhovtsov implementation https://github.com/shekhovt/ZGR/blob/main/zgr.py
    Gregor et al (2014) https://proceedings.mlr.press/v32/gregor14.pdf
    Paulus et al (2021) -- https://openreview.net/pdf?id=Mk6PZtgAgfq
  """
  p = jax.nn.sigmoid(logits)
  b = jax.random.bernoulli(key, p)
  jacobian = (b*(1-p)+(1-b)*p)/2
  sg = jax.lax.stop_gradient
  return b + sg(jacobian)*(logits - sg(logits))


def tempered_softmax(
    logits: jax.Array, *, temperature: float, axis: int|tuple[int, ...]
    ) -> jax.Array:
  """Softmax with temperature parameter."""
  return jax.nn.softmax(logits/temperature, axis=axis)


def posterior_gumbel(
    location: jax.Array, sample: jax.Array, *, key: jax.Array,
    axis: int|tuple[int, ...] = -1) -> jax.Array:
  """Samples a set of gumbels that produces the target sample given the logits.

  This function corresponds to Equation 9 from Paulus et al 2022.

  Args:
    location: Logits of the distribution to sample from.
    sample: Target one-hot sample for which returned gumbels should match.
    key: Random key.
    axis: Axis over which categorical distribution is defined.
  Returns:
    Gumbel noise that corresponds to the provided input sample. Note that these
    Gumbels are with their location. If you want them without location
    (i.e. location=0) you need to substract the provided location.


  References:
    Paulus et al 2021: https://openreview.net/pdf?id=Mk6PZtgAgfq
  """
  assert location.shape == sample.shape
  # pylint: disable=invalid-name
  logZ = jax.nn.logsumexp(location, axis=axis, keepdims=True)
  E = jax.random.exponential(key, location.shape)
  Ei = (sample * E).sum(axis=axis, keepdims=True)
  # pylint: enable=invalid-name
  return jnp.where(
      sample, -jnp.log(Ei) + logZ,
      -jnp.logaddexp(jnp.log(E)-location, jnp.log(Ei)-logZ))


def gumbel_rao(
    logits: jax.Array, *, key: jax.Array, k: int, temperature: float = 1.,
    axis: int|tuple[int, ...] = -1, use_scan: bool = False) -> jax.Array:
  """Differentiable Gumbel-Rao sampling.

  Args:
    logits: Logits of the distribution to sample from.
    key: Random key.
    k: Number of samples to take.
    temperature: Temperature for smoothing the gradients.
    axis: Axis over which to sample.
    use_scan: If True (default) uses a more memory efficient implementation.
        If False it is parallelized over all samples but less memory efficient.
  Returns:
    Sampled one-hot vector which is differentiable via Gumbel-Rao estimator.

  This gradient estimator is equivalent to Straight-Trough Gumbel-Softmax
  for k=1. For larger k it should decrease the variance of gradient estimator
  (but not the bias). This idea was published by Paulus et al (2020). The
  implementation here loosely follows Appendix A of arXiv version of the paper
  that is slightly different than the OpenReview one.

  Reference:
    Paulus et al (2020) -- https://arxiv.org/pdf/2010.04838#page=12
  """
  @jax.custom_vjp
  def _gumbel_rao_core(logits, key):
    key_fwd, _ = jax.random.split(key)
    return sample_one_hot(logits, key=key_fwd, relaxation=None, axis=axis)

  def _gumbel_rao_fwd(logits: jax.Array, key: jax.Array):
    return _gumbel_rao_core(logits, key), (logits, key)

  def _gumbel_rao_bwd(res, g):
    logits, key = res
    key_fwd, key_bwd = jax.random.split(key)
    sample = sample_one_hot(logits, key=key_fwd, relaxation=None, axis=axis)
    def single_gumbel(acc, akey):
      gumbels = posterior_gumbel(logits, sample=sample, key=akey, axis=axis)
      t_softmax = functools.partial(
          tempered_softmax, temperature=temperature, axis=axis)
      grad = jax.vjp(t_softmax, logits + gumbels)[1](g)[0]
      return acc+grad / k, None
    keys_bwd = jax.random.split(key_bwd, k)
    if use_scan:
      logit_grad = jax.lax.scan(
          jax.remat(single_gumbel), jnp.zeros_like(logits), keys_bwd)[0]
    else:
      logit_grad = jax.vmap(
          functools.partial(single_gumbel, 0))(keys_bwd)[0].sum(0)
    return logit_grad, None

  _gumbel_rao_core.defvjp(_gumbel_rao_fwd, _gumbel_rao_bwd)

  return _gumbel_rao_core(logits, key)


def sample_one_hot(
    log_potential: Array, *, key: Array, axis: Union[int, Tuple[int, ...]] = -1,
    temperature: float = 1.0, noise_scale: float = 1.0,
    relaxation: Optional[str] = None) -> Array:
  """Returns sampled vector from the input for a given key.

  Args:
    log_potential: Log-potentials to sample from.
    key: Random key.
    axis: Axis over which to sample.
    temperature: Temperature for smoothing (or sharpening) the soft samples.
    noise_scale: Scale of the noise to add to the log-potentials.
    relaxation: Relaxation to apply to the sampling so that sampling is
      differentiable. The available options are:
      - None: If nothing is provided (the default setting) non-differentiable
          1-hot sample is returned.
      - Gumbel-Softmax: Gumbel-Softmax sampling.
      - ST-Gumbel-Softmax: Straight-Through Gumbel-Softmax sampling.
      - ST-Argmax: Straight-Through Argmax sampling.
      - ReinMax: ReinMax sampling.
      - Gumbel-Rao-X : Gumbel-Rao sampling with X Gumbel noises for computing
          the posterior. E.g. Gumbel-Rao-128 for 128 Gumbel noises.
          Gumbel-Rao-1 is equivalent to ST-Gumbel-Softmax.
      - ZGR: ZGR sampling.
  Returns:
    One-hot sample from the input distribution (unless Gumbel-Softmax relaxation
    is used in which case it is a soft sample).
  """
  # Normalization is not important for Gumbel trick, but it is for temperature.
  log_potential = jax.nn.log_softmax(log_potential, axis=axis)  # Normalization.
  noise = jax.random.gumbel(key, log_potential.shape) * noise_scale

  t_softmax = functools.partial(
      tempered_softmax, temperature=temperature, axis=axis)

  if relaxation is None:  # Standard Gumbel-Argmax trick with no gradients.
    return max_one_hot(t_softmax(log_potential+noise), axis,
                       straight_through=False)
  elif relaxation == "ST-Gumbel-Softmax":
    return max_one_hot(t_softmax(log_potential+noise), axis,
                       straight_through=True)
  elif relaxation == "Gumbel-Softmax":
    return t_softmax(log_potential+noise)
  elif relaxation == "Gapped-ST":
    return gapped_straight_through(
        log_potential, key=key, axis=axis, temperature=temperature)
  elif relaxation == "ZGR":
    return zgr(log_potential, key=key, axis=axis)
  elif relaxation.startswith("Gumbel-Rao-"):
    k = int(relaxation.split("-")[-1])
    return gumbel_rao(
        log_potential, key=key, k=k, temperature=temperature, axis=axis)
  elif relaxation == "ReinMax":
    return reinmax_sample(
        log_potential, temperature=temperature, key=key, axis=axis)
  elif relaxation == "ST-Argmax":
    return max_one_hot(t_softmax(log_potential), axis, straight_through=True)
  else:
    raise NotImplementedError(f"Unknown relaxation: {relaxation}")


############################################################################
####  Shape utils
############################################################################


def asshape(shape: Union[Shape, int]) -> Shape:
  return (shape,) if isinstance(shape, int) else tuple(shape)


def shape_size(shape: Union[Shape, int]) -> int:
  return functools.reduce(operator.mul, asshape(shape), 1)


def vmap_ndim(f, ndim: int, in_axes: Union[int, None, Sequence[Any]] = 0):
  for _ in range(ndim):
    f = jax.vmap(f, in_axes=in_axes)
  return f


def grad_ndim(f, ndim: int, has_aux: bool = False):
  gf = eqx.filter_grad(f, has_aux=has_aux)
  gf = vmap_ndim(gf, ndim)
  return lambda *inputs: jtu.tree_map(jnp.nan_to_num, gf(*inputs))

is_shape = lambda x: isinstance(x, tuple) and all(isinstance(y, int) for y in x)

############################################################################
####  PyTree utils.
############################################################################

tadd = functools.partial(jtu.tree_map, jnp.add)
tsub = functools.partial(jtu.tree_map, jnp.subtract)
tlog = functools.partial(jtu.tree_map, safe_log)
tmul = functools.partial(jtu.tree_map, jnp.multiply)
tsum_all = lambda x: functools.reduce(jnp.add, map(jnp.sum, jtu.tree_leaves(x)))


def tscale_inexact_arrays(scalar: Union[float, Array], tree):
  if isinstance(scalar, float) and scalar == 1.:
    return tree
  else:
    return jtu.tree_map(lambda x: scalar * x if eqx.is_inexact_array(x) else x,
                        tree)


############################################################################
####  Alternative activations.
############################################################################


@jax.custom_jvp
def straight_through_replace(differentiable_input, non_differentiable_output):
  """Replaces value and passes trough the gradient."""
  shape_input = jtu.tree_map(lambda x: x.shape, differentiable_input)
  shape_output = jtu.tree_map(lambda x: x.shape, non_differentiable_output)
  if shape_input != shape_output:
    raise ValueError(
        f"Shapes for straight-through replacement of input {shape_input} "
        f"and output {shape_output} don't match.")
  return non_differentiable_output


def _straight_through_replace_jvp(primals, tangents):
  (tangent_of_differentiable_input, _) = tangents
  return straight_through_replace(*primals), tangent_of_differentiable_input


straight_through_replace.defjvp(_straight_through_replace_jvp)


def sparsemax(x: Array, axis: Union[int, Shape] = -1) -> Array:
  """Implements sparsemax activation from Martins and Astudillo (2016).

  Args:
    x: Input array.
    axis: Axis over which to apply sparsemax activation
  Returns:
    Array of the same size with target axis projected to probability simplex.
  References:
    Martins and Astudillo, 2016: http://proceedings.mlr.press/v48/martins16.pdf
  """

  @jax.custom_jvp
  def _sparsemax(x: Array) -> Array:
    # This sub-function projects the last axis to probability simplex.
    s = 1.0
    n_features = x.shape[-1]
    u = jnp.sort(x, axis=-1)[..., ::-1]
    cumsum_u = jnp.cumsum(u, axis=-1)
    ind = jnp.arange(n_features) + 1
    cond = s / ind + (u - cumsum_u / ind) > 0
    idx = jnp.count_nonzero(cond, axis=-1, keepdims=True)
    return jax.nn.relu(s/idx + (x-jnp.take_along_axis(cumsum_u, idx-1, -1)/idx))

  @_sparsemax.defjvp
  def _sparsemax_jvp(  # pylint: disable=g-one-element-tuple
      primals: Tuple[Array], tangents: Tuple[Array]) -> Tuple[Array, Array]:
    x, = primals
    x_dot, = tangents
    primal_out = _sparsemax(x)
    supp = primal_out > 0
    card = jnp.count_nonzero(supp, -1, keepdims=True)
    tangent_out = supp*x_dot - jnp.sum(supp*x_dot, -1, keepdims=True)/card*supp
    return primal_out, tangent_out

  if axis == -1:
    return _sparsemax(x)
  else:
    axis = asshape(axis)
    x1 = jnp.moveaxis(x, axis, range(-len(axis), 0))
    x2 = jnp.reshape(x1, x1.shape[:-len(axis)] + (-1,))
    x2b = _sparsemax(x2)
    x1b = jnp.reshape(x2b, x1.shape)
    xb = jnp.moveaxis(x1b, range(-len(axis), 0), axis)
    return xb
