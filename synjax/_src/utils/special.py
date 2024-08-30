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


def bound(x: Array, min_val: float, max_val: float) -> Array:
  """Clipping that has a margin loss gradients for elements that are clipped.

  This is inspired Donahue et al (2019) who used an additional margin loss to
  penalize elements that are out of the specified range. This implementation
  differs in having gradients for the margin loss directly built-in so that
  the user does not have to manually pass the loss through the whole network.

  Args:
    x: Input array.
    min_val: Minimum value for the output array.
    max_val: Maximum value for the output array.
  Returns:
    Clipped array.

  References:
    Donahue et al (2019) -- Piano Genie -- Equation (1).
    https://arxiv.org/pdf/1810.05246#page=3
  """
  @jax.custom_vjp
  def f(y):
    return jnp.clip(y, min_val, max_val)

  def f_fwd(y):
    return f(y), y

  def f_bwd(res, g):
    y = res
    return jnp.where(y < min_val, -1, jnp.where(y > max_val, 1, g)),

  f.defvjp(f_fwd, f_bwd)
  return f(x)


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


def max_one_hot(x: Array, axis: Union[int, Tuple[int, ...]]) -> Array:
  return (x == x.max(axis=axis, keepdims=True)).astype(jnp.float32)


def sample_one_hot(logits: Array, *, key: Array,
                   axis: Union[int, Tuple[int, ...]] = -1,
                   relaxation: Optional[Literal["Gumbel-Softmax",
                                                "ST-Gumbel-Softmax"]]) -> Array:
  """Returns sampled vector from the input for a given key."""
  noise = jax.random.gumbel(key, logits.shape)
  perturbed = logits + noise
  if relaxation == "ST-Gumbel-Softmax":
    # ST-Gumbel-Softmax
    soft = jax.nn.softmax(perturbed, axis=axis)
    hard = max_one_hot(perturbed, axis)
    return straight_through_replace(soft, hard)
  elif relaxation == "Gumbel-Softmax":
    # Gumbel-Softmax
    return jax.nn.softmax(perturbed, axis=axis)
  elif relaxation is None:
    # Gumbel-Max
    return max_one_hot(perturbed, axis)
  else:
    raise NotImplementedError


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


def straight_through_replace(differentiable_input, non_differentiable_output):
  """Replaces value and passes trough the gradient."""
  if jtu.tree_map(lambda x: x.shape, differentiable_input) != jtu.tree_map(
      lambda x: x.shape, non_differentiable_output):
    raise ValueError("Shapes for straight-through replacement don't match.")
  return tadd(jax.lax.stop_gradient(tsub(non_differentiable_output,
                                         differentiable_input)),
              differentiable_input)


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
