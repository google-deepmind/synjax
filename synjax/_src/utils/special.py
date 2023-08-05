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
from typing import Union, Tuple, Optional, Literal
import equinox as eqx
import jax
import jax.numpy as jnp

from synjax._src import constants


Array = jax.Array
Shape = Tuple[int, ...]
KeyArray = jax.random.KeyArray
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


InversionMethod = Literal["solve", "qr"]


def inv(x: Array, *, inv_method: Optional[InversionMethod] = None,
        matmul_precision: Optional[jax.lax.Precision] = None,
        test_invertability: bool = True):
  """Matrix inverse with controlable precision, algorithm and invertability."""
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
  @jax.custom_vjp
  def slogdet_fn(y):
    sign, log_abs_det = jnp.linalg.slogdet(y, method=logdet_method)
    return sign, jnp.clip(jnp.nan_to_num(log_abs_det, neginf=-INF, posinf=INF),
                          -INF, INF)
  def slogdet_fn_fwd(y: Array) -> Tuple[Tuple[Array, Array], Array]:
    return slogdet_fn(y), y
  def slogdet_fn_bwd(y: Array, g) -> Tuple[Array]:
    inverse = inv(y, inv_method=inv_method, matmul_precision=matmul_precision,
                  test_invertability=test_invertability)
    return (jnp.einsum("...,...ij->...ji", g[1], inverse),)
  slogdet_fn.defvjp(slogdet_fn_fwd, slogdet_fn_bwd)
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
  max_val = jnp.max(x, axis=axis, keepdims=True)
  return jnp.where(x == max_val, x-jax.lax.stop_gradient(x)+1., 0.)


def sample_one_hot(logits: Array, *, key: KeyArray,
                   axis: Union[int, Tuple[int, ...]] = -1) -> Array:
  noise = jax.random.gumbel(key, logits.shape)
  return max_one_hot(logits + noise, axis)


############################################################################
####  Shape utils
############################################################################


def split_key_for_shape(key: KeyArray, shape):
  shape = asshape(shape)
  keys = jax.random.split(key, shape_size(shape))
  return keys.reshape(shape+key.shape)


def asshape(shape: Union[Shape, int]) -> Shape:
  return (shape,) if isinstance(shape, int) else tuple(shape)


def shape_size(shape: Union[Shape, int]) -> int:
  return functools.reduce(operator.mul, asshape(shape), 1)


def vmap_ndim(f, ndim: int):
  for _ in range(ndim):
    f = jax.vmap(f)
  return f


def grad_ndim(f, ndim: int, has_aux: bool = False):
  gf = eqx.filter_grad(f, has_aux=has_aux)
  gf = vmap_ndim(gf, ndim)
  return lambda *inputs: jax.tree_map(jnp.nan_to_num, gf(*inputs))
