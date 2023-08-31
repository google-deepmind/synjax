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

"""Definitions of semirings."""
import abc
import functools
import operator
from typing import Sequence, Optional, Union, Literal

import jax
import jax.numpy as jnp

# pylint: disable=g-importing-member
from synjax._src.config import get_config
from synjax._src.constants import INF
from synjax._src.utils import semirings_dot_general
from synjax._src.utils import semirings_einsum
from synjax._src.utils import special


Array = jax.Array
KeyArray = jax.random.KeyArray
Axis = Union[int, Sequence[int]]


def einsum_builder(sum_fn, mul_op):
  dot_general = semirings_dot_general.build_dot_general(sum_fn, mul_op)
  def einsum_fn(subscripts, *operands, key=None, **kwargs):
    return semirings_einsum.einsum_generalized(
        subscripts, *operands, key=key, sum_fn=sum_fn, mul_op=mul_op,
        dot_general=dot_general, **kwargs)
  if get_config().checkpoint_semiring_einsum:
    einsum_fn = jax.checkpoint(einsum_fn, static_argnums=(0,))
  return einsum_fn


def _wrap_fn_multi_axis_reduce(fn):
  """Extends support from single to multiple axes reduce."""
  def fn2(a, axis, *, key):
    if isinstance(axis, int):
      return fn(a, key=key, axis=axis)
    elif isinstance(axis, Sequence):
      reduce_axes = tuple(x%a.ndim for x in axis)
      other_axes = tuple(x for x in range(a.ndim) if x not in reduce_axes)
      a = jnp.transpose(a, other_axes+reduce_axes)
      a = jnp.reshape(a, a.shape[:len(other_axes)]+(-1,))
      return fn(a, key=key, axis=-1)
    else:
      raise ValueError(f"Axis cannot be of type {type(axis)}.")
  return fn2


class Semiring(metaclass=abc.ABCMeta):
  """Semiring interface."""

  def wrap(self, log_potentials: Array) -> Array:
    """Wraps raw log-potentials into their semi-ring form.

    For most semirings that form will be the same as in the standard
    log-potentials with only the difference in having an additional axis in
    the beginning. In top-k semiring this axis will have size k. In other
    semirings the size is 1. In effect, other semirings do not need this special
    axis but we keep it for the sake of having consistent shapes across
    semirings. The default implementation below covers all semirings except
    top-k semiring.

    Args:
      log_potentials: jnp.ndarray with log potentials.
    Returns:
      Log-potentials adapted to a particular semiring.
    """
    return jax.tree_map(lambda x: x[None], log_potentials)

  def unwrap(self, wrapped):
    """Reverses the effect of Semiring.wrap()."""
    return jax.tree_map(lambda x: x.squeeze(0), wrapped)

  def one(self, shape=()) -> Array:
    return self.wrap(jnp.zeros(shape))

  def zero(self, shape=()) -> Array:
    return self.wrap(jnp.full(shape, -INF))

  def mul(self, a: Array, b: Array, *cs: Array) -> Array:
    return functools.reduce(operator.add, [a, b, *cs])

  def add(self, a: Array, b: Array, *cs: Array, key: Optional[KeyArray] = None
          ) -> Array:
    return self.sum(jnp.stack((a, b, *cs), axis=1), axis=1, key=key)

  def sum(self, a: Array, axis: Axis, *, key: Optional[KeyArray] = None
          ) -> Array:
    raise NotImplementedError

  def einsum(self, subscripts: str, *operands, key: Optional[KeyArray] = None,
             **kwargs) -> Array:
    fn = einsum_builder(self.sum, self.mul)
    return fn(subscripts, *operands, key=key, **kwargs)


class LogSemiring(Semiring):
  """Implements the log-space semiring (logsumexp, +, -inf, 0).

  Gradients give marginals.
  """

  def sum(self, a: Array, axis: Axis, *, key: Optional[KeyArray] = None
          ) -> Array:
    return jax.nn.logsumexp(a, axis=axis)

  def add(self, a: Array, b: Array, *cs: Array, key: Optional[KeyArray] = None
          ) -> Array:
    return functools.reduce(jnp.logaddexp, [a, b, *cs])


class MaxSemiring(Semiring):
  """Implements the max semiring (max, +, -inf, 0) with optional smoothing.

  Gradients give argmax.
  """

  def __init__(
      self,
      smoothing: Optional[Literal["softmax", "st-softmax", "sparsemax"]] = None,
      temperature: float = 1.0):
    super().__init__()
    self.smoothing = smoothing
    self.temperature = temperature

  def sum(self, a: Array, axis: Axis, *, key: Optional[KeyArray] = None
          ) -> Array:
    if self.smoothing is None:
      return jnp.max(a, axis=axis)
    elif self.smoothing == "softmax":
      selection = jax.nn.softmax(a / self.temperature, axis=axis)
      return jnp.sum(selection * a, axis=axis)
    elif self.smoothing == "st-softmax":
      selection = special.straight_trough_replace(
          differentiable_input=jax.nn.softmax(a / self.temperature, axis=axis),
          non_differentiable_output=special.max_one_hot(a, axis=axis))
      return jnp.sum(selection * a, axis=axis)
    elif self.smoothing == "sparsemax":
      selection = special.sparsemax(a / self.temperature, axis=axis)
      return jnp.sum(selection * a, axis=axis)
    else:
      raise NotImplementedError

  def add(self, a: Array, b: Array, *cs: Array, key: Optional[KeyArray] = None
          ) -> Array:
    if self.smoothing:
      return super().add(a, b, *cs, key=key)
    else:
      return functools.reduce(jnp.maximum, [a, b, *cs])


class KBestSemiring(Semiring):
  """Implements semiring of which a gradient give a sample."""

  def __init__(self, k: int, approximate: bool):
    super().__init__()
    self.k = k
    self.approximate = approximate

  def wrap(self, log_potentials: Array) -> Array:
    x = jnp.full((self.k, *log_potentials.shape), -INF)
    x = x.at[0].set(log_potentials)
    return x

  def unwrap(self, wrapped):
    return wrapped

  def mul(self, a: Array, b: Array, *cs: Array) -> Array:
    for c in [b, *cs]:
      a = self.sum(a[:, None] + c[None], key=None, axis=1)
    return a

  def sum(self, a: Array, axis: Axis, *, key: Optional[KeyArray] = None
          ) -> Array:
    if self.k == 1:
      return MaxSemiring().sum(a, axis, key=key)
    else:
      fn = _wrap_fn_multi_axis_reduce(self._sum_single_axis)
      return fn(a, key=key, axis=axis)

  def _sum_single_axis(self, a: Array, key: KeyArray, axis: int) -> Array:
    """Computes sum within one axis only."""
    del key
    if self.approximate:
      a = jnp.moveaxis(a, axis, 1)  # Reduce axis should be SECOND.
      a = a.reshape(-1, *a.shape[2:])
      a = jax.lax.approx_max_k(a, k=self.k, reduction_dimension=0)[0]
    else:
      a = jnp.moveaxis(a, axis, -1)  # Reduce axis should be LAST.
      a = jnp.moveaxis(a, 0, -1)
      a = a.reshape(*a.shape[:-2], -1)
      a = jax.lax.top_k(a, k=self.k)[0]
      a = jnp.moveaxis(a, -1, 0)
    return a


class SamplingSemiring(Semiring):
  """Implements the semiring whose gradients provide samples."""

  def sum(self, a: Array, axis: Axis, *, key: Optional[KeyArray] = None
          ) -> Array:
    if key is None:
      raise ValueError("KeyArray cannot be None.")
    @jax.custom_gradient
    def _sum_sampling(a, key):
      def grad(g):
        g = jnp.expand_dims(g, axis)
        return special.sample_one_hot(a, axis=axis, key=key)*g, None
      return jax.nn.logsumexp(a, axis), grad
    return _sum_sampling(a, key)
