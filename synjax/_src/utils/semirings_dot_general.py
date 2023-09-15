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

"""Flexible dot_general that supports arbitrary semirings."""

import functools
import inspect
import operator
from typing import Tuple, Sequence, Optional

import jax
import jax.numpy as jnp


DotDimensionNumbers = Tuple[Tuple[Sequence[int], Sequence[int]],
                            Tuple[Sequence[int], Sequence[int]]]


def build_dot_general(sum_fn, mul_op):
  """Constructs a dot_general function from arbitrary sum and multiplication.

  Note that this implementation will not optimize for using
  matrix-multiplication cores.

  Args:
    sum_fn: Function with the same interface as jnp.sum except that it
            optionally supports an additional key argument for randomness.
    mul_op: Binary operator that multiplies two JAX arrays.
  Returns:
    Function with the same interface as jax.lax.dot_general.
  """
  def add_unused_key_arg(fn):
    def fn2(*args, key=None, **kwargs):
      del key
      return fn(*args, **kwargs)
    return fn2
  spec = inspect.getfullargspec(sum_fn)
  if "key" not in (spec.args + spec.kwonlyargs):
    sum_fn = add_unused_key_arg(sum_fn)

  def matmul(lhs: jax.Array, rhs: jax.Array,
             key: Optional[jax.Array] = None) -> jax.Array:
    return sum_fn(mul_op(lhs[..., None], rhs[..., None, :, :]),
                  axis=-2, key=key)
  return dot_general_from_matmul(matmul)


def dot_general_from_matmul(matmul):
  """Constructs a dot_general function from matmul.

  Args:
    matmul: Function with same interface as jnp.matmul except for one additional
            key parameter for optional randomness.
  Returns:
    A new function with the same interface as jax.lax.dot_general
  """
  def dot_general(lhs: jax.Array, rhs: jax.Array,
                  dimension_numbers: DotDimensionNumbers, precision=None,
                  preferred_element_type=None,
                  key: Optional[jax.Array] = None) -> jax.Array:
    # This function will reorder axes of lhs and rhs so that they are
    # (*batch_dims, *lhs_other_dims, *contracting_dims) for lhs and
    # (*batch_dims, *contracting_dims, *rhs_other_dims) for rhs and
    # after that flatten the other_dims and contracting_dims so that
    # lhs has shape (*batch_dims, size(lhs_other_dims), size(contracting_dims))
    # rhs has shape (*batch_dims, size(contracting_dims), size(rhs_other_dims)).
    # This will make lhs and rhs in the right shape to apply matmul that will
    # broadcast over *batch_dims and contract the contracting_dim so that
    # result has shape (*batch_dims, size(lhs_other_dims), size(rhs_other_dims))
    # which is unflattened before returning to
    # (*batch_dims, *lhs_other_dims, *rhs_other_dims).
    del precision
    del preferred_element_type
    ((lhs_contracting_dims, rhs_contracting_dims),
     (lhs_batch_dims, rhs_batch_dims)) = dimension_numbers
    lhs_other_dims = [
        x for x in range(lhs.ndim)
        if x not in lhs_contracting_dims and x not in lhs_batch_dims]
    rhs_other_dims = [
        x for x in range(rhs.ndim)
        if x not in rhs_contracting_dims and x not in rhs_batch_dims]
    lhs_permutation = [*lhs_batch_dims, *lhs_other_dims, *lhs_contracting_dims]
    rhs_permutation = [*rhs_batch_dims, *rhs_contracting_dims, *rhs_other_dims]

    def product(xs):
      return functools.reduce(operator.mul, xs, 1)

    batch_shape = tuple(lhs.shape[x] for x in lhs_batch_dims)
    lhs_other_shape = tuple(lhs.shape[x] for x in lhs_other_dims)
    rhs_other_shape = tuple(rhs.shape[x] for x in rhs_other_dims)

    contracting_size = product(lhs.shape[x] for x in lhs_contracting_dims)

    lhs4matmul = jnp.transpose(lhs, lhs_permutation
                              ).reshape(*batch_shape, -1, contracting_size)
    rhs4matmul = jnp.transpose(rhs, rhs_permutation
                              ).reshape(*batch_shape, contracting_size, -1)
    res = matmul(lhs4matmul, rhs4matmul, key=key)
    return res.reshape(*batch_shape, *lhs_other_shape, *rhs_other_shape)
  return dot_general
